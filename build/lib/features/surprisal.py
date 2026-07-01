"""Model surprisal from a decoder-only causal LM (step 2).

Word surprisal = sum of sub-token surprisals (-log2 p(token | left context)),
aligned via the tokenizer's ``word_ids``. Each text is scored as one sequence so
context spans sentence boundaries (Eq. 1). An optional ``prompt`` prepends
domain-matched context.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEFAULT_MODEL, WORD_KEY


def _load_tokenizer(name_or_path: str):
    # add_prefix_space (BPE/german-gpt2): first word tokenizes like a mid-sentence
    # word. SentencePiece (Llama) adds it itself and may reject the kwarg.
    try:
        return AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=True)
    except (TypeError, ValueError):
        return AutoTokenizer.from_pretrained(name_or_path)


def load_causal_lm(name_or_path: str = DEFAULT_MODEL):
    """Load a causal LM + tokenizer for surprisal.

    Accepts an HF model id, a full fine-tuned checkpoint, or a LoRA (PEFT) adapter
    checkpoint — the latter is detected by ``adapter_config.json`` and folded onto
    its base model so callers get a plain merged model either way.
    """
    kwargs = {}
    # fp16 on GPU: halves VRAM, inference-safe (log_softmax upcasts via .float()).
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    if (Path(name_or_path) / "adapter_config.json").is_file():
        # LoRA checkpoint: load the adapter's base model, match embedding size,
        # then merge the adapter so the forward needs no PEFT machinery.
        from peft import PeftConfig, PeftModel

        base = PeftConfig.from_pretrained(name_or_path).base_model_name_or_path
        tokenizer = _load_tokenizer(base)
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, name_or_path).merge_and_unload()
    else:
        tokenizer = _load_tokenizer(name_or_path)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model.eval()
    # from_pretrained leaves the model on CPU; move to GPU once here.
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def _resolve_prompt(prompt, domain: str | None) -> str | None:
    """Allow ``prompt`` to be None, a single string, or a {domain: str} dict."""
    if prompt is None:
        return None
    if isinstance(prompt, dict):
        return prompt.get(domain)
    return prompt


@torch.no_grad()
def score_words(word_list, model, tokenizer, prompt=None, domain=None):
    """Per-word surprisal (bits) for an ordered word sequence (a full text).

    Returns a list the same length as ``word_list``; the first word is ``None``
    when it has no left context (no prompt prepended).
    """
    enc = tokenizer(word_list, is_split_into_words=True, return_tensors="pt")
    sent_ids = enc.input_ids[0]
    word_ids = enc.word_ids(0)

    prompt_text = _resolve_prompt(prompt, domain)
    if prompt_text:
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        input_ids = torch.cat([prompt_ids, sent_ids])
        offset = len(prompt_ids)
    else:
        input_ids = sent_ids
        offset = 0

    input_ids = input_ids.to(next(model.parameters()).device)
    logits = model(input_ids.unsqueeze(0)).logits[0]  # [seq, vocab]
    logprobs = torch.log_softmax(logits.float(), dim=-1)  # natural log

    word_bits: dict[int, float] = {}
    for j, wid in enumerate(word_ids):
        if wid is None:
            continue
        pos = offset + j  # absolute position of this sentence token
        if pos == 0:
            continue  # first token, no predictor -> first word NaN
        # log2 p(token_pos | tokens < pos) = logprobs[pos-1, token_pos]
        lp = logprobs[pos - 1, int(sent_ids[j])].item() / math.log(2)
        word_bits[wid] = word_bits.get(wid, 0.0) - lp

    return [word_bits.get(i) for i in range(len(word_list))]


def compute_surprisal(
    words_df: pd.DataFrame, model, tokenizer, prompt=None
) -> pd.DataFrame:
    """Per-word surprisal for every PoTeC text.

    Each text is rebuilt as one sequence (ordered by ``word_index_in_text``) so
    context carries across sentences (Eq. 1). The text's first/last word is
    dropped to match the reading-time cleaning. ``prompt`` may be None, a string,
    or a ``{text_domain: str}`` dict.

    Returns columns: ``text_id``, ``word_index_in_text``, ``surprisal``.
    """
    rows = []
    for text_id, text in words_df.sort_values(
        ["text_id", "word_index_in_text"]
    ).groupby("text_id", sort=False):
        domain = text["text_domain"].iloc[0]
        words = text["word"].fillna("").astype(str).tolist()
        bits = score_words(words, model, tokenizer, prompt=prompt, domain=domain)
        idx = text["word_index_in_text"].tolist()
        last = len(words) - 1
        for k, (wi, b) in enumerate(zip(idx, bits)):
            if b is None or k == last:  # text first (no context) / last word
                continue
            rows.append((text_id, wi, b))

    return pd.DataFrame(rows, columns=WORD_KEY + ["surprisal"])
