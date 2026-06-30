"""Model surprisal from a decoder-only causal LM (step 2).

Word-level surprisal = sum of token surprisals (-log2 p(token | left context))
over the sub-tokens of a word. Sub-token -> word alignment uses the tokenizer's
``word_ids``. The loader stays model-agnostic within the decoder family, so the
default HF model or a fine-tuned local checkpoint can be plugged in.

An optional ``prompt`` prepends domain-matched context (a system prompt) before
the sentence; only the sentence words receive surprisal scores.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEFAULT_MODEL

WORD_KEY = ["text_id", "word_index_in_text"]


def _load_tokenizer(name_or_path: str):
    # add_prefix_space is a BPE concern (german-gpt2): it makes the first word
    # tokenize like a mid-sentence word. Llama/SentencePiece tokenizers add the
    # leading space themselves and may reject the kwarg, so fall back without it.
    try:
        return AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=True)
    except (TypeError, ValueError):
        return AutoTokenizer.from_pretrained(name_or_path)


def load_causal_lm(name_or_path: str = DEFAULT_MODEL, attn: bool = False):
    """Load a causal LM + tokenizer for surprisal (and optionally attention).

    ``attn=True`` selects the eager attention implementation and turns on
    ``output_attentions`` so attention matrices are returned (step 3).
    Accepts an HF model id, a full fine-tuned checkpoint, or a LoRA (PEFT) adapter
    checkpoint — the latter is detected by ``adapter_config.json`` and folded onto
    its base model so callers get a plain merged model either way.
    """
    kwargs = {}
    if attn:
        kwargs["attn_implementation"] = "eager"
    # fp16 weights on the GPU: halves VRAM (12 GB box) and speeds the forward.
    # Inference only — log_softmax upcasts via .float() (and attention is read
    # post-softmax), so this is numerically safe. Training stays fp32 (Trainer
    # adds its own mixed precision).
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    if (Path(name_or_path) / "adapter_config.json").is_file():
        # LoRA checkpoint: load the base model named in the adapter config, match
        # its training-time embedding size, then attach + merge the adapter so the
        # forward pass needs no PEFT machinery.
        from peft import PeftConfig, PeftModel

        base = PeftConfig.from_pretrained(name_or_path).base_model_name_or_path
        tokenizer = _load_tokenizer(base)
        model = AutoModelForCausalLM.from_pretrained(
            base, output_attentions=attn, **kwargs
        )
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, name_or_path).merge_and_unload()
    else:
        tokenizer = _load_tokenizer(name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, output_attentions=attn, **kwargs
        )
    model.eval()
    # Use the GPU when present: the per-sentence forward passes dominate runtime,
    # and load_*_pretrained leaves the model on CPU otherwise. extract_attention /
    # sentence_surprisal move their inputs to model.device, so this is the single
    # placement point for the whole surprisal+attention path.
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
def sentence_surprisal(word_list, model, tokenizer, prompt=None, domain=None):
    """Per-word surprisal (bits) for one sentence given as a list of words.

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
    """Per-word surprisal for every PoTeC sentence.

    Sentences are reconstructed from ``words_df`` (one row per word, ordered by
    ``word_index_in_sent``). The first and last word of each sentence are dropped
    (no usable left context / sentence-final punctuation effects), matching the
    reading-time cleaning. ``prompt`` may be None, a string, or a
    ``{text_domain: str}`` dict for domain-matched prompting.

    Returns columns: ``text_id``, ``word_index_in_text``, ``surprisal``.
    """
    rows = []
    for (text_id, _sent), sent in words_df.sort_values(
        ["text_id", "sent_index_in_text", "word_index_in_sent"]
    ).groupby(["text_id", "sent_index_in_text"], sort=False):
        domain = sent["text_domain"].iloc[0]
        words = sent["word"].fillna("").astype(str).tolist()
        bits = sentence_surprisal(words, model, tokenizer, prompt=prompt, domain=domain)
        idx = sent["word_index_in_text"].tolist()
        is_beg = sent["is_sent_beginning"].tolist()
        is_end = sent["is_sent_end"].tolist()
        for k, (wi, b) in enumerate(zip(idx, bits)):
            if b is None or is_beg[k] == 1 or is_end[k] == 1:
                continue
            rows.append((text_id, wi, b))

    return pd.DataFrame(rows, columns=WORD_KEY + ["surprisal"])
