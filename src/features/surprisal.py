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

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEFAULT_MODEL

WORD_KEY = ["text_id", "word_index_in_text"]


def load_causal_lm(name_or_path: str = DEFAULT_MODEL, attn: bool = False):
    """Load a causal LM + tokenizer for surprisal (and optionally attention).

    ``attn=True`` selects the eager attention implementation and turns on
    ``output_attentions`` so attention matrices are returned (step 3).
    Accepts an HF model id or a local fine-tuned checkpoint path.
    """
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=True)
    kwargs = {}
    if attn:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, output_attentions=attn, **kwargs
    )
    model.eval()
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
        lp = logprobs[pos - 1, sent_ids[j]].item() / math.log(2)
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
