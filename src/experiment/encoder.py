"""Encoder (BERT) raw attention for the gaze-alignment experiment.

Following Mouratidi & Poesio (2025): raw (head-averaged) encoder attention vs
reading gaze (paper §3.2.1, Fig 1). We load a German BERT, extract raw attention
with the shared ``features.attention`` machinery, then track how domain MLM
fine-tuning shifts that alignment.
"""
from __future__ import annotations

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.config import ENCODER_MODEL
from src.features import attention as at


def load_encoder(name_or_path: str = ENCODER_MODEL, attn: bool = True):
    """Load a masked-LM encoder + tokenizer, with attentions on."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    kwargs = {"attn_implementation": "eager"} if attn else {}
    model = AutoModelForMaskedLM.from_pretrained(
        name_or_path, output_attentions=attn, **kwargs
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def extract_raw(words_df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Per-word per-layer raw (head-averaged) attention from an encoder.

    The paper reports raw attention (§3.2.1, Fig 1); cheap forward-only method.
    """
    return at.extract_attention(words_df, model, tokenizer, method="raw")


def attention_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame, method: str = "raw"
) -> pd.DataFrame:
    """Re-extract raw attention with each fine-tuned checkpoint.

    Mirrors ``finetune.recompute_surprisal_over_checkpoints`` but for attention:
    returns the per-word per-layer table for every checkpoint, tagged with
    ``checkpoint`` / ``index`` / ``epoch`` / ``tokens_seen`` / ``domain``
    (``index`` 0 = un-fine-tuned baseline, ``tokens_seen`` 0). ``tokens_seen`` is
    the fine-tuning x-axis: correlation is expected to rise with it.

    The step-0 baseline carries un-fine-tuned weights, so it is byte-identical
    across domains and its attention is computed once and reused — saving a
    corpus pass.
    """
    frames = []
    baseline = None  # index-0 weights are identical across domains
    for _, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc=f"{method} over checkpoints"
    ):
        if row["index"] == 0 and baseline is not None:
            scores = baseline.copy()
        else:
            model, tok = load_encoder(row.checkpoint)
            scores = extract_raw(words_df, model, tok)
            if row["index"] == 0:
                baseline = scores.copy()
        scores["checkpoint"] = row.checkpoint
        scores["index"] = row["index"]
        scores["epoch"] = row.epoch
        scores["tokens_seen"] = row.tokens_seen
        scores["domain"] = row.domain
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)
