"""Encoder (BERT) attention for the gaze-alignment experiment.

Reproduces the strongest result of Mouratidi & Poesio (2025): attention **flow**
from an encoder aligns with reading gaze better than raw attention or saliency,
and better than a decoder. We load a German BERT, extract flow attention with the
shared ``features.attention`` machinery (``decay=0`` — an encoder is
bidirectional, so the decoder early-token correction is switched off), then track
how domain MLM fine-tuning shifts that alignment.
"""
from __future__ import annotations

import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.config import ENCODER_MODEL
from src.features import attention as at

# encoder is bidirectional -> no early-token bias -> no Metzger position decay.
ENCODER_DECAY = 0.0


def load_encoder(name_or_path: str = ENCODER_MODEL, attn: bool = True):
    """Load a masked-LM encoder + tokenizer, with attentions on for flow."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    kwargs = {"attn_implementation": "eager"} if attn else {}
    model = AutoModelForMaskedLM.from_pretrained(
        name_or_path, output_attentions=attn, **kwargs
    )
    model.eval()
    return model, tokenizer


def extract_flow(words_df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Per-word per-layer attention-flow scores from an encoder (decay off)."""
    return at.extract_attention(
        words_df, model, tokenizer, method="flow", decay=ENCODER_DECAY
    )


def flow_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Re-extract flow attention with each fine-tuned checkpoint.

    Mirrors ``finetune.recompute_surprisal_over_checkpoints`` but for attention:
    returns the per-word per-layer flow table for every checkpoint, tagged with
    ``checkpoint`` / ``index`` / ``epoch`` / ``domain`` (``index`` 0 = baseline).
    """
    frames = []
    for _, row in manifest.iterrows():
        model, tok = load_encoder(row.checkpoint)
        flow = extract_flow(words_df, model, tok)
        flow["checkpoint"] = row.checkpoint
        flow["index"] = row["index"]
        flow["epoch"] = row.epoch
        flow["domain"] = row.domain
        frames.append(flow)
    return pd.concat(frames, ignore_index=True)
