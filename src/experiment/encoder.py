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
import torch
from tqdm.auto import tqdm
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
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def extract_raw(words_df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Per-word per-layer raw (head-averaged) attention from an encoder.

    The paper reports raw attention (§3.2.1, Fig 1) before attention flow; this is
    the cheap forward-only method, no max-flow.
    """
    return at.extract_attention(words_df, model, tokenizer, method="raw")


def extract_flow(words_df: pd.DataFrame, model, tokenizer, n_jobs: int = -1) -> pd.DataFrame:
    """Per-word per-layer attention-flow scores from an encoder (decay off).

    ``n_jobs`` cores run the (pure-CPU) max-flow across sentences (-1 = all).
    """
    return at.extract_attention(
        words_df, model, tokenizer, method="flow", decay=ENCODER_DECAY, n_jobs=n_jobs
    )


def attention_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame, method: str = "flow", n_jobs: int = -1
) -> pd.DataFrame:
    """Re-extract attention (``raw`` or ``flow``) with each fine-tuned checkpoint.

    Mirrors ``finetune.recompute_surprisal_over_checkpoints`` but for attention:
    returns the per-word per-layer table for every checkpoint, tagged with
    ``checkpoint`` / ``index`` / ``epoch`` / ``tokens_seen`` / ``domain``
    (``index`` 0 = un-fine-tuned baseline, ``tokens_seen`` 0). ``tokens_seen`` is
    the fine-tuning x-axis: correlation is expected to rise with it.

    ``n_jobs`` cores run the per-checkpoint max-flow (flow only; -1 = all). The
    step-0 baseline carries un-fine-tuned weights, so it is byte-identical across
    domains and its attention is computed once and reused — saving a corpus pass.
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
            scores = (
                extract_flow(words_df, model, tok, n_jobs=n_jobs)
                if method == "flow"
                else extract_raw(words_df, model, tok)
            )
            if row["index"] == 0:
                baseline = scores.copy()
        scores["checkpoint"] = row.checkpoint
        scores["index"] = row["index"]
        scores["epoch"] = row.epoch
        scores["tokens_seen"] = row.tokens_seen
        scores["domain"] = row.domain
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)


def flow_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame, n_jobs: int = -1
) -> pd.DataFrame:
    """Flow attention over every checkpoint (see ``attention_over_checkpoints``)."""
    return attention_over_checkpoints(words_df, manifest, method="flow", n_jobs=n_jobs)
