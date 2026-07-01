"""Clean and aggregate PoTeC reading times (step 1).

Cleaning follows Škrjanec & Demberg (2026): drop the text's first/last word (no
left/right context), skips (RT == 0), and per-reader ±3 SD outliers.
"""
from __future__ import annotations

import pandas as pd

from src.config import WORD_KEY

LEVEL_LABELS = {0: "undergraduate", 1: "graduate"}


def clean_reading_times(
    rm: pd.DataFrame,
    measure: str = "TFT",
    sd_k: float = 3.0,
) -> pd.DataFrame:
    """Return ``rm`` with text-edge words, skips, and per-reader outliers removed.

    ``sd_k`` = number of standard deviations around each reader's mean RT beyond
    which a data point is dropped (paper uses 3).
    """
    df = rm

    # first / last word of each text (no preceding / following context)
    grp_idx = df.groupby("text_id")["word_index_in_text"]
    df = df[(df["word_index_in_text"] != grp_idx.transform("min"))
            & (df["word_index_in_text"] != grp_idx.transform("max"))]

    # skipped words (RT 0, equivalently Fix == 0)
    df = df[df[measure] > 0]

    # per-reader ±sd_k·SD fence around that reader's mean RT
    grp = df.groupby("reader_id")[measure]
    mean = grp.transform("mean")
    sd = grp.transform("std")
    lo, hi = mean - sd_k * sd, mean + sd_k * sd
    keep = sd.isna() | ((df[measure] >= lo) & (df[measure] <= hi))
    return df[keep].copy()


def _agg(rm: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Per-word mean / std / median / n of ``measure`` across readers."""
    out = (rm.groupby(WORD_KEY)[measure]
             .agg(mean="mean", std="std", median="median", n="size")
             .reset_index())
    return out.rename(columns={
        "mean": f"mean_{measure}",
        "std": f"std_{measure}",
        "median": f"median_{measure}",
    })


def aggregate_rt(rm: pd.DataFrame, measure: str = "TFT") -> dict[str, pd.DataFrame]:
    """Per-word reading-time tables keyed by reader group.

    Groups: ``all``, ``experts`` (is_expert == 1), ``experts_<level>`` (by study
    level). Each table: WORD_KEY + ``mean_<m>`` / ``std_<m>`` / ``median_<m>`` / ``n``.
    """
    groups: dict[str, pd.DataFrame] = {"all": _agg(rm, measure)}

    experts = rm[rm["is_expert"] == 1]
    groups["experts"] = _agg(experts, measure)

    for lvl, label in LEVEL_LABELS.items():
        sub = experts[experts["level_of_studies_numeric"] == lvl]
        if len(sub):
            groups[f"experts_{label}"] = _agg(sub, measure)

    return groups
