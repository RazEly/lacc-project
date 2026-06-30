"""Clean and aggregate PoTeC reading times (step 1).

Cleaning: drop sentence-edge words (no left/right context), skips (RT == 0), and
Smith & Levy (2013) IQR outliers (per-word fence).
"""
from __future__ import annotations

import pandas as pd

from src.config import WORD_KEY

LEVEL_LABELS = {0: "undergraduate", 1: "graduate"}


def clean_reading_times(
    rm: pd.DataFrame,
    measure: str = "TFT",
    iqr_k: float = 1.5,
    by=("text_id", "word_index_in_text"),
    min_count: int = 4,
) -> pd.DataFrame:
    """Return ``rm`` with sentence-edge words, skips, and IQR outliers removed.

    ``iqr_k`` = Tukey fence whisker length; ``by`` = IQR grouping (per-word);
    ``min_count`` = groups below this many points skip the fence (too few to estimate).
    """
    df = rm

    # sentence-initial / sentence-final words
    df = df[(df["is_sent_beginning"] != 1) & (df["is_sent_end"] != 1)]

    # skipped words (RT 0, equivalently Fix == 0)
    df = df[df[measure] > 0]

    # Smith & Levy (2013) IQR fence, per group
    grp = df.groupby(list(by))[measure]
    q1 = grp.transform("quantile", 0.25)
    q3 = grp.transform("quantile", 0.75)
    n = grp.transform("size")
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    keep = (n < min_count) | ((df[measure] >= lo) & (df[measure] <= hi))
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
