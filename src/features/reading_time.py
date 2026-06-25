"""Clean and aggregate PoTeC reading times (step 1).

Cleaning follows the plan:
  1. drop sentence-initial and sentence-final words (no left/right context);
  2. drop skipped words (reading time == 0);
  3. Smith & Levy (2013) outlier filtering — IQR rule on the reading-time
     distribution, applied per word so word-specific spread is respected.

Aggregation produces per-word mean / std / median reading times for several
reader groups (everyone, domain-experts only, and each expertise level).
"""
from __future__ import annotations

import pandas as pd

WORD_KEY = ["text_id", "word_index_in_text"]

# Human-readable labels for level_of_studies_numeric.
LEVEL_LABELS = {0: "undergraduate", 1: "graduate"}


def clean_reading_times(
    rm: pd.DataFrame,
    measure: str = "TFT",
    iqr_k: float = 1.5,
    by=("text_id", "word_index_in_text"),
    min_count: int = 4,
) -> pd.DataFrame:
    """Return ``rm`` with sentence-edge words, skips, and IQR outliers removed.

    Parameters
    ----------
    measure   : reading-time column to clean against (e.g. "TFT", "FPRT").
    iqr_k     : whisker length for the IQR rule (1.5 = Tukey fence).
    by        : grouping for the IQR fence; per-word by default.
    min_count : groups with fewer non-skipped observations are left unfiltered
                (too few points to estimate a sensible fence).
    """
    df = rm

    # 1. sentence-initial / sentence-final words.
    df = df[(df["is_sent_beginning"] != 1) & (df["is_sent_end"] != 1)]

    # 2. skipped words: reading time 0 (equivalently Fix == 0).
    df = df[df[measure] > 0]

    # 3. Smith & Levy (2013) IQR fence, per group.
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
    """Aggregate cleaned reading times into per-word tables for reader groups.

    Returns a dict of DataFrames keyed by group name:
      - ``"all"``        : every reader (baseline);
      - ``"experts"``    : domain experts only
                           (``is_expert == 1``, i.e. reader major == text domain);
      - ``"experts_<level>"`` : experts split by expertise level
                           (undergraduate / graduate).
    Each table has columns ``text_id``, ``word_index_in_text``,
    ``mean_<m>``, ``std_<m>``, ``median_<m>``, ``n``.
    """
    groups: dict[str, pd.DataFrame] = {"all": _agg(rm, measure)}

    experts = rm[rm["is_expert"] == 1]
    groups["experts"] = _agg(experts, measure)

    for lvl, label in LEVEL_LABELS.items():
        sub = experts[experts["level_of_studies_numeric"] == lvl]
        if len(sub):
            groups[f"experts_{label}"] = _agg(sub, measure)

    return groups
