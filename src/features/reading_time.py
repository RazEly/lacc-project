"""Clean PoTeC reading times"""

from __future__ import annotations

import pandas as pd


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
    df = df[
        (df["word_index_in_text"] != grp_idx.transform("min"))
        & (df["word_index_in_text"] != grp_idx.transform("max"))
    ]

    # skipped words (RT 0)
    df = df[df[measure] > 0]

    # per-reader ±sd_k·SD fence around that reader's mean RT
    grp = df.groupby("reader_id")[measure]
    mean = grp.transform("mean")
    sd = grp.transform("std")
    lo, hi = mean - sd_k * sd, mean + sd_k * sd
    keep = sd.isna() | ((df[measure] >= lo) & (df[measure] <= hi))
    return df[keep].copy()
