"""Regression of surprisal against reading times (step 5).

Functions take per-reader cleaned reading measures so ``participants`` and
``mode`` filters apply before aggregating to one value per word. The mixed-effects
model comparison is in ``model_comparison.py``; its tests in ``stats.py``.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from src.config import WORD_KEY


# ── filters ──────────────────────────────────────────────────────────────────
def _filter_domain(df, domain):
    return df if domain == "all" else df[df["text_domain"] == domain]


def _filter_participants(df, participants):
    """Keep experts / novices by ``is_expert`` (see data.add_expertise)."""
    if participants == "all":
        return df
    flag = 1 if participants == "experts" else 0
    return df[df["is_expert"] == flag]


# ── surprisal <-> reading time ───────────────────────────────────────────────
def merge_surprisal_rt(surprisal_df: pd.DataFrame, rt_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join per-word surprisal onto the per-reader reading measures."""
    return surprisal_df.merge(rt_df, on=WORD_KEY, how="inner")


def _aggregate_words(df, measure, mode):
    """One row per word: surprisal + aggregated reading time."""
    agg = (
        df.groupby(WORD_KEY)
        .agg(surprisal=("surprisal", "first"), rt=(measure, mode))
        .reset_index()
    )
    return agg.dropna(subset=["surprisal", "rt"])


def regress_rt(
    merged: pd.DataFrame, mode="mean", participants="all", domain="all", measure="TFT"
):
    """OLS reading time ~ surprisal. Returns the fitted statsmodels result."""
    df = _filter_domain(merged, domain)
    df = _filter_participants(df, participants)
    agg = _aggregate_words(df, measure, mode)
    X = sm.add_constant(agg["surprisal"])
    return sm.OLS(agg["rt"], X).fit()
