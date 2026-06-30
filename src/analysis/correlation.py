"""Correlation and regression of surprisal against reading times (step 5).

Functions take per-reader cleaned reading measures so ``participants`` and
``mode`` filters apply before aggregating to one value per word. The mixed-effects
model comparison is in ``model_comparison.py``; its tests in ``stats.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

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


def _filter_domain_only(df, domain_only):
    if not domain_only:
        return df
    return df[
        (df["is_expert_technical_term"] == 1) | (df["is_general_technical_term"] == 1)
    ]


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


def correlate_surprisal(
    merged: pd.DataFrame,
    domain="all",
    domain_only=False,
    mode="mean",
    participants="all",
    measure="TFT",
) -> dict:
    """Pearson + Spearman of surprisal vs reading time.

    Filters by ``domain`` / ``participants`` / ``domain_only`` (technical terms),
    then aggregates RT per word with ``mode`` before correlating.
    """
    df = _filter_domain(merged, domain)
    df = _filter_participants(df, participants)
    df = _filter_domain_only(df, domain_only)
    agg = _aggregate_words(df, measure, mode)
    if len(agg) < 3:
        return {
            "n": len(agg),
            "pearson": np.nan,
            "pearson_p": np.nan,
            "spearman": np.nan,
            "spearman_p": np.nan,
        }
    r, rp = pearsonr(agg["surprisal"], agg["rt"])
    rho, sp = spearmanr(agg["surprisal"], agg["rt"])
    return {
        "n": len(agg),
        "pearson": r,
        "pearson_p": rp,
        "spearman": rho,
        "spearman_p": sp,
    }


def regress_rt(
    merged: pd.DataFrame, mode="mean", participants="all", domain="all", measure="TFT"
):
    """OLS reading time ~ surprisal. Returns the fitted statsmodels result."""
    df = _filter_domain(merged, domain)
    df = _filter_participants(df, participants)
    agg = _aggregate_words(df, measure, mode)
    X = sm.add_constant(agg["surprisal"])
    return sm.OLS(agg["rt"], X).fit()


# ── fine-tuning progress: correlation per checkpoint ─────────────────────────
def correlation_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    domain_only=False,
    mode="mean",
    measure="TFT",
    groups=("experts", "novices"),
) -> pd.DataFrame:
    """Surprisal-RT correlation per fine-tuning checkpoint × reader group.

    ``surp_versions`` from ``finetune.recompute_surprisal_over_checkpoints``. Each
    checkpoint is correlated only against texts of its own fine-tuning domain
    (physics model -> physics texts). Long-form columns: ``epoch``, ``domain``,
    ``group``, ``pearson``, ``spearman``, ``n``.
    """
    rows = []
    for (epoch, model_domain), sv in surp_versions.groupby(["epoch", "domain"]):
        merged = merge_surprisal_rt(sv[WORD_KEY + ["surprisal"]], rt_df)
        for grp in groups:
            r = correlate_surprisal(
                merged,
                domain=model_domain,
                domain_only=domain_only,
                mode=mode,
                participants=grp,
                measure=measure,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "domain": model_domain,
                    "group": grp,
                    "pearson": r["pearson"],
                    "spearman": r["spearman"],
                    "n": r["n"],
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "epoch"])
