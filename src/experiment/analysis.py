"""Attention-flow vs gaze correlation, split by reader expertise.

Reuses ``analysis.correlation`` (``build_et_table`` + ``correlate_attention``) to
score how well encoder attention flow tracks gaze, separately for domain experts
and novices (``is_expert`` = reader major matches text domain). One table per
layer for the baseline reproduction figure; the peak-layer summary for the
fine-tuning curve.
"""
from __future__ import annotations

import pandas as pd

from src.analysis import correlation as co

GROUPS = ("experts", "novices")


def correlate_flow_by_group(
    flow_df: pd.DataFrame, rm_raw: pd.DataFrame, domain: str = "all", feature: str = "pca"
) -> pd.DataFrame:
    """Per-layer flow↔gaze Spearman for experts and novices.

    ``flow_df`` is one model's flow table (``encoder.extract_flow``). Returns
    columns ``layer``, ``feature``, ``spearman``, ``p``, ``n``, ``group``.
    """
    rows = []
    for grp in GROUPS:
        et = co.build_et_table(rm_raw, domain=domain, participants=grp)
        corr = co.correlate_attention(flow_df, et, attention_method="flow")
        corr = corr[corr["feature"] == feature].copy()
        corr["group"] = grp
        rows.append(corr)
    return pd.concat(rows, ignore_index=True)


def flow_correlation_over_checkpoints(
    flow_versions: pd.DataFrame, rm_raw: pd.DataFrame, feature: str = "pca"
) -> pd.DataFrame:
    """Peak-layer flow↔gaze Spearman per checkpoint × reader group.

    ``flow_versions`` is ``encoder.flow_over_checkpoints`` output. Each checkpoint
    is correlated only against its own fine-tuning domain's texts (a physics model
    on physics texts), per expert/novice group, keeping the best-aligned layer.
    Returns ``index``, ``epoch``, ``domain``, ``group``, ``layer``, ``spearman``.
    """
    rows = []
    for (index, epoch, domain), fv in flow_versions.groupby(["index", "epoch", "domain"]):
        for grp in GROUPS:
            et = co.build_et_table(rm_raw, domain=domain, participants=grp)
            corr = co.correlate_attention(fv, et, attention_method="flow")
            corr = corr[corr["feature"] == feature]
            if corr["spearman"].notna().any():
                best = corr.loc[corr["spearman"].idxmax()]
                rows.append(
                    {
                        "index": index,
                        "epoch": epoch,
                        "domain": domain,
                        "group": grp,
                        "layer": int(best["layer"]),
                        "spearman": float(best["spearman"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["domain", "group", "index"])
