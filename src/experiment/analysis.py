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


def correlate_by_group(
    attn_df: pd.DataFrame,
    rm_raw: pd.DataFrame,
    method: str = "flow",
    domain: str = "all",
    feature: str = "pca",
) -> pd.DataFrame:
    """Per-layer attention↔gaze Spearman for experts and novices.

    ``attn_df`` is one model's attention table (``encoder.extract_raw`` /
    ``extract_flow``); ``method`` selects which (``raw``/``flow``). Returns columns
    ``layer``, ``feature``, ``spearman``, ``p``, ``n``, ``group``.
    """
    rows = []
    for grp in GROUPS:
        et = co.build_et_table(rm_raw, domain=domain, participants=grp)
        corr = co.correlate_attention(attn_df, et, attention_method=method)
        corr = corr[corr["feature"] == feature].copy()
        corr["group"] = grp
        rows.append(corr)
    return pd.concat(rows, ignore_index=True)


# back-compat alias: the flow baseline figure used this name.
def correlate_flow_by_group(
    flow_df: pd.DataFrame, rm_raw: pd.DataFrame, domain: str = "all", feature: str = "pca"
) -> pd.DataFrame:
    """Per-layer flow↔gaze Spearman for experts and novices (see correlate_by_group)."""
    return correlate_by_group(flow_df, rm_raw, method="flow", domain=domain, feature=feature)


def correlate_by_feature(
    attn_df: pd.DataFrame, rm_raw: pd.DataFrame, method: str = "raw", domain: str = "all"
) -> pd.DataFrame:
    """Per-layer attention↔gaze Spearman for every eye-tracking feature (all readers).

    Direct analog of the paper's Figure 1 (layer-wise raw-attention correlation
    across the eye-tracking features). Returns ``layer``, ``feature``,
    ``attention_method``, ``spearman``, ``p``, ``n``.
    """
    et = co.build_et_table(rm_raw, domain=domain, participants="all")
    return co.correlate_attention(attn_df, et, attention_method=method)


def correlation_over_checkpoints(
    versions: pd.DataFrame, rm_raw: pd.DataFrame, method: str = "flow", feature: str = "pca"
) -> pd.DataFrame:
    """Peak-layer attention↔gaze Spearman per checkpoint × reader group.

    ``versions`` is ``encoder.attention_over_checkpoints`` output (``raw``/``flow``
    selected by ``method``). Each checkpoint is correlated only against its own
    fine-tuning domain's texts (a physics model on physics texts), per
    expert/novice group, keeping the best-aligned layer. ``tokens_seen`` is the
    fine-tuning x-axis. Returns ``index``, ``epoch``, ``tokens_seen``, ``domain``,
    ``group``, ``layer``, ``spearman``, ``method``.
    """
    rows = []
    for (index, epoch, tokens_seen, domain), fv in versions.groupby(
        ["index", "epoch", "tokens_seen", "domain"]
    ):
        for grp in GROUPS:
            et = co.build_et_table(rm_raw, domain=domain, participants=grp)
            corr = co.correlate_attention(fv, et, attention_method=method)
            corr = corr[corr["feature"] == feature]
            if corr["spearman"].notna().any():
                best = corr.loc[corr["spearman"].idxmax()]
                rows.append(
                    {
                        "index": index,
                        "epoch": epoch,
                        "tokens_seen": tokens_seen,
                        "domain": domain,
                        "group": grp,
                        "layer": int(best["layer"]),
                        "spearman": float(best["spearman"]),
                        "method": method,
                    }
                )
    return pd.DataFrame(rows).sort_values(["domain", "group", "tokens_seen"])


def flow_correlation_over_checkpoints(
    flow_versions: pd.DataFrame, rm_raw: pd.DataFrame, feature: str = "pca"
) -> pd.DataFrame:
    """Flow↔gaze peak-layer Spearman per checkpoint (see correlation_over_checkpoints)."""
    return correlation_over_checkpoints(flow_versions, rm_raw, method="flow", feature=feature)
