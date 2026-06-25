"""Plots for the encoder attention experiment — expert vs novice gaze alignment."""
from __future__ import annotations

import matplotlib.pyplot as plt


def expert_novice_layer_curve(corr_df, ax=None):
    """Per-layer flow↔gaze Spearman, one line per reader group (baseline model).

    ``corr_df`` is ``analysis.correlate_flow_by_group`` output. Reproduces the
    paper's layer curve, split by expertise: does encoder attention flow align
    more with experts' or novices' reading?
    """
    ax = ax or plt.gca()
    for group, g in corr_df.groupby("group"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["spearman"], marker="o", label=group)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="layer", ylabel="Spearman r (flow vs gaze PCA)")
    ax.legend(title="readers")
    return ax


def expert_novice_finetune_curve(curve_df, ax=None):
    """Peak-layer flow↔gaze Spearman vs fine-tuning epoch, per group × domain.

    ``curve_df`` is ``analysis.flow_correlation_over_checkpoints`` output. Shows
    whether domain MLM fine-tuning shifts attention alignment, and whether the
    shift differs for experts vs novices.
    """
    ax = ax or plt.gca()
    for (group, domain), g in curve_df.groupby(["group", "domain"]):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g["spearman"], marker="o", label=f"{group}·{domain}")
    ax.set(xlabel="fine-tuning epoch", ylabel="peak Spearman r (flow vs gaze PCA)")
    ax.legend(title="readers·domain")
    return ax
