"""Plots for the encoder attention experiment — expert vs novice gaze alignment."""
from __future__ import annotations

import matplotlib.pyplot as plt


def expert_novice_layer_curve(corr_df, ax=None, method="flow"):
    """Per-layer attention↔gaze Spearman, one line per reader group (baseline).

    ``corr_df`` is ``analysis.correlate_by_group`` output. Reproduces the paper's
    layer curve, split by expertise: does encoder ``method`` attention align more
    with experts' or novices' reading?
    """
    ax = ax or plt.gca()
    for group, g in corr_df.groupby("group"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["spearman"], marker="o", label=group)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="layer", ylabel=f"Spearman r ({method} vs gaze PCA)")
    ax.legend(title="readers")
    return ax


def feature_layer_curve(corr_df, ax=None, method="raw"):
    """Per-layer attention↔gaze Spearman, one line per eye-tracking feature.

    ``corr_df`` is ``analysis.correlate_by_feature`` output. Direct analog of the
    paper's Figure 1: how each eye-tracking measure correlates with the model's
    ``method`` attention across layers (all readers).
    """
    ax = ax or plt.gca()
    for feat, g in corr_df.groupby("feature"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["spearman"], marker="o", label=feat)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="layer", ylabel=f"Spearman r ({method} vs gaze)")
    ax.legend(title="ET feature", fontsize="small", ncol=2)
    return ax


def expert_novice_finetune_curve(curve_df, ax=None, method="flow"):
    """Peak-layer attention↔gaze Spearman vs fine-tuning tokens, per group × domain.

    ``curve_df`` is ``analysis.correlation_over_checkpoints`` output. x-axis is
    tokens seen during DAPT (``index`` 0 / 0 tokens = baseline anchor); ideally
    correlation rises as the encoder is fine-tuned. Shows whether the shift
    differs for experts vs novices.
    """
    ax = ax or plt.gca()
    for (group, domain), g in curve_df.groupby(["group", "domain"]):
        g = g.sort_values("tokens_seen")
        ax.plot(g["tokens_seen"], g["spearman"], marker="o", label=f"{group}·{domain}")
    ax.set(
        xlabel="tokens seen (fine-tuning)",
        ylabel=f"peak Spearman r ({method} vs gaze PCA)",
    )
    ax.legend(title="readers·domain")
    return ax


def expert_novice_regression_curve(reg_df, ax=None, method="raw"):
    """Predictive gain (held-out adjusted R²) attention adds over text features.

    ``reg_df`` is ``analysis.regression_over_checkpoints`` output. y is ``delta_r2``
    = R²(text + ``method`` attention) − R²(text-only baseline) on the 20% held-out
    split; x is fine-tuning tokens. One line per group × domain. Points above the
    grey zero line mean attention predicts gaze beyond word frequency/length;
    ideally the gain grows as the encoder adapts to the domain.
    """
    ax = ax or plt.gca()
    for (group, domain), g in reg_df.groupby(["group", "domain"]):
        g = g.sort_values("tokens_seen")
        ax.plot(g["tokens_seen"], g["delta_r2"], marker="o", label=f"{group}·{domain}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(
        xlabel="tokens seen (fine-tuning)",
        ylabel=f"Δ adjusted R² (text + {method} − text-only)",
    )
    ax.legend(title="readers·domain")
    return ax
