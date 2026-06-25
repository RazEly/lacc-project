"""Plots for the surprisal / attention / fine-tuning results (step 6).

Each function takes prepared DataFrames (from ``analysis`` / ``finetune``) and
returns a matplotlib Axes so the caller can save or compose figures.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def surprisal_scatter(agg_df, x="surprisal", y="rt", bins=15, ax=None):
    """Binned surprisal-vs-reading-time scatter with an OLS fit line."""
    ax = ax or plt.gca()
    sns.regplot(data=agg_df, x=x, y=y, x_bins=bins, ax=ax)
    ax.set(xlabel="surprisal (bits)", ylabel="reading time (ms)")
    return ax


def attention_layer_curve(corr_table, feature="pca", ax=None):
    """Per-layer Spearman curve for one eye-tracking feature, raw vs flow."""
    ax = ax or plt.gca()
    sub = corr_table[corr_table["feature"] == feature]
    for method, g in sub.groupby("attention_method"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["spearman"], marker="o", label=method)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="layer", ylabel=f"Spearman r ({feature})")
    ax.legend(title="attention")
    return ax


def expert_novice_slopes(slopes_df, ax=None):
    """Bar chart of regression slope (ms/bit) per reader group."""
    ax = ax or plt.gca()
    sns.barplot(data=slopes_df, x="group", y="slope", ax=ax)
    ax.set(xlabel="reader group", ylabel="slope (ms/bit)")
    return ax


def finetune_correlation_curve(curve_df, metric="pearson", ax=None):
    """Surprisal-RT correlation vs fine-tuning epoch, one line per reader group.

    ``curve_df`` is the output of ``analysis.correlation_over_epochs``.
    """
    ax = ax or plt.gca()
    for group, g in curve_df.groupby("group"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g[metric], marker="o", label=group)
    ax.set(xlabel="fine-tuning epoch", ylabel=f"{metric} r (surprisal vs RT)")
    ax.legend(title="readers")
    return ax


_REG_METRIC_LABELS = {"ll": "log-likelihood", "rsquared": "R²"}


def finetune_regression_curve(reg_df, spec, metric="rsquared", ax=None):
    """Regression fit vs fine-tuning epoch, one line per reader group.

    ``reg_df`` is the output of ``analysis.regression_over_epochs``. ``spec``
    selects the model spec (a key of ``analysis.REGRESSION_SPECS``) and
    ``metric`` is ``"ll"`` (log-likelihood) or ``"rsquared"``.
    """
    ax = ax or plt.gca()
    sub = reg_df[reg_df["spec"] == spec]
    for group, g in sub.groupby("group"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g[metric], marker="o", label=group)
    label = _REG_METRIC_LABELS.get(metric, metric)
    ax.set(xlabel="fine-tuning epoch", ylabel=f"{label}  (rt ~ {spec})")
    ax.legend(title="readers")
    return ax


def perplexity_curve(manifest_df, ax=None):
    """Fine-tuning progress: validation perplexity vs words processed."""
    ax = ax or plt.gca()
    for domain, g in manifest_df.groupby("domain"):
        g = g.sort_values("words_seen")
        ax.plot(g["words_seen"], g["perplexity"], marker="o", label=domain)
    ax.set(xlabel="words processed", ylabel="validation perplexity")
    ax.legend(title="domain")
    return ax
