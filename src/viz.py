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


def perplexity_curve(manifest_df, ax=None):
    """Fine-tuning progress: validation perplexity vs words processed."""
    ax = ax or plt.gca()
    for domain, g in manifest_df.groupby("domain"):
        g = g.sort_values("words_seen")
        ax.plot(g["words_seen"], g["perplexity"], marker="o", label=domain)
    ax.set(xlabel="words processed", ylabel="validation perplexity")
    ax.legend(title="domain")
    return ax
