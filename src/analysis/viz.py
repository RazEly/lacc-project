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


def finetune_correlation_curve(curve_df, metric="pearson", ax=None):
    """Surprisal-RT correlation vs fine-tuning epoch, per reader group × domain.

    ``curve_df`` is the output of ``correlation.correlation_over_epochs``, which
    holds BOTH fine-tuning domains (physics model on physics texts, biology on
    biology). Each (group, domain) is its own line — pooling the two domains into
    one line per group would weave a single series through both runs' checkpoints,
    whose ``epoch`` floats differ, and conflate physics with biology adaptation.
    """
    ax = ax or plt.gca()
    for (group, domain), g in curve_df.groupby(["group", "domain"]):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g[metric], marker="o", label=f"{group}·{domain}")
    ax.set(xlabel="fine-tuning epoch", ylabel=f"{metric} r (surprisal vs RT)")
    ax.legend(title="readers·domain")
    return ax


def model_comparison_curve(cmp_df, metric="delta_ll", ax=None):
    """Four-model surprisal fit vs fine-tuning epoch, one line per surprisal model.

    ``cmp_df`` is the output of ``model_comparison.model_comparison_over_epochs``: the
    ``baseline`` / ``physics`` / ``biology`` / ``aligned`` surprisal sources. With
    ``metric="delta_ll"`` a higher line = that source improves the fit more. If
    ``aligned`` sits above the three single-model sources, matching the language
    model to the reader's discipline helps.
    """
    ax = ax or plt.gca()
    for model, g in cmp_df.groupby("model"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g[metric], marker="o", label=model)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="fine-tuning epoch", ylabel=metric.replace("_", " "))
    ax.legend(title="surprisal model")
    return ax


def perplexity_curve(manifest_df, ax=None):
    """Fine-tuning progress: validation perplexity vs training tokens processed."""
    ax = ax or plt.gca()
    for domain, g in manifest_df.groupby("domain"):
        g = g.sort_values("tokens_seen")
        ax.plot(g["tokens_seen"], g["perplexity"], marker="o", label=domain)
    ax.set(xlabel="tokens processed", ylabel="validation perplexity")
    ax.legend(title="domain")
    return ax
