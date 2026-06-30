"""Plots for the surprisal / fine-tuning results (step 6).

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


def finetune_correlation_curve(curve_df, metric="pearson", ax=None):
    """Surprisal-RT correlation vs fine-tuning epoch, one line per (group, domain).

    ``curve_df`` from ``correlation.correlation_over_epochs``. Domains stay
    separate lines (their epoch floats differ; pooling would conflate them).
    """
    ax = ax or plt.gca()
    for (group, domain), g in curve_df.groupby(["group", "domain"]):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g[metric], marker="o", label=f"{group}·{domain}")
    ax.set(xlabel="fine-tuning epoch", ylabel=f"{metric} r (surprisal vs RT)")
    ax.legend(title="readers·domain")
    return ax


def model_comparison_curve(cmp_df, metric="delta_ll", ax=None):
    """Surprisal fit vs fine-tuning epoch, one line per surprisal model.

    ``cmp_df`` from ``model_comparison.model_comparison_over_epochs``. With
    ``delta_ll``, higher = better fit; ``aligned`` above the single models means
    matching the LM to the reader's discipline helps.
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


# ── cross-model comparison (all models on one axes) ──────────────────────────
def across_models_correlation_bars(summary_df, metrics=("pearson", "spearman"), ax=None):
    """Grouped bar of the surprisal-vs-RT correlation per model.

    ``summary_df``: one row per model + a column per metric in ``metrics``.
    """
    ax = ax or plt.gca()
    models = summary_df["model"].tolist()
    x = range(len(models))
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ax.bar([xi + i * width for xi in x], summary_df[m], width, label=m)
    ax.set_xticks([xi + width * (len(metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(ylabel="correlation (surprisal vs RT)")
    ax.legend(title="metric")
    return ax


def across_models_bar(summary_df, value, ylabel=None, ax=None):
    """Single-metric bar across models (e.g. regression slope, aligned ΔLL)."""
    ax = ax or plt.gca()
    pos = list(range(len(summary_df)))
    ax.bar(pos, summary_df[value])
    ax.set_xticks(pos)
    ax.set_xticklabels(summary_df["model"], rotation=15, ha="right")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(ylabel=ylabel or value.replace("_", " "))
    return ax
