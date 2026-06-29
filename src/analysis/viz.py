"""Plots for the surprisal / attention / fine-tuning results (step 6).

Each function takes prepared DataFrames (from ``analysis`` / ``finetune``) and
returns a matplotlib Axes so the caller can save or compose figures.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def surprisal_scatter(agg_df, x="surprisal", y="rt", bins=15, ax=None):
    """Binned surprisal-vs-reading-time scatter with an OLS fit line."""
    ax = ax or plt.gca()
    sns.regplot(data=agg_df, x=x, y=y, x_bins=bins, ax=ax)
    ax.set(xlabel="surprisal (bits)", ylabel="reading time (ms)")
    return ax


def attention_layer_curve(corr_table, feature="pca", ax=None):
    """Per-layer Spearman curve for one eye-tracking feature (raw attention)."""
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


def attention_prediction_bars(pred_df, ax=None):
    """Adjusted-R² bars (text-only vs +attention) per gaze target.

    ``pred_df`` is a single (model, group) slice of
    ``attention_prediction.run_model_prediction``'s summary. Two bars per target;
    a ★ over the pair marks a significant Wilcoxon test on holdout squared errors
    (p<0.05), i.e. raw attention significantly changes prediction error.
    """
    ax = ax or plt.gca()
    df = pred_df.sort_values("target")
    x = range(len(df))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], df["adj_r2_text"], width, label="text only")
    ax.bar([xi + width / 2 for xi in x], df["adj_r2_attn"], width, label="text + attention")
    top = max(df[["adj_r2_text", "adj_r2_attn"]].to_numpy().max(), 0)
    for xi, (_, r) in zip(x, df.iterrows()):
        if pd.notna(r["wilcoxon_p"]) and r["wilcoxon_p"] < 0.05:
            ax.text(xi, top * 1.02 + 0.005, "★", ha="center", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["target"], rotation=20, ha="right")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(ylabel="adjusted R² (holdout)")
    ax.legend(title="predictors")
    return ax


def feature_importance_bars(imp_df, ax=None):
    """Mean |t-value| per predictor across gaze targets (paper Fig 4).

    ``imp_df`` is a single (model, group) slice of the importance table; bars are
    averaged over targets so each text/attention predictor's overall contribution
    to the regression is read off at a glance.
    """
    ax = ax or plt.gca()
    m = imp_df.groupby("predictor")["abs_t"].mean().sort_values(ascending=False)
    ax.bar(range(len(m)), m.values)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(m.index, rotation=20, ha="right")
    ax.set(ylabel="mean |t-value|")
    return ax


def cross_domain_perplexity_bars(ppl_df, ax=None):
    """Grouped perplexity bars: each model on the general/physics/biology val sets.

    ``ppl_df`` is ``finetune.cross_domain_perplexity`` output (``model``,
    ``eval_domain``, ``perplexity``). One bar group per model, one bar per eval
    domain — so a DAPT model's bars on the *other* domains (a physics-adapted
    model on biology / general text) show what fine-tuning for one field costs
    elsewhere, read against the baseline group.
    """
    ax = ax or plt.gca()
    domains = ["general", "physics", "biology"]
    models = ppl_df["model"].drop_duplicates().tolist()
    table = ppl_df.pivot(index="model", columns="eval_domain", values="perplexity")
    table = table.reindex(index=models, columns=domains)
    x = range(len(models))
    width = 0.8 / len(domains)
    for i, d in enumerate(domains):
        ax.bar([xi + i * width for xi in x], table[d].values, width, label=d)
    ax.set_xticks([xi + width * (len(domains) - 1) / 2 for xi in x])
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set(ylabel="validation perplexity")
    ax.legend(title="eval domain")
    return ax


# ── cross-model comparison (all models on one axes) ──────────────────────────
def across_models_correlation_bars(summary_df, metrics=("pearson", "spearman"), ax=None):
    """Grouped bar of the surprisal-vs-RT correlation per model.

    ``summary_df`` has one row per model (column ``model``) plus a column per
    metric in ``metrics`` — the all-readers, both-domains correlation from
    ``correlation.correlate_surprisal``. One bar group per model lets the
    models be read off side by side.
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


def across_models_attention_curve(attn_by_model, feature="pca", method="raw", ax=None):
    """Per-layer attention-vs-gaze Spearman curve, one line per model.

    ``attn_by_model`` maps model slug -> the ``correlate_attention`` table for
    that model. Models differ in depth, so the x-axis is the raw layer index;
    the comparison is of where (and how strongly) each model's attention tracks
    gaze, not a layer-for-layer alignment.
    """
    ax = ax or plt.gca()
    for model, corr in attn_by_model.items():
        sub = corr[(corr["feature"] == feature) & (corr["attention_method"] == method)]
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub["spearman"], marker="o", label=model)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="layer", ylabel=f"Spearman r ({feature})")
    ax.legend(title="model")
    return ax
