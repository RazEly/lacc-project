"""Correlation and regression of model scores against reading times (step 5).

All functions take the per-reader cleaned reading measures so that the
``participants`` (experts / novices / all) and ``mode`` (mean / median) options
can be applied before aggregating to one value per word.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from src.attention import eyetracking_features, pca_eyetracking
from src.config import ET_MEASURE_MAP

WORD_KEY = ["text_id", "word_index_in_text"]


# ── filters ──────────────────────────────────────────────────────────────────
def _filter_domain(df, domain):
    return df if domain == "all" else df[df["text_domain"] == domain]


def _filter_participants(df, participants):
    """Split readers by ``is_expert`` (reader major == text domain).

    See ``data.add_expertise``: experts are readers whose discipline matches the
    text's domain, novices the rest. Requires the ``is_expert`` column.
    """
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
    """Pearson and Spearman correlation between surprisal and reading time.

    Filters by text ``domain``, reader ``participants`` group, and (optionally)
    ``domain_only`` technical-term words, then aggregates reading time per word
    with ``mode`` ('mean'/'median') before correlating.
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


# ── attention <-> eye-tracking ───────────────────────────────────────────────
def build_et_table(rm: pd.DataFrame, domain="all", participants="all"):
    """Per-word eye-tracking features (+ per-domain PCA component).

    Applies the ``participants`` filter before averaging across readers, so the
    feature table reflects the chosen reader group. PCA is fit per domain and
    concatenated.
    """
    rm = _filter_participants(rm, participants)
    et = eyetracking_features(rm)
    et = _filter_domain(et, domain)
    pcas = []
    for dom in et["text_domain"].unique():
        scored, _ = pca_eyetracking(et, dom)
        pcas.append(scored)
    if pcas:
        et = et.merge(pd.concat(pcas), on=WORD_KEY, how="left")
    return et


def correlate_attention(
    attn_df: pd.DataFrame, et_df: pd.DataFrame, layer="all", attention_method="raw"
) -> pd.DataFrame:
    """Layer-wise Spearman of attention against each eye-tracking feature.

    Returns a table with columns ``layer``, ``feature``, ``attention_method``,
    ``spearman``, ``p``, ``n`` — one row per (layer, feature). Features are the
    six mapped eye-tracking measures plus the ``pca`` component.
    """
    attn = attn_df[attn_df["attention_method"] == attention_method]
    if layer != "all":
        attn = attn[attn["layer"] == layer]

    features = [m for m in dict.fromkeys(ET_MEASURE_MAP.values()) if m in et_df.columns]
    if "pca" in et_df.columns:
        features.append("pca")

    merged = attn.merge(et_df, on=WORD_KEY, how="inner")
    rows = []
    for lyr, g in merged.groupby("layer"):
        for feat in features:
            sub = g[["attention", feat]].dropna()
            if len(sub) < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(sub["attention"], sub[feat])
            rows.append((lyr, feat, attention_method, rho, p, len(sub)))
    return pd.DataFrame(
        rows, columns=["layer", "feature", "attention_method", "spearman", "p", "n"]
    )


# ── fine-tuning progress: correlation per checkpoint ─────────────────────────
def correlation_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    domain_only=False,
    mode="mean",
    measure="TFT",
    groups=("experts", "novices"),
) -> pd.DataFrame:
    """Surprisal-RT correlation at each fine-tuning checkpoint, per reader group.

    ``surp_versions`` is the output of
    ``finetune.recompute_surprisal_over_checkpoints`` (per-word surprisal tagged
    with ``epoch`` and the fine-tuning ``domain``). A domain-adapted model is
    only a meaningful predictor on texts of its own domain, so each checkpoint's
    surprisal is correlated **only against texts whose domain matches the model's
    fine-tuning domain** (physics model -> physics texts, biology -> biology).
    Returns long-form columns ``epoch``, ``domain``, ``group``, ``pearson``,
    ``spearman``, ``n``.
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


# ── regression fit per checkpoint (log-likelihood / R²) ──────────────────────
# Two model specs for mean reading time per word. The first is surprisal alone;
# the second adds word frequency and log word length as lexical covariates.
REGRESSION_SPECS = {
    "surprisal": ["surprisal"],
    "surprisal+freq+length": ["surprisal", "log_word_freq", "log_word_length"],
}


def _aggregate_for_regression(df, measure, mode):
    """One row per word: aggregated reading time + surprisal + lexical features."""
    agg = (
        df.groupby(WORD_KEY)
        .agg(
            rt=(measure, mode),
            surprisal=("surprisal", "first"),
            word_freq=("lemma_frequency_normalized", "first"),
            word_length=("word_length", "first"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["rt", "surprisal", "word_freq", "word_length"])
    agg = agg[agg["word_length"] > 0]
    agg["log_word_length"] = np.log(agg["word_length"])
    # dlexDB lemma freq (per million words) is heavily right-skewed; log it.
    # log1p keeps the zero-frequency words (missing dlexDB entries) finite.
    agg["log_word_freq"] = np.log1p(agg["word_freq"])
    return agg


def _fit_spec(agg, predictors):
    """OLS ``rt ~ predictors``; return (log-likelihood, R²)."""
    X = sm.add_constant(agg[predictors])
    res = sm.OLS(agg["rt"], X).fit()
    return res.llf, res.rsquared


def regression_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    mode="mean",
    measure="TFT",
    groups=("experts", "novices"),
    domain_only=False,
) -> pd.DataFrame:
    """Per-checkpoint regression fit of mean reading time, per reader group.

    For every fine-tuning checkpoint (matched to its own text domain, see
    ``correlation_over_epochs``) and reader group, the per-word mean reading time
    is regressed on each spec in ``REGRESSION_SPECS`` and the model
    log-likelihood + R² recorded. Returns long-form columns ``epoch``,
    ``domain``, ``group``, ``spec``, ``ll``, ``rsquared``, ``n``.
    """
    rows = []
    for (epoch, model_domain), sv in surp_versions.groupby(["epoch", "domain"]):
        merged = merge_surprisal_rt(sv[WORD_KEY + ["surprisal"]], rt_df)
        for grp in groups:
            df = _filter_domain(merged, model_domain)
            df = _filter_participants(df, grp)
            df = _filter_domain_only(df, domain_only)
            agg = _aggregate_for_regression(df, measure, mode)
            for spec, predictors in REGRESSION_SPECS.items():
                if len(agg) < 5:
                    ll, r2 = np.nan, np.nan
                else:
                    ll, r2 = _fit_spec(agg, predictors)
                rows.append(
                    {
                        "epoch": epoch,
                        "domain": model_domain,
                        "group": grp,
                        "spec": spec,
                        "ll": ll,
                        "rsquared": r2,
                        "n": len(agg),
                    }
                )
    return pd.DataFrame(rows).sort_values(["spec", "group", "epoch"])


# ── multiple-comparison correction (plan TODO, line 100) ─────────────────────
def bh_correct(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject, p_adjusted)."""
    p = np.asarray(pvalues, dtype=float)
    mask = ~np.isnan(p)
    reject = np.zeros(len(p), bool)
    p_adj = np.full(len(p), np.nan)
    if mask.sum():
        rej, padj, _, _ = multipletests(p[mask], alpha=alpha, method="fdr_bh")
        reject[mask] = rej
        p_adj[mask] = padj
    return reject, p_adj
