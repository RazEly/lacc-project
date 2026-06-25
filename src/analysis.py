"""Correlation and regression of model scores against reading times (step 5).

All functions take the per-reader cleaned reading measures so that the
``participants`` (experts / novices / all) and ``mode`` (mean / median) options
can be applied before aggregating to one value per word.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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


# ── four-model surprisal comparison: reader-aligned vs single model ──────────
# On the WHOLE corpus (both text domains, all readers) compare how well four
# surprisal sources predict reading time, each in the same mixed model:
#
#   RT ~ log_word_freq + log_word_length + S + (1 | reader)
#
# with S one of:
#   baseline : the un-adapted step-0 model's surprisal (same for every reader)
#   physics  : the physics fine-tuned model's surprisal (every reader)
#   biology  : the biology fine-tuned model's surprisal (every reader)
#   aligned  : reader-aligned — physics surprisal for physicists, biology
#              surprisal for biologists, i.e.
#              physicist * S_physics + (1 - physicist) * S_biology.
#
# In ``baseline`` / ``physics`` / ``biology`` the reader's discipline is ignored;
# only ``aligned`` matches the language model to the reader's field. Alignment is
# by READER discipline, not text domain, so a physicist gets physics surprisal
# even on a biology text. Every model adds a single surprisal slope, so their
# log-likelihoods are directly comparable. ΔLL is measured against the
# no-surprisal baseline ``RT ~ freq + length + (1|reader)``. The question: does
# ``aligned`` improve the fit over the three single-model sources? RT is the raw
# measure (swap to ``np.log`` here for log-RT). Only ``(1|reader)`` is random, so
# each fit is sub-second.
SURPRISAL_MODELS = ("baseline", "physics", "biology", "aligned")


def _prep_models(df, measure):
    """One row per reader×word with the four surprisal columns + covariates.

    ``df`` must already carry per-word ``s_base`` / ``s_phys`` / ``s_bio``
    surprisal (base, physics-adapted, biology-adapted) and the reading measures.
    """
    d = df[df[measure] > 0].copy()
    d = d.dropna(
        subset=[measure, "word_length", "lemma_frequency_normalized",
                "s_base", "s_phys", "s_bio"]
    )
    d = d[d["word_length"] > 0]
    # dlexDB lemma freq (per million) is heavily right-skewed; log1p keeps the
    # zero-frequency words (missing dlexDB entries) finite.
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])
    d["log_word_length"] = np.log(d["word_length"])
    # reader discipline (1 = physics, 0 = biology); see data.add_expertise.
    physicist = (d["reader_discipline_numeric"] == 1).to_numpy(dtype=float)
    d["S_baseline"] = d["s_base"]
    d["S_physics"] = d["s_phys"]
    d["S_biology"] = d["s_bio"]
    d["S_aligned"] = physicist * d["s_phys"] + (1.0 - physicist) * d["s_bio"]
    return d


def _fit_model(d, measure, surprisal_col=None):
    """Mixed model ``measure ~ freq + length [+ surprisal_col] + (1|reader)``."""
    rhs = "log_word_freq + log_word_length"
    if surprisal_col is not None:
        rhs += f" + {surprisal_col}"
    md = smf.mixedlm(f"{measure} ~ {rhs}", d, groups=d["reader_id"])
    return md.fit(reml=False)  # ML so LLs are comparable across the surprisal term


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    measure="TFT",
) -> pd.DataFrame:
    """Per-checkpoint four-model surprisal comparison on the whole corpus.

    ``surp_versions`` must contain BOTH a ``physics`` and a ``biology`` domain
    (fine-tune both). The smallest ``epoch`` is the un-adapted step-0 model and
    supplies the ``baseline`` surprisal (domain-agnostic). For every checkpoint
    (epoch) the physics, biology and reader-aligned surprisal columns are built
    and each of the four models in ``SURPRISAL_MODELS`` is fit on all readers ×
    words of the whole corpus, plus the no-surprisal baseline, recording the
    log-likelihood, ΔLL over that baseline and the surprisal slope. Returns
    long-form columns ``epoch``, ``model``, ``n``, ``ll``, ``delta_ll``,
    ``b_surprisal``, ``p_surprisal``.
    """
    epochs = sorted(surp_versions["epoch"].unique())
    base_epoch = epochs[0]

    def _surp(epoch, domain, name):
        sel = surp_versions[
            (surp_versions["epoch"] == epoch) & (surp_versions["domain"] == domain)
        ]
        return sel[WORD_KEY + ["surprisal"]].rename(columns={"surprisal": name})

    # step-0 weights are domain-agnostic, so either domain's epoch-0 works.
    s_base = _surp(base_epoch, "physics", "s_base")
    rows = []
    for epoch in epochs:
        surp = (
            s_base.merge(_surp(epoch, "physics", "s_phys"), on=WORD_KEY)
            .merge(_surp(epoch, "biology", "s_bio"), on=WORD_KEY)
        )
        merged = surp.merge(rt_df, on=WORD_KEY, how="inner")
        d = _prep_models(merged, measure)
        ll0 = _fit_model(d, measure, None).llf  # no-surprisal reference
        for name in SURPRISAL_MODELS:
            col = f"S_{name}"
            res = _fit_model(d, measure, col)
            rows.append(
                {
                    "epoch": epoch,
                    "model": name,
                    "n": len(d),
                    "ll": res.llf,
                    "delta_ll": res.llf - ll0,
                    "b_surprisal": res.fe_params.get(col, np.nan),
                    "p_surprisal": res.pvalues.get(col, np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "epoch"])


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
