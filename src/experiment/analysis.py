"""Attention-flow vs gaze correlation, split by reader expertise.

Reuses ``analysis.correlation`` (``build_et_table`` + ``correlate_attention``) to
score how well encoder attention flow tracks gaze, separately for domain experts
and novices (``is_expert`` = reader major matches text domain). One table per
layer for the baseline reproduction figure; the peak-layer summary for the
fine-tuning curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis import correlation as co

GROUPS = ("experts", "novices")

# Paper §3.3 text predictors that survive in this corpus (functional category +
# saliency are out of scope here); attention is added on top of these.
TEXT_FEATURES = ["log_word_freq", "log_word_length"]


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


# ── predictive modeling (paper §3.3): does attention add R² over text features? ─
def _word_text_features(rm: pd.DataFrame) -> pd.DataFrame:
    """Per-word regression covariates: log frequency + log word length.

    Deduplicated to one row per word (the reading measures are per reader×word).
    Mirrors the covariate transforms in ``model_comparison._prep_models``: log1p
    on dlexDB lemma frequency (per-million, right-skewed; keeps zero-freq finite)
    and log on word length.
    """
    cols = co.WORD_KEY + ["word_length", "lemma_frequency_normalized"]
    tf = rm[cols].drop_duplicates(co.WORD_KEY).copy()
    tf = tf[tf["word_length"] > 0]
    tf["log_word_length"] = np.log(tf["word_length"])
    tf["log_word_freq"] = np.log1p(tf["lemma_frequency_normalized"])
    return tf[co.WORD_KEY + TEXT_FEATURES]


def _adj_r2_holdout(df, x_cols, y_col="pca", test_frac=0.2, seed=0):
    """Adjusted R² of an OLS fit, scored on a held-out ``test_frac`` split.

    Paper §3.3 measures predictive capacity as adjusted R² on unseen data (20% of
    the dataset). We fit on the train split and adjust the test R² for the
    predictor count so models with different ``x_cols`` are comparable. Returns
    NaN when too few words remain to fit/score reliably.
    """
    d = df[x_cols + [y_col]].dropna()
    p = len(x_cols)
    if len(d) < 5 * (p + 2):
        return np.nan
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(d))
    cut = int(round(len(d) * (1 - test_frac)))
    tr, te = d.iloc[perm[:cut]], d.iloc[perm[cut:]]
    res = sm.OLS(tr[y_col], sm.add_constant(tr[x_cols])).fit()
    pred = res.predict(sm.add_constant(te[x_cols], has_constant="add"))
    ss_tot = ((te[y_col] - te[y_col].mean()) ** 2).sum()
    if ss_tot <= 0 or len(te) - p - 1 <= 0:
        return np.nan
    r2 = 1 - ((te[y_col] - pred) ** 2).sum() / ss_tot
    return 1 - (1 - r2) * (len(te) - 1) / (len(te) - p - 1)


def regression_over_checkpoints(
    versions: pd.DataFrame,
    rm_raw: pd.DataFrame,
    method: str = "raw",
    feature: str = "pca",
    seed: int = 0,
) -> pd.DataFrame:
    """Held-out adjusted R² predicting gaze PCA from text features ± attention.

    Paper §3.3's second method (predictive modeling) ported to the fine-tuning
    experiment. Per checkpoint × reader group, two OLS models predict the gaze
    PCA component: a text-only baseline (``TEXT_FEATURES``) and text+attention.
    Both are scored by adjusted R² on a 20% held-out split; ``delta_r2`` is the
    predictive gain attention adds over the surface text features. Attention is
    read at the peak-correlation layer (the same layer
    ``correlation_over_checkpoints`` reports), and each checkpoint is scored only
    against its own fine-tuning domain's texts. Returns ``index``, ``epoch``,
    ``tokens_seen``, ``domain``, ``group``, ``layer``, ``r2_base``, ``r2_attn``,
    ``delta_r2``, ``n``, ``method``.
    """
    text = _word_text_features(rm_raw)
    rows = []
    for (index, epoch, tokens_seen, domain), fv in versions.groupby(
        ["index", "epoch", "tokens_seen", "domain"]
    ):
        for grp in GROUPS:
            et = co.build_et_table(rm_raw, domain=domain, participants=grp)
            corr = co.correlate_attention(fv, et, attention_method=method)
            corr = corr[corr["feature"] == feature]
            if not corr["spearman"].notna().any():
                continue
            layer = int(corr.loc[corr["spearman"].idxmax(), "layer"])
            attn = fv[(fv["attention_method"] == method) & (fv["layer"] == layer)]
            df = (
                et[co.WORD_KEY + [feature]]
                .merge(text, on=co.WORD_KEY)
                .merge(attn[co.WORD_KEY + ["attention"]], on=co.WORD_KEY)
            )
            r2_base = _adj_r2_holdout(df, TEXT_FEATURES, feature, seed=seed)
            r2_attn = _adj_r2_holdout(df, TEXT_FEATURES + ["attention"], feature, seed=seed)
            rows.append(
                {
                    "index": index,
                    "epoch": epoch,
                    "tokens_seen": tokens_seen,
                    "domain": domain,
                    "group": grp,
                    "layer": layer,
                    "r2_base": r2_base,
                    "r2_attn": r2_attn,
                    "delta_r2": r2_attn - r2_base,
                    "n": len(df),
                    "method": method,
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "group", "tokens_seen"])
