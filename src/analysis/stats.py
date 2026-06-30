"""Significance tests for the surprisal-model comparison.

ΔLL ranks the models but carries no standard error. The models are non-nested
(no LR test), so we use the Vuong (1989) test, clustered by reader: a
random-intercept ``mixedlm`` log-likelihood factorises over readers, so each
reader's marginal log-likelihood is one i.i.d. Vuong unit (the cluster-robust
variance the per-observation test misses). All models add one surprisal slope, so
the complexity correction cancels. Comparisons are Benjamini-Hochberg corrected;
positive z = aligned wins, p two-sided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from statsmodels.stats.multitest import multipletests

from src.analysis import model_comparison as mc

# aligned is the model under test; compare it against every other model.
_BASELINES = ("baseline", "physics", "biology", "prompted", "prompted_level")


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


def _per_reader_loglik(result) -> tuple[np.ndarray, np.ndarray]:
    """Marginal log-likelihood of each reader group under a fitted MixedLM.

    Random-intercept marginal per reader: ``N(Xβ, Z·cov_re·Zᵀ + scale·I)``.
    Returns ``(group_ids, loglik)``.
    """
    model = result.model
    y = np.asarray(model.endog, dtype=float)
    x = np.asarray(model.exog, dtype=float)
    z = np.asarray(model.exog_re, dtype=float)
    groups = np.asarray(model.groups)
    beta = np.asarray(result.fe_params.values, dtype=float)
    cov_re = np.asarray(result.cov_re, dtype=float)
    scale = float(result.scale)

    mu = x @ beta
    uniq = pd.unique(groups)
    ll = np.empty(len(uniq))
    for i, g in enumerate(uniq):
        m = groups == g
        zg = z[m]
        cov = zg @ cov_re @ zg.T + scale * np.eye(m.sum())
        ll[i] = multivariate_normal.logpdf(y[m], mean=mu[m], cov=cov)
    return uniq, ll


def vuong_test(result_a, result_b) -> dict:
    """Reader-clustered Vuong test that model A fits better than model B.

    Both results must be fit on the same rows (same reader groups). Returns the
    z-statistic, two-sided p-value, per-reader mean log-likelihood difference and
    the number of reader clusters.
    """
    ga, lla = _per_reader_loglik(result_a)
    gb, llb = _per_reader_loglik(result_b)
    # align by reader id so differences are paired
    order = pd.Series(np.arange(len(gb)), index=gb)
    d = lla - llb[order.loc[ga].to_numpy()]

    n = len(d)
    sd = d.std(ddof=0)
    z = np.sqrt(n) * d.mean() / sd if sd > 0 else 0.0
    return {
        "n_readers": n,
        "mean_dll": d.mean(),
        "z": z,
        "p": 2.0 * norm.sf(abs(z)),
    }


def aligned_vs_single_models(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure: str = "TFT",
    index=None,
    spec: str = "covariates",
) -> pd.DataFrame:
    """Vuong-test reader-aligned surprisal against every other model.

    Fits at one checkpoint (default: aligned's best-ΔLL checkpoint) under ``spec``,
    BH-corrects across comparisons. One row per comparison: z, raw + adjusted p, winner.
    """
    if index is None:
        index = _best_aligned_index(surp_versions, rt_df, prompt_surp, measure, spec)

    d = mc.build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
    fits = {
        name: mc._fit_model(d, measure, f"S_{name}", spec=spec)
        for name in ("aligned", *_BASELINES)
    }

    rows = []
    for other in _BASELINES:
        res = vuong_test(fits["aligned"], fits[other])
        rows.append(
            {
                "index": index,
                "comparison": f"aligned vs {other}",
                "n_readers": res["n_readers"],
                "mean_dll": res["mean_dll"],
                "z": res["z"],
                "p": res["p"],
                "winner": "aligned" if res["z"] > 0 else other,
            }
        )
    out = pd.DataFrame(rows)
    reject, p_adj = bh_correct(out["p"].to_numpy())
    out["p_adj"] = p_adj
    out["significant"] = reject
    return out


def _best_aligned_index(surp_versions, rt_df, prompt_surp, measure, spec="covariates"):
    """Checkpoint index at which the aligned model has the largest ΔLL."""
    cmp = mc.model_comparison_over_epochs(
        surp_versions, rt_df, prompt_surp, measure=measure, spec=spec
    )
    aligned = cmp[cmp["model"] == "aligned"]
    return aligned.loc[aligned["delta_ll"].idxmax(), "index"]
