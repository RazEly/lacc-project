"""Significance tests for the surprisal-model comparison.

ΔLL ranks the models but carries no standard error. The models are non-nested
(no LR test), so we use the Vuong (1989) test, clustered by reader. Under the
crossed random effects ``(1|reader_id) + (1 + is_expert|word_id)`` the likelihood
no longer factorises over readers (shared words couple them), so we take the
*conditional* per-observation log-density given the fitted values (BLUPs) as the
pointwise Vuong contribution and sum it within reader to get one cluster-robust
unit per reader. All models add one surprisal slope, so the complexity correction
cancels. Comparisons are Benjamini-Hochberg corrected; positive z = aligned wins,
p two-sided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
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


def _pointwise_loglik(m) -> pd.DataFrame:
    """Conditional per-observation log-density from a fitted pymer4 ``Lmer``.

    Uses the residuals about the fitted values (which include the random-effect
    BLUPs): ``logpdf(resid; 0, σ̂)`` with σ̂ the conditional residual SD. Returns a
    frame with ``row_id`` (pairing key), ``reader_id`` and ``ll``. ``m.data`` is the
    polars frame pymer4 0.9 augments with a ``resid`` column.
    """
    df = m.data
    resid = df["resid"].to_numpy().astype(float)
    sigma = resid.std(ddof=0) or 1.0
    return pd.DataFrame(
        {
            "row_id": df["row_id"].to_numpy(),
            "reader_id": df["reader_id"].to_numpy(),
            "ll": norm.logpdf(resid, loc=0.0, scale=sigma),
        }
    )


def vuong_test(m_a, m_b) -> dict:
    """Reader-clustered Vuong test that model A fits better than model B.

    Both fits must come from the same rows (paired by ``row_id``). Pointwise
    log-density differences are summed within reader, giving one cluster unit per
    reader. Returns the z-statistic, two-sided p-value, mean per-reader summed
    log-likelihood difference and the number of reader clusters.
    """
    a = _pointwise_loglik(m_a)
    b = _pointwise_loglik(m_b)
    merged = a.merge(b, on=["row_id", "reader_id"], suffixes=("_a", "_b"))
    merged["d"] = merged["ll_a"] - merged["ll_b"]
    s = merged.groupby("reader_id")["d"].sum()

    n = len(s)
    sd = s.std(ddof=0)
    z = np.sqrt(n) * s.mean() / sd if sd > 0 else 0.0
    return {
        "n_readers": n,
        "mean_dll": float(s.mean()),
        "z": float(z),
        "p": 2.0 * norm.sf(abs(z)),
    }


def aligned_vs_single_models(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure: str = "TFT",
    index=None,
    spec: str = "paper_full",
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


def _best_aligned_index(surp_versions, rt_df, prompt_surp, measure, spec="paper_full"):
    """Checkpoint index at which the aligned model has the largest ΔLL."""
    cmp = mc.model_comparison_over_epochs(
        surp_versions, rt_df, prompt_surp, measure=measure, spec=spec
    )
    aligned = cmp[cmp["model"] == "aligned"]
    return aligned.loc[aligned["delta_ll"].idxmax(), "index"]
