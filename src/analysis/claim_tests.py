"""Claim-level tests on top of the checkpoint sweep (step 6).

Two claims need more than the per-checkpoint sweep:

1. "Reader-aligned DAPT surprisal helps" — tested ONCE per LM × measure at the
   checkpoint selected by max ΔLL (Škrjanec & Demberg's protocol: select the
   best-fitting amount of training first, then run a single test, avoiding the
   multiple-comparison problem of testing every checkpoint).
2. "The effect holds for GPT-2 but not for the bigger LM" — a difference claim,
   which needs a DIRECT cross-LM comparison, not two separate significance
   verdicts (the difference between significant and non-significant is not
   itself significant; Gelman & Stern 2006). Two complementary tests:
     * ``slope_difference`` — Wald z on the difference of the residual slopes.
       Both fits use the same readers/items, so the estimates are correlated;
       treating them as independent overstates the SE of the difference, making
       the test conservative.
     * ``reader_pair_test`` — per-reader conditional ΔLL (aligned − base
       reference, from ``model_comparison._reader_loglik``) paired across LMs by
       reader; paired t-test + Wilcoxon. Conditional densities approximate each
       reader's contribution to the fit; the pairing cancels shared terms.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def best_checkpoints(results: pd.DataFrame, model: str = "aligned") -> pd.DataFrame:
    """Best checkpoint per LM × measure for ``model``, selected by max ΔLL.

    ``results`` is the concatenated sweep output with ``model_lm`` and
    ``measure`` columns. Returns one row per (model_lm, measure) with the
    selected checkpoint's stats + a 95% Wald CI on the residual main-effect
    slope. ``p_lrt`` at the selected point is the single claim test.
    """
    sel = results[(results["model"] == model) & results["index"].notna()]
    rows = []
    for (lm, measure), g in sel.groupby(["model_lm", "measure"]):
        r = g.loc[g["delta_ll"].idxmax()]
        b, se = float(r["b_resid"]), float(r["se_resid"])
        rows.append(
            {
                "model_lm": lm,
                "measure": measure,
                "index": r["index"],
                "epoch": r["epoch"],
                "delta_ll": r["delta_ll"],
                "p_lrt": r["p_lrt"],
                "b_resid": b,
                "se_resid": se,
                "ci_low": b - 1.96 * se,
                "ci_high": b + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def slope_difference(best: pd.DataFrame) -> pd.DataFrame:
    """Wald z-test on the difference of residual slopes between LM pairs.

    Per measure and LM pair: z = (b1 - b2) / sqrt(se1² + se2²). The two fits
    share readers/items, so their estimates are positively correlated and the
    independent-SE denominator is an upper bound — the test is conservative.
    """
    rows = []
    for measure, g in best.groupby("measure"):
        g = g.set_index("model_lm")
        for lm1, lm2 in combinations(g.index, 2):
            b1, se1 = g.loc[lm1, "b_resid"], g.loc[lm1, "se_resid"]
            b2, se2 = g.loc[lm2, "b_resid"], g.loc[lm2, "se_resid"]
            z = (b1 - b2) / np.sqrt(se1**2 + se2**2)
            rows.append(
                {
                    "measure": measure,
                    "lm1": lm1,
                    "lm2": lm2,
                    "test": "slope_diff_z",
                    "estimate": b1 - b2,
                    "stat": z,
                    "p": float(2 * stats.norm.sf(abs(z))),
                }
            )
    return pd.DataFrame(rows)


def _reader_delta(reader_ll: pd.DataFrame, lm, measure, index) -> pd.Series:
    """Per-reader conditional ΔLL (aligned at ``index`` − base reference)."""
    g = reader_ll[(reader_ll["model_lm"] == lm) & (reader_ll["measure"] == measure)]
    ref = g[g["model"] == "base_ref"].set_index("reader_id")["ll_reader"]
    exp = g[(g["model"] == "aligned") & (g["index"] == index)].set_index("reader_id")[
        "ll_reader"
    ]
    return (exp - ref).dropna()


def reader_pair_test(reader_ll: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Paired cross-LM test of per-reader conditional ΔLL at best checkpoints.

    For each measure and LM pair: pair readers, test Δ(lm1) − Δ(lm2) with a
    paired t-test and a Wilcoxon signed-rank test. Also reports each LM's mean
    per-reader Δ so the direction is readable off the row.
    """
    rows = []
    for measure, g in best.groupby("measure"):
        g = g.set_index("model_lm")
        deltas = {
            lm: _reader_delta(reader_ll, lm, measure, g.loc[lm, "index"])
            for lm in g.index
        }
        for lm1, lm2 in combinations(g.index, 2):
            d1, d2 = deltas[lm1].align(deltas[lm2], join="inner")
            diff = d1 - d2
            t, p_t = stats.ttest_rel(d1, d2)
            try:
                w, p_w = stats.wilcoxon(diff)
            except ValueError:  # all-zero differences
                w, p_w = np.nan, np.nan
            rows.append(
                {
                    "measure": measure,
                    "lm1": lm1,
                    "lm2": lm2,
                    "n_readers": len(diff),
                    "mean_delta_lm1": d1.mean(),
                    "mean_delta_lm2": d2.mean(),
                    "mean_diff": diff.mean(),
                    "t": t,
                    "p_t": p_t,
                    "wilcoxon": w,
                    "p_wilcoxon": p_w,
                }
            )
    return pd.DataFrame(rows)


def run_claim_tests(
    results: pd.DataFrame, reader_ll: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best-checkpoint selection + both cross-LM tests.

    Returns ``(best, cross)``: ``best`` one row per LM × measure (single LRT at
    the selected checkpoint, slope CI); ``cross`` one row per LM pair × measure
    × test (slope-difference z + paired reader test merged long-form).
    """
    best = best_checkpoints(results)
    slope = slope_difference(best)
    paired = reader_pair_test(reader_ll, best)
    cross = slope.merge(paired, on=["measure", "lm1", "lm2"], how="outer")
    return best, cross
