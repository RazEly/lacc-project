"""Significance machinery: BH correction + reader-clustered Vuong test."""
import numpy as np

from src.analysis import stats
from src.analysis import model_comparison as mc


# ── Benjamini-Hochberg ───────────────────────────────────────────────────────
def test_bh_correct_preserves_nan_and_length():
    p = [0.01, np.nan, 0.04, 0.03]
    reject, p_adj = stats.bh_correct(p)
    assert len(reject) == len(p) == len(p_adj)
    assert np.isnan(p_adj[1]) and not reject[1]
    assert reject.dtype == bool


def test_bh_correct_adjusted_not_below_raw():
    p = np.array([0.001, 0.01, 0.02, 0.03])
    _, p_adj = stats.bh_correct(p)
    assert np.all(p_adj >= p - 1e-12)
    assert np.all(p_adj <= 1.0 + 1e-12)


def test_bh_correct_all_nan():
    reject, p_adj = stats.bh_correct([np.nan, np.nan])
    assert not reject.any()
    assert np.isnan(p_adj).all()


# ── Vuong test ───────────────────────────────────────────────────────────────
def _fit_two(mc_inputs, col_a, col_b):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    a = mc._fit_model(d, "TFT", col_a, spec="covariates")
    b = mc._fit_model(d, "TFT", col_b, spec="covariates")
    return a, b


def test_pointwise_loglik_columns_and_readers(mc_inputs):
    a, _ = _fit_two(mc_inputs, "S_aligned", "S_baseline")
    out = stats._pointwise_loglik(a)
    assert set(out.columns) == {"_row", "reader_id", "ll"}
    assert np.isfinite(out["ll"]).all()
    assert out["reader_id"].nunique() == mc_inputs["rt_df"]["reader_id"].nunique()


def test_vuong_test_keys_and_ranges(mc_inputs):
    a, b = _fit_two(mc_inputs, "S_aligned", "S_baseline")
    res = stats.vuong_test(a, b)
    assert set(res) == {"n_readers", "mean_dll", "z", "p"}
    assert res["n_readers"] == mc_inputs["rt_df"]["reader_id"].nunique()
    assert 0.0 <= res["p"] <= 1.0
    assert np.isfinite(res["z"])


def test_vuong_identical_models_zero_z(mc_inputs):
    a, b = _fit_two(mc_inputs, "S_baseline", "S_baseline")
    res = stats.vuong_test(a, b)
    # same model on both sides -> per-reader differences all zero -> z == 0, p == 1.
    assert abs(res["mean_dll"]) < 1e-9
    assert res["z"] == 0.0
    assert res["p"] == 1.0


def test_aligned_vs_single_models_table(mc_inputs):
    out = stats.aligned_vs_single_models(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        measure="TFT", index=1, spec="covariates",
    )
    assert set(out["comparison"]) == {
        "aligned vs baseline", "aligned vs physics",
        "aligned vs biology", "aligned vs prompted",
        "aligned vs prompted_level",
    }
    assert {"z", "p", "p_adj", "significant", "winner"} <= set(out.columns)
    assert out["p"].between(0, 1).all()
