"""Surprisal-source comparison (step 5): data prep + paper Eq. 2/4/5 fits."""
import numpy as np
import pytest

from src.analysis import model_comparison as mc


@pytest.fixture
def index_df(mc_inputs):
    """Prepared reader×word frame at checkpoint index 1."""
    return mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )


def test_prep_models_builds_surprisal_columns(index_df):
    d = index_df
    assert {f"s_{m}" for m in mc.SURPRISAL_MODELS} <= set(d.columns)
    # reader-aligned = discipline-matched surprisal.
    physicist = d["reader_discipline_numeric"] == 1
    np.testing.assert_allclose(
        d["s_aligned"], d["s_physics"].where(physicist, d["s_biology"])
    )
    # continuous covariates present and finite.
    assert np.isfinite(d[["word_length", "log_word_freq", "word_position"]]).all().all()
    # factors sum-coded (−1 / +1); rows with non-positive RT dropped.
    assert set(np.unique(d[["is_expert", "is_technical"]])) <= {-1, 1}
    assert (d["TFT"] > 0).all()


def test_build_index_df_base_uses_index_zero(index_df):
    # at index 1 the physics-adapted surprisal differs from the index-0 baseline.
    assert not np.allclose(index_df["s_baseline"], index_df["s_physics"])


def test_fit_returns_loglik(index_df):
    res = mc._fit(index_df, "TFT", "s_baseline")
    assert np.isfinite(mc._stat(res, "logLik"))
    assert "s_baseline" in res.result_fit["term"].to_list()


def test_model_comparison_over_steps_paper(mc_inputs):
    # every source scored against the shared no-surprisal baseline (LRT + AIC);
    # Vuong table compares baseline-surprisal vs the reader-conditioned arms.
    out, vuong, reader_ll = mc.model_comparison_over_steps(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        measure="TFT",
    )
    expected = {"index", "training_steps", "model", "n", "ll", "delta_ll", "chisq",
                "p_lrt", "aic", "b_surprisal", "se_surprisal"}
    assert expected <= set(out.columns)
    assert set(out["model"]) == set(mc.SURPRISAL_MODELS)
    assert np.isfinite(out["delta_ll"]).all()
    np.testing.assert_allclose(out["chisq"], 2.0 * out["delta_ll"])
    assert np.isfinite(out["aic"]).all()
    # LRT vs the null: p in [0, 1].
    assert out["p_lrt"].between(0, 1).all()
    # Vuong: one row per reader-conditioned arm (aligned per checkpoint).
    assert {"index", "training_steps", "model", "n_readers", "vuong_z", "p_vuong"} <= set(
        vuong.columns
    )
    assert set(vuong["model"]) == mc._VUONG_ARMS
    n_aligned = out.loc[out["model"] == "aligned", "index"].nunique()
    assert len(vuong) == n_aligned + 2  # + prompted + prompt_neutral
    # aligned at checkpoint 0 duplicates the baseline model -> undefined (NaN).
    ok = vuong["vuong_z"].notna()
    assert ok.any()
    assert np.isfinite(vuong.loc[ok, "vuong_z"]).all()
    assert vuong.loc[ok, "p_vuong"].between(0, 1).all()
    # per-reader conditional LLs collected for the baseline reference + every fit.
    assert {"model", "index", "reader_id", "ll_reader"} <= set(reader_ll.columns)
    assert {"base_ref", *mc.SURPRISAL_MODELS} <= set(reader_ll["model"])
    assert np.isfinite(reader_ll["ll_reader"]).all()
