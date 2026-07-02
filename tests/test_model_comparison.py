"""Surprisal-model comparison (step 5): data prep + mixed fits."""
import numpy as np

from src.analysis import model_comparison as mc


def test_prep_models_builds_surprisal_columns(mc_inputs):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    for col in ("S_baseline", "S_physics", "S_biology", "S_aligned", "S_prompted"):
        assert col in d.columns
    # reader-aligned = discipline-matched surprisal.
    physicist = d["reader_discipline_numeric"] == 1
    np.testing.assert_allclose(
        d.loc[physicist, "S_aligned"], d.loc[physicist, "S_physics"]
    )
    np.testing.assert_allclose(
        d.loc[~physicist, "S_aligned"], d.loc[~physicist, "S_biology"]
    )
    # derived predictors present; continuous covariates scaled + centered.
    assert {"log_word_freq", "is_technical"} <= set(d.columns)
    for col in ("word_length", "log_word_freq"):
        assert abs(d[col].mean()) < 1e-9
        assert abs(d[col].std() - 1) < 1e-9
    assert (d["TFT"] > 0).all()


def test_build_index_df_base_uses_index_zero(mc_inputs):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    # at index 1 the physics-adapted surprisal differs from the index-0 baseline.
    assert not np.allclose(d["S_baseline"], d["S_physics"])


def test_fit_model_returns_loglik(mc_inputs):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    res = mc._fit_model(d, "TFT", "S_baseline")
    assert np.isfinite(mc._loglik(res))
    assert "S_baseline" in res.result_fit["term"].to_list()


def test_model_comparison_over_epochs_residual_default(mc_inputs):
    # split-signal residual mode: baseline vs null, the rest vs base surprisal.
    out, reader_ll = mc.model_comparison_over_epochs(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        measure="TFT",
    )
    expected = {"index", "epoch", "ref", "model", "n", "ll", "delta_ll",
                "b_surprisal", "se_surprisal", "p_surprisal",
                "b_resid", "se_resid", "p_resid", "p_lrt"}
    assert expected <= set(out.columns)
    assert set(out["model"].unique()) == set(mc.SURPRISAL_MODELS)
    # baseline row = standard LRT vs the no-surprisal null; others residual-split.
    assert (out.loc[out["model"] == "baseline", "ref"] == "null").all()
    assert (out.loc[out["model"] != "baseline", "ref"] == "base_surprisal").all()
    # at index 0 the checkpoint-derived surprisals equal the base, so their
    # residual D == 0 and the experimental model collapses onto the reference
    # -> delta_ll == 0. (prompted residuals are not checkpoint-dependent.)
    i0 = out[out["index"] == 0].set_index("model")["delta_ll"]
    np.testing.assert_allclose(
        [i0["physics"], i0["biology"], i0["aligned"]], 0.0, atol=1e-4
    )
    assert np.isfinite(out["delta_ll"]).all()
    # per-reader conditional LLs collected for the base reference + every fit.
    assert {"model", "index", "reader_id", "ll_reader"} <= set(reader_ll.columns)
    assert "base_ref" in set(reader_ll["model"])
    assert "aligned" in set(reader_ll["model"])
    assert np.isfinite(reader_ll["ll_reader"]).all()
