"""Four/five-model surprisal comparison (step 5): data prep + mixed fits."""
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
    # derived predictors present.
    assert {"log_word_freq", "log_word_length", "is_technical", "is_domain_term"} <= set(d.columns)
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
    res = mc._fit_model(d, "TFT", "S_baseline", spec="covariates")
    assert np.isfinite(res.logLike)
    assert "S_baseline" in res.coefs.index


def test_model_comparison_over_epochs_columns_and_baseline(mc_inputs):
    out = mc.model_comparison_over_epochs(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        measure="TFT", spec="covariates",
    )
    expected = {"index", "epoch", "spec", "model", "n", "ll", "delta_ll", "b_surprisal", "p_surprisal"}
    assert expected <= set(out.columns)
    assert set(out["model"].unique()) == set(mc.SURPRISAL_MODELS)
    assert set(out["index"].unique()) == {0, 1}
    # at index 0, baseline / physics / biology / aligned share the same surprisal,
    # so they fit identically -> equal log-likelihood.
    i0 = out[out["index"] == 0].set_index("model")["ll"]
    assert i0["baseline"] == i0["physics"] == i0["biology"] == i0["aligned"]
    assert np.isfinite(out["delta_ll"]).all()
