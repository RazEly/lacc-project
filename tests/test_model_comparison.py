"""Surprisal-source comparison (step 5): data prep + paper Eq. 2/4/5 fits."""
import numpy as np

from src.analysis import model_comparison as mc


def test_prep_models_builds_surprisal_columns(mc_inputs):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    for col in ("S_baseline", "S_physics", "S_biology", "S_aligned", "S_prompted",
                "S_prompt_neutral"):
        assert col in d.columns
    # reader-aligned = discipline-matched surprisal.
    physicist = d["reader_discipline_numeric"] == 1
    np.testing.assert_allclose(
        d.loc[physicist, "S_aligned"], d.loc[physicist, "S_physics"]
    )
    np.testing.assert_allclose(
        d.loc[~physicist, "S_aligned"], d.loc[~physicist, "S_biology"]
    )
    # every continuous predictor scaled + centered (paper), incl. each surprisal.
    for col in ("z_length", "z_logfreq", "z_position", "z_S_baseline", "z_S_aligned"):
        assert col in d.columns
        assert abs(d[col].mean()) < 1e-9
        assert abs(d[col].std() - 1) < 1e-9
    # factors sum-coded (−1 / +1).
    assert set(np.unique(d["is_expert"])) <= {-1, 1}
    assert set(np.unique(d["is_technical"])) <= {-1, 1}
    assert (d["TFT"] > 0).all()


def test_build_index_df_base_uses_index_zero(mc_inputs):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    # at index 1 the physics-adapted surprisal differs from the index-0 baseline.
    assert not np.allclose(d["S_baseline"], d["S_physics"])


def test_fit_returns_loglik(mc_inputs):
    d = mc.build_index_df(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        index=1, measure="TFT",
    )
    res = mc._fit(d, "TFT", "z_S_baseline")
    assert np.isfinite(mc._loglik(res))
    assert "z_S_baseline" in res.result_fit["term"].to_list()


def test_model_comparison_over_epochs_paper(mc_inputs):
    # most sources scored by ΔLL over the no-surprisal baseline; the reader-
    # conditioned arms (aligned/prompted) over the surprisal baseline.
    out, reader_ll = mc.model_comparison_over_epochs(
        mc_inputs["surp_versions"], mc_inputs["rt_df"], mc_inputs["prompt_surp"],
        measure="TFT",
    )
    expected = {"index", "epoch", "ref", "model", "n", "ll", "delta_ll",
                "b_surprisal", "se_surprisal", "p_surprisal", "p_lrt"}
    assert expected <= set(out.columns)
    assert set(out["model"].unique()) == set(mc.SURPRISAL_MODELS)
    # aligned/prompted referenced to the surprisal baseline; the rest to no-surprisal.
    ref_by_model = out.set_index("model")["ref"].to_dict()
    for m in mc.SURPRISAL_MODELS:
        want = "surprisal_baseline" if m in mc._VS_SURP_BASELINE else "no_surprisal"
        assert ref_by_model[m] == want, m
    assert np.isfinite(out["delta_ll"]).all()
    # 1-df nested LRT: p in [0, 1].
    assert out["p_lrt"].between(0, 1).all()
    # per-reader conditional LLs collected for the baseline reference + every fit.
    assert {"model", "index", "reader_id", "ll_reader"} <= set(reader_ll.columns)
    assert "base_ref" in set(reader_ll["model"])
    assert "aligned" in set(reader_ll["model"])
    assert np.isfinite(reader_ll["ll_reader"]).all()
