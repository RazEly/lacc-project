"""Correlation / regression of model scores vs reading time (step 5)."""
import numpy as np
import pandas as pd
import pytest

from src.features.data import add_expertise
from src.features import attention as at
from src.analysis import correlation as co


@pytest.fixture
def rt_df(rm):
    return add_expertise(rm)


@pytest.fixture
def surprisal_df(rm):
    """Per-word surprisal for interior words, correlated with word index."""
    words = rm[at.WORD_KEY].drop_duplicates()
    words = words[words["word_index_in_text"].isin([1, 2, 3])].copy()
    words["surprisal"] = words["word_index_in_text"].astype(float) + 1.0
    return words.reset_index(drop=True)


def test_filter_helpers(rt_df):
    assert (co._filter_domain(rt_df, "physics")["text_domain"] == "physics").all()
    assert len(co._filter_domain(rt_df, "all")) == len(rt_df)
    assert (co._filter_participants(rt_df, "experts")["is_expert"] == 1).all()
    assert (co._filter_participants(rt_df, "novices")["is_expert"] == 0).all()
    only = co._filter_domain_only(rt_df, True)
    assert ((only["is_expert_technical_term"] == 1) | (only["is_general_technical_term"] == 1)).all()


def test_merge_surprisal_rt_inner(surprisal_df, rt_df):
    merged = co.merge_surprisal_rt(surprisal_df, rt_df)
    # only interior words survive the inner join.
    assert set(merged["word_index_in_text"].unique()) == {1, 2, 3}
    assert "surprisal" in merged and "TFT" in merged


def test_correlate_surprisal_keys_and_perfect_corr(surprisal_df, rt_df):
    merged = co.merge_surprisal_rt(surprisal_df, rt_df)
    res = co.correlate_surprisal(merged, measure="TFT")
    assert set(res) == {"n", "pearson", "pearson_p", "spearman", "spearman_p"}
    # surprisal == index+1 and mean TFT rises with index -> strong positive corr
    # (not exactly 1: two texts share surprisal values, so ranks tie).
    assert res["n"] >= 3
    assert res["pearson"] > 0.7
    assert res["spearman"] > 0.7


def test_correlate_surprisal_too_few_points_returns_nan(rt_df):
    tiny = pd.DataFrame(
        {"text_id": ["p1"], "word_index_in_text": [1], "surprisal": [2.0]}
    )
    merged = co.merge_surprisal_rt(tiny, rt_df)
    res = co.correlate_surprisal(merged, measure="TFT")
    assert np.isnan(res["pearson"])


def test_regress_rt_slope_positive(surprisal_df, rt_df):
    merged = co.merge_surprisal_rt(surprisal_df, rt_df)
    fit = co.regress_rt(merged, measure="TFT")
    assert "surprisal" in fit.params.index
    assert fit.params["surprisal"] > 0  # higher surprisal -> longer RT


def test_build_et_table_has_pca_column(rt_df):
    et = co.build_et_table(rt_df, domain="all", participants="all")
    assert "pca" in et.columns
    assert set(et["text_domain"].unique()) <= {"physics", "biology"}


def test_correlate_attention_layerwise(rt_df):
    et = co.build_et_table(rt_df)
    # build a small attention frame for interior words across 2 layers.
    rows = []
    words = et[at.WORD_KEY].drop_duplicates()
    for _, w in words.iterrows():
        for layer in (0, 1):
            rows.append(
                {
                    "text_id": w["text_id"],
                    "word_index_in_text": w["word_index_in_text"],
                    "layer": layer,
                    "attention": float(w["word_index_in_text"] + layer),
                    "attention_method": "raw",
                }
            )
    attn = pd.DataFrame(rows)
    out = co.correlate_attention(attn, et, layer="all", attention_method="raw")
    assert set(out.columns) == {"layer", "feature", "attention_method", "spearman", "p", "n"}
    assert set(out["layer"].unique()) == {0, 1}
    assert "pca" in out["feature"].unique()


def test_correlation_over_epochs_long_form(rt_df, surprisal_df):
    # two checkpoints x both domains.
    frames = []
    for epoch, idx in [(0.0, 0), (1.0, 1)]:
        for domain in ("physics", "biology"):
            sv = surprisal_df.copy()
            sv["epoch"] = epoch
            sv["domain"] = domain
            frames.append(sv)
    surp_versions = pd.concat(frames, ignore_index=True)
    out = co.correlation_over_epochs(surp_versions, rt_df, groups=("experts", "novices"))
    assert set(out.columns) == {"epoch", "domain", "group", "pearson", "spearman", "n"}
    assert set(out["group"].unique()) == {"experts", "novices"}
    assert set(out["epoch"].unique()) == {0.0, 1.0}
