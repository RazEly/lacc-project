"""Regression of model scores vs reading time (step 5)."""
import pytest

from src.features.data import add_expertise
from src.analysis import correlation as co


@pytest.fixture
def rt_df(rm):
    return add_expertise(rm)


@pytest.fixture
def surprisal_df(rm):
    """Per-word surprisal for interior words, rising with word index."""
    words = rm[co.WORD_KEY].drop_duplicates()
    words = words[words["word_index_in_text"].isin([1, 2, 3])].copy()
    words["surprisal"] = words["word_index_in_text"].astype(float) + 1.0
    return words.reset_index(drop=True)


def test_filter_helpers(rt_df):
    assert (co._filter_domain(rt_df, "physics")["text_domain"] == "physics").all()
    assert len(co._filter_domain(rt_df, "all")) == len(rt_df)
    assert (co._filter_participants(rt_df, "experts")["is_expert"] == 1).all()
    assert (co._filter_participants(rt_df, "novices")["is_expert"] == 0).all()


def test_merge_surprisal_rt_inner(surprisal_df, rt_df):
    merged = co.merge_surprisal_rt(surprisal_df, rt_df)
    # only interior words survive the inner join.
    assert set(merged["word_index_in_text"].unique()) == {1, 2, 3}
    assert "surprisal" in merged and "TFT" in merged


def test_regress_rt_slope_positive(surprisal_df, rt_df):
    merged = co.merge_surprisal_rt(surprisal_df, rt_df)
    fit = co.regress_rt(merged, measure="TFT")
    assert "surprisal" in fit.params.index
    assert fit.params["surprisal"] > 0  # higher surprisal -> longer RT
