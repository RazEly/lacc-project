"""Reading-time cleaning + aggregation (step 1)."""
import numpy as np
import pandas as pd

from src.features import reading_time as rt


def test_clean_drops_sentence_edges_and_skips(rm):
    out = rt.clean_reading_times(rm, measure="TFT")
    # sentence-initial / final words gone.
    assert (out["is_sent_beginning"] != 1).all()
    assert (out["is_sent_end"] != 1).all()
    # skipped words (TFT == 0) gone.
    assert (out["TFT"] > 0).all()


def test_clean_iqr_removes_outlier():
    # one interior word, many readers, one gross outlier; min_count default 4.
    n = 20
    df = pd.DataFrame(
        {
            "text_id": ["t"] * n,
            "word_index_in_text": [1] * n,
            "is_sent_beginning": [0] * n,
            "is_sent_end": [0] * n,
            "TFT": [200] * (n - 1) + [100000],  # last = outlier
        }
    )
    out = rt.clean_reading_times(df, measure="TFT")
    assert 100000 not in out["TFT"].values
    assert len(out) == n - 1


def test_clean_iqr_skipped_when_below_min_count():
    # only 3 obs (< min_count=4) -> fence not applied, outlier survives.
    df = pd.DataFrame(
        {
            "text_id": ["t"] * 3,
            "word_index_in_text": [1] * 3,
            "is_sent_beginning": [0] * 3,
            "is_sent_end": [0] * 3,
            "TFT": [200, 210, 99999],
        }
    )
    out = rt.clean_reading_times(df, measure="TFT")
    assert len(out) == 3


def test_clean_does_not_mutate_input(rm):
    before = rm.copy()
    rt.clean_reading_times(rm, measure="TFT")
    pd.testing.assert_frame_equal(rm, before)


def test_aggregate_rt_groups_and_columns(rm):
    from src.features.data import add_expertise

    cleaned = add_expertise(rt.clean_reading_times(rm, measure="TFT"))
    groups = rt.aggregate_rt(cleaned, measure="TFT")
    assert "all" in groups and "experts" in groups
    cols = set(groups["all"].columns)
    assert {"text_id", "word_index_in_text", "mean_TFT", "std_TFT", "median_TFT", "n"} <= cols
    # one row per surviving word.
    assert len(groups["all"]) == cleaned[rt.WORD_KEY].drop_duplicates().shape[0]


def test_aggregate_experts_subset_of_all(rm):
    cleaned = rt.clean_reading_times(rm, measure="TFT")
    # add_expertise flag needed by aggregate_rt.
    from src.features.data import add_expertise

    cleaned = add_expertise(cleaned)
    groups = rt.aggregate_rt(cleaned, measure="TFT")
    # mean over a reader subset can't exceed the global max count.
    assert groups["experts"]["n"].max() <= groups["all"]["n"].max()
    assert np.isfinite(groups["all"]["mean_TFT"]).all()
