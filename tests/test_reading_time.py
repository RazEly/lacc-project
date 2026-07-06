"""Reading-time cleaning (step 1)."""
import pandas as pd

from src.features import dataset as ds


def test_clean_drops_sentence_edges_and_skips(rm):
    out = ds.clean_reading_times(rm, measure="TFT")
    # sentence-initial / final words gone.
    assert (out["is_sent_beginning"] != 1).all()
    assert (out["is_sent_end"] != 1).all()
    # skipped words (TFT == 0) gone.
    assert (out["TFT"] > 0).all()


def test_clean_sd_fence_removes_outlier():
    # one reader, many interior words, one gross outlier; ±3 SD fence drops it.
    n = 42
    tft = [200 + i for i in range(n)]
    tft[20] = 100000  # interior outlier (text-edge words are dropped anyway)
    df = pd.DataFrame(
        {
            "text_id": ["t"] * n,
            "word_index_in_text": list(range(n)),
            "reader_id": ["r1"] * n,
            "TFT": tft,
        }
    )
    out = ds.clean_reading_times(df, measure="TFT")
    assert 100000 not in out["TFT"].values
    assert len(out) == n - 3  # first + last text word + the outlier


def test_clean_sd_fence_keeps_single_observation():
    # one interior word for the reader -> SD is NaN -> fence not applied.
    df = pd.DataFrame(
        {
            "text_id": ["t"] * 3,
            "word_index_in_text": [0, 1, 2],
            "reader_id": ["r1"] * 3,
            "TFT": [200, 99999, 210],
        }
    )
    out = ds.clean_reading_times(df, measure="TFT")
    assert out["TFT"].tolist() == [99999]


def test_clean_does_not_mutate_input(rm):
    before = rm.copy()
    ds.clean_reading_times(rm, measure="TFT")
    pd.testing.assert_frame_equal(rm, before)
