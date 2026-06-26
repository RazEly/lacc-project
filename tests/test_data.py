"""Data loading helpers: expertise recompute + PoTeC NA handling."""
import pandas as pd

from src.features import data


def test_add_expertise_matches_discipline_to_domain(rm):
    out = data.add_expertise(rm)
    expert = out["reader_discipline_numeric"] == out["text_domain_numeric"]
    assert (out["is_expert"] == expert.astype(int)).all()
    assert set(out["is_expert"].unique()) <= {0, 1}


def test_add_expertise_does_not_mutate(rm):
    before = rm.copy()
    data.add_expertise(rm)
    pd.testing.assert_frame_equal(rm, before)
    assert "is_expert" not in rm.columns


def test_read_potec_keeps_german_word_null(tmp_path):
    # PoTeC contains the German word "null"; default pandas NA would drop it.
    f = tmp_path / "wf.tsv"
    f.write_text("word\tword_length\nnull\t4\nHaus\t4\n")
    df = data.read_potec(f)
    # "null" stays a real string, not NaN.
    assert df["word"].tolist() == ["null", "Haus"]
    assert not df["word"].isna().any()


def test_read_potec_empty_string_is_na(tmp_path):
    f = tmp_path / "wf.tsv"
    f.write_text("word\tx\nA\t\nB\t5\n")
    df = data.read_potec(f)
    assert pd.isna(df["x"].iloc[0])
    assert df["x"].iloc[1] == 5
