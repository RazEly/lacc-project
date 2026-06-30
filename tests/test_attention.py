"""Attention representations (step 3): pure helpers + raw attention + PCA."""
import numpy as np
import pandas as pd

from src.features import attention as at


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_word_positions_groups_subtokens_and_skips_none():
    word_ids = [None, 0, 0, 1, 2, 2]
    assert at._word_positions(word_ids, 3) == [[1, 2], [3], [4, 5]]


def test_normalize_sums_to_one():
    out = at._normalize(np.array([1.0, 3.0]))
    assert abs(out.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(out, [0.25, 0.75])


def test_normalize_all_zero_is_unchanged():
    z = np.zeros(3)
    np.testing.assert_array_equal(at._normalize(z), z)


# ── model-facing wrappers via fake attention LM ──────────────────────────────
def test_raw_attention_shape_and_normalized(fake_attn_lm, fake_tokenizer):
    out = at.raw_attention(["w0", "w1", "w2", "w3"], fake_attn_lm, fake_tokenizer)
    assert out.shape == (4, fake_attn_lm.n_layers)
    for l in range(fake_attn_lm.n_layers):
        assert abs(out[:, l].sum() - 1.0) < 1e-6  # float32 attentions


def test_extract_attention_long_form_drops_edges(fake_attn_lm, fake_tokenizer, words_df):
    out = at.extract_attention(words_df, fake_attn_lm, fake_tokenizer, method="raw")
    assert list(out.columns) == at.WORD_KEY + ["layer", "attention", "attention_method"]
    assert set(out["word_index_in_text"].unique()) == {1, 2, 3}  # edges dropped
    assert out["attention_method"].unique().tolist() == ["raw"]
    assert out["layer"].nunique() == fake_attn_lm.n_layers


# ── eye-tracking features + PCA ───────────────────────────────────────────────
def test_eyetracking_features_one_row_per_word(rm):
    et = at.eyetracking_features(rm)
    assert len(et) == rm[at.WORD_KEY].drop_duplicates().shape[0]
    for m in at.ET_MEASURE_MAP.values():
        assert m in et.columns
    assert {"text_domain", "sent_index_in_text"} <= set(et.columns)


def test_eyetracking_features_excludes_skips_from_mean(rm):
    # word idx2 in b1 was skipped (0) by reader rb2; mean must ignore the zero.
    # relative=False isolates the skip-exclusion from the sentence-share rescale.
    et = at.eyetracking_features(rm, relative=False)
    row = et[(et["text_id"] == "b1") & (et["word_index_in_text"] == 2)]
    nonzero_mean = rm[(rm["text_id"] == "b1") & (rm["word_index_in_text"] == 2) & (rm["TFT"] > 0)]["TFT"].mean()
    assert abs(row["TFT"].iloc[0] - nonzero_mean) < 1e-6


def test_pca_eyetracking_returns_scores_and_variance(rm):
    et = at.eyetracking_features(rm)
    scored, evr = at.pca_eyetracking(et, "physics")
    assert list(scored.columns) == at.WORD_KEY + ["pca"]
    assert 0.0 <= evr <= 1.0
    assert np.isfinite(scored["pca"]).all()
