"""Attention representations (step 3): pure helpers + flow algorithm + PCA."""
import networkx as nx
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


# ── flow network ─────────────────────────────────────────────────────────────
def test_add_flow_layer_capacities_form_distribution():
    G = nx.DiGraph()
    seq = 3
    A = np.array([[0.2, 0.8, 0.0], [0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
    at._add_flow_layer(G, A, l=0, seq=seq, residual=0.5)
    # incoming capacities to each receiver node (l+1, i) sum to ~1 (row-normalized).
    for i in range(seq):
        incoming = sum(d["capacity"] for _, _, d in G.in_edges((1, i), data=True))
        assert abs(incoming - 1.0) < 1e-9


def test_flow_scores_shape_and_normalization():
    n_layers, seq = 3, 5
    rng = np.random.default_rng(1)
    atts = rng.random((n_layers, seq, seq))
    atts = atts / atts.sum(-1, keepdims=True)
    word_ids = [0, 0, 1, 2, 3]  # 4 words, first has 2 subtokens
    out = at._flow_scores(atts, word_ids, n_words=4, decay=1.0)
    assert out.shape == (4, n_layers)
    # each layer column sum-normalized to 1.
    for l in range(n_layers):
        assert abs(out[:, l].sum() - 1.0) < 1e-9
    assert np.isfinite(out).all()


def test_flow_scores_decay_zero_runs():
    atts = np.full((2, 4, 4), 0.25)
    out = at._flow_scores(atts, [0, 1, 2, 3], n_words=4, decay=0.0)
    assert out.shape == (4, 2)


# ── model-facing wrappers via fake attention LM ──────────────────────────────
def test_raw_attention_shape_and_normalized(fake_attn_lm, fake_tokenizer):
    out = at.raw_attention(["w0", "w1", "w2", "w3"], fake_attn_lm, fake_tokenizer)
    assert out.shape == (4, fake_attn_lm.n_layers)
    for l in range(fake_attn_lm.n_layers):
        assert abs(out[:, l].sum() - 1.0) < 1e-6  # float32 attentions


def test_attention_flow_wrapper_shape(fake_attn_lm, fake_tokenizer):
    out = at.attention_flow(["w0", "w1", "w2", "w3"], fake_attn_lm, fake_tokenizer)
    assert out.shape == (4, fake_attn_lm.n_layers)


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
    et = at.eyetracking_features(rm)
    row = et[(et["text_id"] == "b1") & (et["word_index_in_text"] == 2)]
    nonzero_mean = rm[(rm["text_id"] == "b1") & (rm["word_index_in_text"] == 2) & (rm["TFT"] > 0)]["TFT"].mean()
    assert abs(row["TFT"].iloc[0] - nonzero_mean) < 1e-6


def test_pca_eyetracking_returns_scores_and_variance(rm):
    et = at.eyetracking_features(rm)
    scored, evr = at.pca_eyetracking(et, "physics")
    assert list(scored.columns) == at.WORD_KEY + ["pca"]
    assert 0.0 <= evr <= 1.0
    assert np.isfinite(scored["pca"]).all()
