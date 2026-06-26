"""Word-level surprisal (step 2), exercised with the fake uniform LM.

With FakeCausalLM every next-token distribution is uniform over `vocab`, so each
sub-token's surprisal is exactly log2(vocab). A word's surprisal is therefore
(#counted sub-tokens) * log2(vocab) — which makes the alignment + offset logic
checkable to the bit.
"""
import math

import numpy as np
import pandas as pd

from src.features import surprisal as su


def test_resolve_prompt_variants():
    assert su._resolve_prompt(None, "physics") is None
    assert su._resolve_prompt("hi", "physics") == "hi"
    d = {"physics": "P", "biology": "B"}
    assert su._resolve_prompt(d, "physics") == "P"
    assert su._resolve_prompt(d, "biology") == "B"
    assert su._resolve_prompt(d, "unknown") is None


def test_sentence_surprisal_first_word_none_and_subtoken_sum(fake_causal_lm, fake_tokenizer):
    bits = math.log2(fake_causal_lm.vocab)
    # "b_c" is a two-subtoken word.
    out = su.sentence_surprisal(["a", "b_c", "d"], fake_causal_lm, fake_tokenizer)
    assert out[0] is None  # first word, single subtoken, no left context
    assert abs(out[1] - 2 * bits) < 1e-4  # "b_c": two subtokens
    assert abs(out[2] - 1 * bits) < 1e-4


def test_sentence_surprisal_prompt_gives_first_word_context(fake_causal_lm, fake_tokenizer):
    bits = math.log2(fake_causal_lm.vocab)
    out = su.sentence_surprisal(
        ["a", "b", "c"], fake_causal_lm, fake_tokenizer, prompt="P"
    )
    # with a prompt prepended the first word now has left context.
    assert out[0] is not None
    assert abs(out[0] - bits) < 1e-4


def test_compute_surprisal_drops_edges_and_columns(fake_causal_lm, fake_tokenizer, words_df):
    out = su.compute_surprisal(words_df, fake_causal_lm, fake_tokenizer)
    assert list(out.columns) == su.WORD_KEY + ["surprisal"]
    # sentence-initial / final words dropped; interior words (idx 1,2,3) kept.
    kept = set(out["word_index_in_text"].unique())
    assert kept == {1, 2, 3}
    assert np.isfinite(out["surprisal"]).all()
    assert (out["surprisal"] > 0).all()


def test_compute_surprisal_handles_empty_words(fake_causal_lm, fake_tokenizer, words_df):
    # NaN words must not crash (fillna('') in compute_surprisal).
    words_df = words_df.copy()
    words_df.loc[words_df["word_index_in_text"] == 2, "word"] = np.nan
    out = su.compute_surprisal(words_df, fake_causal_lm, fake_tokenizer)
    assert isinstance(out, pd.DataFrame)
