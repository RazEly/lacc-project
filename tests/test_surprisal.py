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


def test_score_words_first_word_none_and_subtoken_sum(fake_causal_lm, fake_tokenizer):
    bits = math.log2(fake_causal_lm.vocab)
    # "b_c" is a two-subtoken word.
    out = su.score_words(["a", "b_c", "d"], fake_causal_lm, fake_tokenizer)
    assert out[0] is None  # first word, single subtoken, no left context
    assert abs(out[1] - 2 * bits) < 1e-4  # "b_c": two subtokens
    assert abs(out[2] - 1 * bits) < 1e-4


def test_score_words_prompt_gives_first_word_context(fake_causal_lm, fake_tokenizer):
    bits = math.log2(fake_causal_lm.vocab)
    out = su.score_words(
        ["a", "b", "c"], fake_causal_lm, fake_tokenizer, prompt="P"
    )
    # with a prior + document boundary prepended, the first word now has context.
    assert out[0] is not None
    assert abs(out[0] - bits) < 1e-4


def test_score_words_inserts_native_document_boundary(monkeypatch, fake_causal_lm,
                                                      fake_tokenizer):
    # the prior and the stimulus must be joined by the tokenizer's eos id, not a
    # newline: capture what the model actually receives.
    seen = {}
    orig = fake_causal_lm.forward

    def spy(input_ids):
        seen["ids"] = input_ids[0].tolist()
        return orig(input_ids)

    monkeypatch.setattr(fake_causal_lm, "forward", spy)
    su.score_words(["a", "b"], fake_causal_lm, fake_tokenizer, prompt="P")
    ids = seen["ids"]
    eos = fake_tokenizer.eos_token_id
    # sequence = [prior "P"] + [eos boundary] + [stimulus a, b]; exactly one eos,
    # sitting between the prior and the first stimulus token.
    assert ids.count(eos) == 1
    assert ids[1] == eos
    assert len(ids) == 4


def test_score_words_truncates_prior_to_budget(fake_causal_lm, fake_tokenizer):
    # a long prior is capped to max_prompt_tokens before the boundary.
    long_prior = " ".join(f"w{i}" for i in range(20))
    out = su.score_words(
        ["a", "b"], fake_causal_lm, fake_tokenizer,
        prompt=long_prior, max_prompt_tokens=5,
    )
    assert len(out) == 2 and out[0] is not None


def test_doc_separator_id_requires_special_token(fake_tokenizer):
    fake_tokenizer.eos_token_id = None
    try:
        su._doc_separator_id(fake_tokenizer)
        assert False, "expected ValueError when no eos/sep token exists"
    except ValueError:
        pass
    finally:
        fake_tokenizer.eos_token_id = 0


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
