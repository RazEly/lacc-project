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


def test_as_prompt_list_normalises():
    assert su._as_prompt_list(None) == []
    assert su._as_prompt_list("") == []
    assert su._as_prompt_list("P") == ["P"]
    assert su._as_prompt_list(["P1", "P2"]) == ["P1", "P2"]


def test_mean_bits_ignores_none_elementwise():
    out = su._mean_bits([[1.0, None, 4.0], [3.0, 2.0, None]])
    assert out[0] == 2.0  # (1+3)/2
    assert out[1] == 2.0  # only second prior present
    assert out[2] == 4.0  # only first prior present
    # a word None under every prior stays None.
    assert su._mean_bits([[None], [None]]) == [None]


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


def test_compute_surprisal_scores_each_sentence_independently(
    fake_causal_lm, fake_tokenizer
):
    # one 6-word text split into TWO sentences (words 0-2, 3-5). Per-sentence
    # scoring resets context at the boundary, so EACH sentence's first word has
    # no left context and drops out: word 0 (text+sent first) AND word 3 (second
    # sentence's first). The text-final word 5 drops for RT cleaning. Kept: 1,2,4.
    rows = [
        {
            "text_id": "t",
            "text_domain": "physics",
            "sent_index_in_text": 0 if wi < 3 else 1,
            "word_index_in_text": wi,
            "word_index_in_sent": wi % 3,
            "word": f"w{wi}",
            "is_sent_beginning": int(wi in (0, 3)),
            "is_sent_end": int(wi in (2, 5)),
        }
        for wi in range(6)
    ]
    out = su.compute_surprisal(pd.DataFrame(rows), fake_causal_lm, fake_tokenizer)
    assert set(out["word_index_in_text"]) == {1, 2, 4}


def test_compute_surprisal_averages_over_prior_list(fake_causal_lm, fake_tokenizer, words_df):
    # uniform LM ignores prior content, so every prior yields the same surprisal;
    # averaging a list of priors must reproduce the single-prior column exactly.
    single = su.compute_surprisal(words_df, fake_causal_lm, fake_tokenizer, prompt="P")
    averaged = su.compute_surprisal(
        words_df, fake_causal_lm, fake_tokenizer, prompt=["P1", "P2", "P3"]
    )
    merged = single.merge(averaged, on=su.WORD_KEY, suffixes=("_1", "_avg"))
    assert np.allclose(merged["surprisal_1"], merged["surprisal_avg"])


def test_stimuli_perplexity_recovers_per_token_mean(
    fake_causal_lm, fake_tokenizer, words_df
):
    # uniform LM: every sub-token costs exactly log2(vocab)=6 bits, so the
    # per-token perplexity must come out at exactly 2**6 for every checkpoint
    # and text domain — including with a multi-sub-token word (split on "_"),
    # which only works if word bits and token counts stay aligned.
    words_df = words_df.copy()
    words_df.loc[
        (words_df["text_id"] == "p1") & (words_df["word_index_in_text"] == 2), "word"
    ] = "a_b_c"
    sup = su.compute_surprisal(words_df, fake_causal_lm, fake_tokenizer)
    sv = pd.concat(
        [sup.assign(index=0, domain="physics"), sup.assign(index=1, domain="physics")],
        ignore_index=True,
    )
    out = su.stimuli_perplexity(sv, words_df, fake_tokenizer)
    assert list(out.columns) == ["domain", "index", "text_domain", "perplexity"]
    # 1 ft-domain × 2 checkpoints × 2 text domains
    assert len(out) == 4
    assert np.allclose(out["perplexity"], 2**6)


def test_compute_surprisal_handles_empty_words(fake_causal_lm, fake_tokenizer, words_df):
    # NaN words must not crash (fillna('') in compute_surprisal).
    words_df = words_df.copy()
    words_df.loc[words_df["word_index_in_text"] == 2, "word"] = np.nan
    out = su.compute_surprisal(words_df, fake_causal_lm, fake_tokenizer)
    assert isinstance(out, pd.DataFrame)
