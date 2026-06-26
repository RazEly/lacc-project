"""Shared fixtures: synthetic PoTeC-shaped frames + tiny fake LM/tokenizer.

Everything here is CPU-only and network-free. The fakes stand in for the HF
model + tokenizer so the surprisal / attention code paths can be exercised
without downloading weights or touching a GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# ── synthetic reading-measures frame ─────────────────────────────────────────
# Two texts (1 physics, 1 biology), one 5-word sentence each (word indices 0..4;
# 0 and 4 are sentence edges). Four readers: two physics-discipline, two biology.
TEXTS = [
    ("p1", "physics", 1),   # text_id, text_domain, text_domain_numeric
    ("b1", "biology", 0),
]
READERS = [
    ("rp1", 1, 1),  # reader_id, reader_discipline_numeric, level_of_studies_numeric
    ("rp2", 1, 0),
    ("rb1", 0, 1),
    ("rb2", 0, 0),
]
N_WORDS = 5


@pytest.fixture
def rm() -> pd.DataFrame:
    """Per reader×word reading-measures frame (pre-`add_expertise`, raw-ish)."""
    rng = np.random.default_rng(0)
    rows = []
    for text_id, domain, dom_num in TEXTS:
        for wi in range(N_WORDS):
            is_beg = int(wi == 0)
            is_end = int(wi == N_WORDS - 1)
            word_length = 3 + wi
            for reader_id, disc, level in READERS:
                # base reading time grows with word index + reader noise; a couple
                # of zeros injected to exercise the skip filter.
                base = 100 + 40 * wi + rng.integers(-10, 10)
                skip = (wi == 2 and reader_id == "rb2")  # one skipped word
                tft = 0 if skip else int(base)
                rows.append(
                    {
                        "text_id": text_id,
                        "text_domain": domain,
                        "text_domain_numeric": dom_num,
                        "sent_index_in_text": 0,
                        "word_index_in_text": wi,
                        "word_index_in_sent": wi,
                        "word": f"{text_id}w{wi}",
                        "word_length": word_length,
                        "is_sent_beginning": is_beg,
                        "is_sent_end": is_end,
                        "is_expert_technical_term": int(wi == 1),
                        "is_general_technical_term": int(wi == 3),
                        "reader_id": reader_id,
                        "reader_discipline_numeric": disc,
                        "level_of_studies_numeric": level,
                        "lemma_frequency_normalized": float(wi),
                        # eye-tracking measures (FFD<=SFD<=FPRT<=TFT typically);
                        # keep them positive & varying so means/PCA are defined.
                        "FFD": 0 if skip else int(base * 0.4),
                        "SFD": 0 if skip else int(base * 0.5),
                        "FPRT": 0 if skip else int(base * 0.7),
                        "TFT": tft,
                        "TFC": 0 if skip else int(1 + wi),
                        "RPD_inc": 0 if skip else int(base * 0.9),
                        "Fix": 0 if skip else 1,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def words_df() -> pd.DataFrame:
    """Per-word sentence frame for surprisal / attention extraction."""
    rows = []
    for text_id, domain, _ in TEXTS:
        for wi in range(N_WORDS):
            rows.append(
                {
                    "text_id": text_id,
                    "text_domain": domain,
                    "sent_index_in_text": 0,
                    "word_index_in_text": wi,
                    "word_index_in_sent": wi,
                    "word": f"{text_id}w{wi}",
                    "is_sent_beginning": int(wi == 0),
                    "is_sent_end": int(wi == N_WORDS - 1),
                }
            )
    return pd.DataFrame(rows)


# ── fake decoder LM + tokenizer ──────────────────────────────────────────────
class FakeEnc:
    """Mimics a tokenizer BatchEncoding: `.input_ids` + `.word_ids(i)`."""

    def __init__(self, input_ids, word_ids):
        self.input_ids = input_ids
        self._word_ids = word_ids

    def word_ids(self, i=0):
        return self._word_ids


class FakeTokenizer:
    """Deterministic tokenizer. A word splits into subtokens on '_'.

    `is_split_into_words=True` -> input is a word list (one entry per word).
    Otherwise the input string is whitespace-split (used for the prompt path).
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self._vocab.setdefault(tok, len(self._vocab) + 1)

    def __call__(self, text, is_split_into_words=False, return_tensors=None):
        words = text if is_split_into_words else str(text).split()
        ids, wids = [], []
        for wi, w in enumerate(words):
            for sub in str(w).split("_"):
                ids.append(self._id(sub))
                wids.append(wi)
        return FakeEnc(torch.tensor([ids]), wids)


class _Out:
    def __init__(self, logits=None, attentions=None):
        self.logits = logits
        self.attentions = attentions


class FakeCausalLM(nn.Module):
    """Outputs uniform next-token logits, so every token's surprisal == log2(vocab)."""

    def __init__(self, vocab: int = 64):
        super().__init__()
        self.vocab = vocab
        self._p = nn.Parameter(torch.zeros(1))  # gives .parameters()/.device

    def forward(self, input_ids, output_attentions=False):
        seq = input_ids.shape[1]
        logits = torch.zeros(1, seq, self.vocab)
        return _Out(logits=logits)


class FakeAttnLM(nn.Module):
    """Returns deterministic causal (lower-triangular, row-normalized) attentions."""

    def __init__(self, n_layers: int = 3, n_heads: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, output_attentions=True):
        seq = input_ids.shape[1]
        causal = torch.tril(torch.ones(seq, seq))
        causal = causal / causal.sum(-1, keepdim=True)
        att = causal.expand(1, self.n_heads, seq, seq).contiguous()
        return _Out(attentions=tuple(att.clone() for _ in range(self.n_layers)))


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_causal_lm():
    return FakeCausalLM()


@pytest.fixture
def fake_attn_lm():
    return FakeAttnLM()


# ── larger frame for the mixed-effects model comparison (steps 5/stats) ───────
@pytest.fixture
def mc_inputs():
    """Synthetic inputs for `model_comparison` / `stats`.

    Returns dict with `surp_versions`, `rt_df`, `prompt_surp`. Reading time is
    built to depend on the word's intrinsic (base) surprisal so the mixed models
    are well-conditioned; we don't assert which surprisal source wins (research),
    only that the machinery runs and returns sane statistics.
    """
    rng = np.random.default_rng(7)
    n_words = 20
    # words live in two texts (one per domain); both fine-tuned models score all.
    words = [("P", "physics", wi) for wi in range(1, n_words + 1)]
    words += [("B", "biology", wi) for wi in range(1, n_words + 1)]

    base = {w: rng.uniform(2, 8) for w in words}
    phys = {w: base[w] + rng.normal(0, 0.5) for w in words}
    bio = {w: base[w] + rng.normal(0, 0.5) for w in words}
    pphys = {w: base[w] + rng.normal(0, 0.5) for w in words}
    pbio = {w: base[w] + rng.normal(0, 0.5) for w in words}
    wlen = {w: int(rng.integers(3, 11)) for w in words}
    freq = {w: float(rng.uniform(0, 50)) for w in words}

    def surp_rows(index, domain, table):
        return [
            {
                "index": index,
                "epoch": float(index),
                "domain": domain,
                "text_id": tid,
                "word_index_in_text": wi,
                "surprisal": table[(tid, dom, wi)],
            }
            for (tid, dom, wi) in words
        ]

    sv = []
    # index 0 = un-adapted baseline (both domain rows carry base surprisal).
    sv += surp_rows(0, "physics", base)
    sv += surp_rows(0, "biology", base)
    # index 1 = adapted checkpoint.
    sv += surp_rows(1, "physics", phys)
    sv += surp_rows(1, "biology", bio)
    surp_versions = pd.DataFrame(sv)

    prompt_surp = pd.DataFrame(
        [
            {
                "text_id": tid,
                "word_index_in_text": wi,
                "s_prompt_phys": pphys[(tid, dom, wi)],
                "s_prompt_bio": pbio[(tid, dom, wi)],
            }
            for (tid, dom, wi) in words
        ]
    )

    # reading measures: 6 physics + 6 biology readers over every word.
    readers = [(f"rp{i}", 1) for i in range(6)] + [(f"rb{i}", 0) for i in range(6)]
    rt_rows = []
    for (tid, dom, wi) in words:
        dom_num = 1 if dom == "physics" else 0
        for reader_id, disc in readers:
            tft = (
                120
                + 25 * base[(tid, dom, wi)]
                + (10 if disc == dom_num else 0)
                + rng.normal(0, 5)
            )
            rt_rows.append(
                {
                    "text_id": tid,
                    "text_domain": dom,
                    "text_domain_numeric": dom_num,
                    "word_index_in_text": wi,
                    "word_length": wlen[(tid, dom, wi)],
                    "lemma_frequency_normalized": freq[(tid, dom, wi)],
                    "is_expert_technical_term": int(wi % 5 == 0),
                    "is_general_technical_term": int(wi % 7 == 0),
                    "reader_id": reader_id,
                    "reader_discipline_numeric": disc,
                    "TFT": float(max(tft, 1.0)),
                }
            )
    rt_df = pd.DataFrame(rt_rows)
    rt_df["is_expert"] = (
        rt_df["reader_discipline_numeric"] == rt_df["text_domain_numeric"]
    ).astype(int)
    return {"surp_versions": surp_versions, "rt_df": rt_df, "prompt_surp": prompt_surp}
