"""Prior-reading passages for the prompted-surprisal arms.

The prompted arm conditions a base LM on a short passage the reader "just read",
joined to the stimulus by a native document boundary (surprisal.score_words).
Physics/biology priors (each domain's Wikipedia corpus) feed the reader-aligned
``prompted`` arm; the neutral prior (off-domain corpus) is the single
length-matched pseudo-test replacing the two fixed-domain ones."""

from __future__ import annotations

import re

from datasets import load_from_disk

from src.config import DOMAIN_DIRS, N_PRIOR_PASSAGES, NEUTRAL_DIR

PRIOR_PASSAGE_SENTENCES = 20

# Every prior source -> its corpus dir: the two domains (aligned arm) + neutral.
PRIOR_DIRS = {**DOMAIN_DIRS, "neutral": NEUTRAL_DIR}


def _first_sentences(text: str, n: int) -> str:
    """First n sentences of text"""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def _corpus_passages(corpus_dir, seed: int, n_sent: int, k: int) -> list[str]:
    """k distinct priors - opening sentences of k shuffled docs from a corpus."""
    raw = load_from_disk(str(corpus_dir))
    docs = raw.shuffle(seed=seed).select(range(k))
    return [_first_sentences(str(doc["text"]), n_sent) for doc in docs]


def load_prior_passages(
    seed: int = 0,
    n_sent: int = PRIOR_PASSAGE_SENTENCES,
    k: int = N_PRIOR_PASSAGES,
) -> dict[str, list[str]]:
    """{"physics", "biology", "neutral"} -> list of k prior passages."""
    return {
        source: _corpus_passages(d, seed, n_sent, k) for source, d in PRIOR_DIRS.items()
    }
