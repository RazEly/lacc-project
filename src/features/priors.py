"""Prior-reading passages for the prompted-surprisal arms.

The prompted arm conditions a base LM on a short passage the reader "just read",
joined to the stimulus by a native document boundary (surprisal.score_words).
Domain priors come from each domain's Wikipedia corpus"""

from __future__ import annotations

import re

from datasets import load_from_disk

from src.config import DOMAIN_DIRS, N_PRIOR_PASSAGES

PRIOR_PASSAGE_SENTENCES = 20


def _first_sentences(text: str, n: int) -> str:
    """First n sentences of text"""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def _domain_passages(domain: str, seed: int, n_sent: int, k: int) -> list[str]:
    """k distinct domain priors - opening sentences of k shuffled docs."""
    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    docs = raw.shuffle(seed=seed).select(range(k))
    return [_first_sentences(str(doc["text"]), n_sent) for doc in docs]


def load_prior_passages(
    seed: int = 0,
    n_sent: int = PRIOR_PASSAGE_SENTENCES,
    k: int = N_PRIOR_PASSAGES,
) -> dict[str, list[str]]:
    """{"physics", "biology"} -> list of k prior passages."""
    return {domain: _domain_passages(domain, seed, n_sent, k) for domain in DOMAIN_DIRS}
