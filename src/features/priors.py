"""Prior-reading passages for the prompted-surprisal arm.

The prompted arm conditions a base LM on a short passage the reader "just read",
joined to the stimulus by a native document boundary (surprisal.score_words).
Domain priors come from the HELD-OUT german-commons split — the same corpus DAPT
trains on, so "prime via context" and "adapt via weights" draw domain signal from
one source — but from a document the model never trained on, and never the PoTeC
stimulus (which would trigger in-context memorisation and collapse surprisal).

The neutral condition is sampled from the SAME corpus by the SAME code path: the
off-domain pool (``config.DOMAIN_OTHER_DIR``, built by acquire.domain_preprocessing
from docs far from both domain seeds). Scientific register, no physics/biology
content — so the domain-vs-neutral contrast isolates domain content, not register
or sampling procedure. DAPT never trains on the off-domain pool, so every neutral
passage is unseen by every checkpoint; the split is applied anyway to keep the
code path identical across conditions.

Each condition supplies ``config.N_PRIOR_PASSAGES`` DISTINCT priors, not one.
Downstream the stimulus is scored under each prior separately and the per-word
surprisals are averaged (surprisal.compute_surprisal), so no single idiosyncratic
passage drives a condition.
"""

from __future__ import annotations

import re

from datasets import load_from_disk

from src.config import (
    DOMAIN_BIO_DIR,
    DOMAIN_OTHER_DIR,
    DOMAIN_PHY_DIR,
    N_PRIOR_PASSAGES,
)

# How many leading sentences of a held-out domain doc make one prior passage
# (token-truncated to the caller's budget at scoring time).
PRIOR_PASSAGE_SENTENCES = 4

_DOMAIN_DIRS = {
    "physics": DOMAIN_PHY_DIR,
    "biology": DOMAIN_BIO_DIR,
    "neutral": DOMAIN_OTHER_DIR,
}


def _first_sentences(text: str, n: int) -> str:
    """First ``n`` sentences of ``text`` (rough split on sentence-final punct)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def _held_out_passages(
    domain: str, val_frac: float, seed: int, n_sent: int, k: int
) -> list[str]:
    """``k`` distinct domain priors from the held-out split — reproduces DAPT's val.

    Same ``train_test_split(test_size=val_frac, seed=seed)`` as
    ``finetune._prepare_splits``, so the passages are genuinely absent from DAPT
    training. Shuffles the test split by ``seed`` (reproducible) and takes the
    opening sentences of the first ``k`` docs — a spread across the held-out set,
    not ``k`` neighbours, so the average samples the domain broadly. Raises if the
    split holds fewer than ``k`` docs (widen ``val_frac`` or lower ``k``).
    """
    raw = load_from_disk(str(_DOMAIN_DIRS[domain]))
    test = raw.train_test_split(test_size=val_frac, seed=seed)["test"]
    if len(test) < k:
        raise ValueError(
            f"{domain}: held-out split has {len(test)} docs < k={k} priors"
        )
    test = test.shuffle(seed=seed).select(range(k))
    return [_first_sentences(str(doc["text"]), n_sent) for doc in test]


def load_prior_passages(
    val_frac: float = 0.05,
    seed: int = 0,
    n_sent: int = PRIOR_PASSAGE_SENTENCES,
    k: int = N_PRIOR_PASSAGES,
) -> dict[str, list[str]]:
    """``{"physics", "biology", "neutral"}`` -> list of ``k`` prior passages.

    All three lists are ``k`` distinct german-commons openings drawn by the same
    procedure (defaults match DAPT's split); "neutral" draws from the off-domain
    pool. Every passage is token-truncated to the caller's budget at scoring
    time; the caller averages surprisal across each list.
    """
    return {
        domain: _held_out_passages(domain, val_frac, seed, n_sent, k)
        for domain in _DOMAIN_DIRS
    }
