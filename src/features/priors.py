"""Prior-reading passages for the prompted-surprisal arm.

The prompted arm conditions a base LM on a short passage the reader "just read",
joined to the stimulus by a native document boundary (surprisal.score_words).
Domain priors come from the HELD-OUT german-commons split — the same corpus DAPT
trains on, so "prime via context" and "adapt via weights" draw domain signal from
one source — but from a document the model never trained on, and never the PoTeC
stimulus (which would trigger in-context memorisation and collapse surprisal).

The neutral prior is a fixed non-domain German passage (config.NEUTRAL_PRIOR),
used as a length-matched control that isolates domain content from the mere
presence of any prior.
"""

from __future__ import annotations

import re

from datasets import load_from_disk

from src.config import (
    DOMAIN_BIO_DIR,
    DOMAIN_PHY_DIR,
    NEUTRAL_PRIOR,
    PRIOR_PASSAGE_SENTENCES,
)

_DOMAIN_DIRS = {"physics": DOMAIN_PHY_DIR, "biology": DOMAIN_BIO_DIR}


def _first_sentences(text: str, n: int) -> str:
    """First ``n`` sentences of ``text`` (rough split on sentence-final punct)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def _held_out_passage(domain: str, val_frac: float, seed: int, n_sent: int) -> str:
    """A domain prior from the held-out split — reproduces DAPT's val partition.

    Same ``train_test_split(test_size=val_frac, seed=seed)`` as
    ``finetune._prepare_splits``, so the passage is genuinely absent from DAPT
    training. Takes the first test document's opening sentences.
    """
    raw = load_from_disk(str(_DOMAIN_DIRS[domain]))
    test = raw.train_test_split(test_size=val_frac, seed=seed)["test"]
    return _first_sentences(str(test[0]["text"]), n_sent)


def load_prior_passages(
    val_frac: float = 0.05,
    seed: int = 0,
    n_sent: int = PRIOR_PASSAGE_SENTENCES,
) -> dict[str, str]:
    """``{"physics", "biology", "neutral"}`` prior passages.

    Domain priors are held-out german-commons openings (defaults match DAPT's
    split); the neutral prior is the fixed ``config.NEUTRAL_PRIOR``. All are
    token-truncated to ``config.PRIOR_MAX_TOKENS`` at scoring time.
    """
    return {
        "physics": _held_out_passage("physics", val_frac, seed, n_sent),
        "biology": _held_out_passage("biology", val_frac, seed, n_sent),
        "neutral": NEUTRAL_PRIOR,
    }
