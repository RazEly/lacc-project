"""Fig 6 — corpus↔stimulus lemma-vocabulary overlap, before vs after the domain filter.

Replicates the similarity measure of Škrjanec & Demberg (2026), ``03-reader-experience``
(their Fig 2): directional lemma-vocabulary overlap in percent. For a corpus and the
PoTeC stimulus text of a domain,

    overlap% = |V_corpus ∩ V_stimulus| / |V_corpus| × 100,

i.e. the proportion of the corpus's vocabulary that is stimulus vocabulary. Every
vocabulary is the set of spaCy lemmas with nltk German stopwords, numbers and
punctuation removed — the paper's exact preprocessing.

Two phases per domain (physics / biology):
  • pre-filter  — a random, domain-agnostic sample of the full german-commons corpus,
    SIZE-MATCHED to the filtered set (so the overlap gap reflects domain concentration,
    not corpus size — the measure's denominator grows with doc count);
  • post-filter — the selected DAPT corpus (``data/domain_<domain>``).

A pre→post rise shows the two-pass filter concentrating the corpus vocabulary onto
stimulus terms. Writes ``figures/fig6_stimuli_similarity.png``.

    python scripts/stimuli_similarity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

import nltk
import pandas as pd
import spacy
from datasets import load_from_disk

from src.analysis import viz
from src.config import DOMAINS, DOMAIN_DIRS
from src.features import potec
from src.acquire.domain_preprocessing import load_commons

SPACY_MODEL = "de_core_news_sm"
# Per-doc char cap: a doc's vocabulary saturates early, so capping bounds the spaCy
# cost without materially changing the lemma set. Applied equally to both phases.
MAX_DOC_CHARS = 20_000
SEED = 0


def _german_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords
        return set(w.lower() for w in stopwords.words("german"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(w.lower() for w in stopwords.words("german"))


def _load_nlp():
    """German spaCy pipeline, parser/NER disabled — only lemmatisation is needed."""
    return spacy.load(SPACY_MODEL, disable=["parser", "ner"])


def _vocab(texts, nlp, stops: set[str]) -> set[str]:
    """Lemma vocabulary of a text collection: alphabetic lemmas, no stopwords.

    Drops punctuation, spaces and numbers (non-alphabetic tokens) and the nltk
    German stopword list, matching the paper's preprocessing.
    """
    clipped = [(t or "")[:MAX_DOC_CHARS] for t in texts]
    vocab: set[str] = set()
    for doc in nlp.pipe(clipped, batch_size=64):
        for tok in doc:
            if not tok.is_alpha:
                continue
            lemma = tok.lemma_.lower()
            if lemma and lemma not in stops:
                vocab.add(lemma)
    return vocab


def _overlap_pct(v_corpus: set[str], v_stim: set[str]) -> float:
    """Paper's directional overlap: share of the corpus vocab that is stimulus vocab."""
    return 100.0 * len(v_corpus & v_stim) / len(v_corpus) if v_corpus else 0.0


def stimuli_vocab_by_domain(nlp, stops) -> dict[str, set[str]]:
    """Lemma vocabulary of the PoTeC stimulus text, per domain."""
    words = potec.load_word_features()
    out = {}
    for dom, g in words.groupby("text_domain"):
        text = " ".join(
            g.sort_values("word_index_in_text")["word"].fillna("").astype(str)
        )
        out[dom] = _vocab([text], nlp, stops)
    return out


def stimuli_overlap_frame() -> pd.DataFrame:
    """Tidy (domain, phase, overlap_pct) frame feeding ``viz.stimuli_similarity_grid``."""
    nlp = _load_nlp()
    stops = _german_stopwords()
    stim_vocab = stimuli_vocab_by_domain(nlp, stops)

    full = load_commons()
    rows = []
    for domain in DOMAINS:
        post = load_from_disk(str(DOMAIN_DIRS[domain]))
        n = len(post)
        # size-match the pre-filter sample to the filtered set (fair denominator).
        pre = full.shuffle(seed=SEED).select(range(min(n, len(full))))

        v_stim = stim_vocab[domain]
        for phase, ds in (("pre-filter", pre), ("post-filter", post)):
            v_corpus = _vocab(ds["text"], nlp, stops)
            pct = _overlap_pct(v_corpus, v_stim)
            print(
                f"{domain}/{phase}: docs={len(ds):,} |V_corpus|={len(v_corpus):,} "
                f"|V_stim|={len(v_stim):,} overlap={pct:.2f}%"
            )
            rows.append({"domain": domain, "phase": phase, "overlap_pct": pct})
    return pd.DataFrame(rows)


def main() -> None:
    overlaps = stimuli_overlap_frame()
    viz.stimuli_similarity_grid(overlaps)
    print(f"\nwrote {viz.FIGURES_DIR / 'fig6_stimuli_similarity.png'}")


if __name__ == "__main__":
    main()
