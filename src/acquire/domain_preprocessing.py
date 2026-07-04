"""Label the german-commons scientific corpus by domain (physics / biology).

Two-stage labelling — NO learned classifier, following Škrjanec & Demberg (2026,
JML): domain membership is decided by lexical overlap with the PoTeC stimulus
vocabulary, not by a zero-shot NLI model (which was uncalibrated and noisy on the
12 gold texts).

  stage 0 — QUALITY / REGISTER gate. DAPT must adapt to CLEAN MODERN expository
            German (the PoTeC stimulus register), NOT archaic OCR. Drop pre-modern
            orthography sources, OCR-degraded scans, low-fluency docs, and
            length-outlier docs that would dominate the token budget. Applied to
            the WHOLE corpus first, so physics / biology / neutral are all gated
            identically.
  stage 1 — POTEC-VOCABULARY OVERLAP. Lemmatise every doc (spaCy ``de_core_news``)
            and the PoTeC stimuli, drop nltk German stopwords, and score each doc
            by the fraction of its content lemmas that fall in each domain's PoTeC
            vocabulary (``coverage``). A doc takes the argmax domain when that
            coverage clears a floor AND beats the other domain by a margin; else
            it is "other". Both thresholds are calibrated leave-one-text-out on the
            12 gold PoTeC texts, so no PoTeC text scores against its own words.

The NEUTRAL pool (``DOMAIN_OTHER_DIR``) is the "other" docs — same corpus and
register as the domain priors, no physics/biology content — the length-matched
control for the prompt arm (features.priors).

Run from the project root:
    python -m src.acquire.domain_preprocessing
Set ``DOMAIN_REBUILD=1`` to ignore the cached label ids AND overwrite the
existing domain dirs (needed after changing the labelling recipe).
``SPACY_NPROCESS`` (default 4) sets the lemmatiser worker count.
"""

import glob
import json
import os
import shutil

import numpy as np
from datasets import concatenate_datasets, load_from_disk

from src.config import (
    ARTIFACTS_DIR,
    COMMONS_DIR,
    DOMAIN_DIRS,
    DOMAIN_OTHER_DIR,
    POTEC_DIR,
)
from src.features.potec import read_potec

# When this exists, labelling is skipped and datasets are rebuilt from it —
# UNLESS DOMAIN_REBUILD=1, which forces a full relabel + overwrite.
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"
REBUILD = os.environ.get("DOMAIN_REBUILD", "0") == "1"

# Chars per doc fed to the lemmatiser for the coverage estimate. Coverage is a
# ratio, stable on a leading sample, so this bounds spaCy cost without biasing.
LABEL_CHARS = 4000
SPACY_MODEL = "de_core_news_md"
SPACY_NPROCESS = int(os.environ.get("SPACY_NPROCESS", "4"))
SPACY_BATCH = 64
MIN_LEMMA_LEN = 3  # drop 1-2 char tokens (units, stray letters)

# ── stage 0: document quality / register gate ────────────────────────────────
# german-commons is dominated (by TOKENS) by 19th-century Fraktur-OCR journals
# whose orthography ("dafs", "fiber", "grofs") and OCR noise are nothing like the
# clean modern PoTeC stimuli — so DAPT on them lowers in-corpus perplexity
# without transferring to PoTeC surprisal. Gate them out.
EXCLUDE_SOURCES = {"Polytechnisches Journal"}  # pre-modern orthography, OCR
MIN_OCR_SCORE = 90.0  # OpenAlex OCR confidence; None (born-digital) kept
MAX_DOC_PERPLEXITY = 1000.0  # GPT-2 doc ppl; drops OCR-garble / non-prose
MIN_DOC_TOKENS = 128  # drop stubs / redirect pages
MAX_DOC_TOKENS = 40_000  # cap so no single huge doc dominates the token budget

# ── neutral (off-domain) pool ────────────────────────────────────────────────
# The neutral pool is the stage-1 "other" set: docs whose PoTeC-vocabulary
# coverage puts them in neither domain. Same corpus and register as the domain
# priors, no physics/biology content — the length-matched control for the prompt
# arm. Fluency gate: drop the worst-OCR / least-fluent docs by perplexity
# percentile, so the neutral priors read as cleanly as the domain priors.
NEUTRAL_MAX_PPL_PCTL = 60
N_OTHER_DOCS = 1000  # broad sample to draw K priors from, not the whole mass
OTHER_SEED = 0

DOMAINS = ("physics", "biology")


def load_commons():
    """Load the saved german-commons scientific corpus as a single Dataset.

    Raw load only; the quality/register gate is applied separately
    (``apply_quality_filter``) so every downstream pool shares it.
    """
    ds = load_from_disk(str(COMMONS_DIR))
    if not hasattr(ds, "column_names") or isinstance(ds.column_names, dict):
        ds = concatenate_datasets([ds[name] for name in ds.keys()])
    return ds


def apply_quality_filter(ds):
    """Drop archaic-OCR / low-fluency / length-outlier docs (stage 0).

    Keeps a doc when: its source is not pre-modern, its OCR score (when present)
    clears ``MIN_OCR_SCORE``, its GPT-2 perplexity (when present) is under
    ``MAX_DOC_PERPLEXITY``, and its token count sits in
    ``[MIN_DOC_TOKENS, MAX_DOC_TOKENS]``. Missing ocr/ppl (born-digital sources
    like Wikibooks) are treated as clean.
    """

    def _keep(src, ocr, ppl, ntok):
        ntok = ntok or 0
        return (
            src not in EXCLUDE_SOURCES
            and (ocr is None or float(ocr) >= MIN_OCR_SCORE)
            and (ppl is None or float(ppl) <= MAX_DOC_PERPLEXITY)
            and MIN_DOC_TOKENS <= ntok <= MAX_DOC_TOKENS
        )

    idx = [
        i
        for i, (src, ocr, ppl, ntok) in enumerate(
            zip(ds["source"], ds["ocr_score"], ds["perplexity"], ds["num_tokens"])
        )
        if _keep(src, ocr, ppl, ntok)
    ]
    kept = ds.select(idx)
    print(
        f"\nStage 0 — quality gate: {len(kept):,}/{len(ds):,} docs kept "
        f"({len(ds) - len(kept):,} dropped: archaic/OCR/ppl/length)."
    )
    return kept


# ── lemmatisation (spaCy) + stopwords (nltk), lazily loaded ───────────────────
def _load_spacy():
    """Load the German spaCy pipeline (lemmatiser only), auto-downloading once."""
    import spacy

    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except OSError:
        from spacy.cli import download

        print(f"  spaCy model {SPACY_MODEL} missing — downloading (one-time)...")
        download(SPACY_MODEL)
        return spacy.load(SPACY_MODEL, disable=["parser", "ner"])


def _german_stopwords():
    """nltk German stopword set, auto-downloading the corpus once."""
    import nltk

    try:
        from nltk.corpus import stopwords

        words = stopwords.words("german")
    except LookupError:
        nltk.download("stopwords")
        from nltk.corpus import stopwords

        words = stopwords.words("german")
    return {w.lower() for w in words}


def _lemmatise(texts, nlp, stop, n_process=1):
    """Map each text to its set of content lemmas.

    Keeps alphabetic, non-stopword lemmas of length >= ``MIN_LEMMA_LEN``,
    lowercased — mirrors the paper's lemmatised, stopword-removed vocabulary.
    """
    out = []
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=SPACY_BATCH):
        out.append(
            {
                t.lemma_.lower()
                for t in doc
                if t.is_alpha
                and not t.is_stop
                and len(t.lemma_) >= MIN_LEMMA_LEN
                and t.lemma_.lower() not in stop
            }
        )
    return out


def _potec_domain_lemmas(potec_dir, nlp, stop):
    """Per-PoTeC-text ``text_id -> (domain, lemma_set)`` from the stimuli.

    One lemma set per gold text lets calibration leave a text out (so it never
    scores against its own words), while the union over a domain is the labelling
    vocabulary.
    """
    per_text = {}
    for f in sorted(
        glob.glob(os.path.join(potec_dir, "stimuli/word_features/word_features_*.tsv"))
    ):
        df = read_potec(f)
        tid = os.path.basename(f).replace("word_features_", "").replace(".tsv", "")
        text = " ".join(df["word"].fillna("").astype(str).tolist())
        per_text[tid] = (df["text_domain"].iloc[0], _lemmatise([text], nlp, stop)[0])
    return per_text


def _union_vocab(per_text, domain, exclude_tid=None):
    """Union of lemma sets for ``domain``, optionally excluding one text (LOO)."""
    vocab = set()
    for tid, (dom, lemmas) in per_text.items():
        if dom == domain and tid != exclude_tid:
            vocab |= lemmas
    return vocab


def _coverage(doc_lemmas, vocab):
    """Fraction of a doc's content lemmas that fall in ``vocab`` (0..1)."""
    if not doc_lemmas:
        return 0.0
    return len(doc_lemmas & vocab) / len(doc_lemmas)


def calibrate_overlap(per_text):
    """Coverage floor + margin, calibrated leave-one-text-out on the gold texts.

    For each gold text the domain vocabularies are built from the OTHER texts, so
    a text never scores against its own words. Positives are the in-domain
    coverage; the floor keeps most true in-domain docs and the margin is how much
    the correct domain must beat the other.
    """
    print("\nStage 1 calibration — PoTeC coverage (leave-one-text-out):")
    in_cov, off_cov, margins = [], [], []
    for tid, (dom, lemmas) in per_text.items():
        cov = {
            d: _coverage(lemmas, _union_vocab(per_text, d, exclude_tid=tid))
            for d in DOMAINS
        }
        indom = cov[dom]
        off = max(cov[d] for d in DOMAINS if d != dom)
        in_cov.append(indom)
        off_cov.append(off)
        margins.append(indom - off)
        print(
            f"  [{tid}] gold={dom} in={indom:.3f} off={off:.3f} margin={indom - off:+.3f}"
        )

    # floor: half the weakest true in-domain coverage (keep recall high while
    # still rejecting near-zero-overlap off-domain docs).
    cov_floor = 0.5 * min(in_cov) if in_cov else 0.03
    # margin: half the smallest correct separation; 0 if any gold text misorders.
    margin_floor = max(0.0, min(margins)) * 0.5
    print(
        f"\n  in-domain coverage : min={min(in_cov):.3f} max={max(in_cov):.3f}\n"
        f"  off-domain coverage: min={min(off_cov):.3f} max={max(off_cov):.3f}\n"
        f"  → coverage floor={cov_floor:.3f}, margin floor={margin_floor:.3f}"
    )
    return cov_floor, margin_floor


def label_by_overlap(texts, per_text, nlp, stop, cov_floor, margin_floor):
    """Assign every doc a domain by PoTeC-vocabulary coverage (argmax + gates).

    Returns ``(labels, coverages)`` over ALL docs: the argmax domain when its
    coverage clears ``cov_floor`` and beats the other domain by ``margin_floor``,
    else "other".
    """
    vocab = {d: _union_vocab(per_text, d) for d in DOMAINS}
    print(
        f"\nStage 1 — lemmatising {len(texts):,} docs (n_process={SPACY_NPROCESS})..."
    )
    doc_lemmas = _lemmatise(texts, nlp, stop, n_process=SPACY_NPROCESS)

    labels = ["other"] * len(texts)
    scores = [0.0] * len(texts)
    for i, lemmas in enumerate(doc_lemmas):
        cov = {d: _coverage(lemmas, vocab[d]) for d in DOMAINS}
        best = max(DOMAINS, key=lambda d: cov[d])
        other = max(cov[d] for d in DOMAINS if d != best)
        if cov[best] >= cov_floor and (cov[best] - other) >= margin_floor:
            labels[i] = best
            scores[i] = cov[best]
    print(
        f"  physics: {labels.count('physics'):,}  "
        f"biology: {labels.count('biology'):,}  other: {labels.count('other'):,}"
    )
    return labels, scores


def select_other(other_ds):
    """Neutral pool from the stage-1 "other" docs: fluency gate + sample.

    A per-pool perplexity gate drops the worst-OCR docs, then a shuffled sample of
    up to ``N_OTHER_DOCS`` is returned.
    """
    ppl = np.array(
        [p if p is not None else np.nan for p in other_ds["perplexity"]], dtype=float
    )
    finite = ppl[np.isfinite(ppl)]
    if len(finite):
        cutoff = np.percentile(finite, NEUTRAL_MAX_PPL_PCTL)
        keep = [i for i, p in enumerate(ppl) if not np.isfinite(p) or p <= cutoff]
        other_ds = other_ds.select(keep)
    other_ds = other_ds.shuffle(seed=OTHER_SEED)
    return other_ds.select(range(min(N_OTHER_DOCS, len(other_ds))))


# ── label-id cache ───────────────────────────────────────────────────────────
def save_label_ids(physics_ds, biology_ds, other_ds, path=LABEL_IDS_PATH):
    """Persist the selected physics/biology/other document ids as JSON for reuse."""
    ids = {
        "physics": list(physics_ds["id"]),
        "biology": list(biology_ds["id"]),
        "other": list(other_ds["id"]),
    }
    with open(path, "w") as f:
        json.dump(ids, f)
    print(
        f"  wrote label ids: {path} "
        f"({len(ids['physics']):,} physics, {len(ids['biology']):,} biology, "
        f"{len(ids['other']):,} other)"
    )


def build_from_label_ids(path=LABEL_IDS_PATH):
    """Rebuild the physics/biology/other datasets from a saved id file.

    Returns ``(physics_ds, biology_ds, other_ds)`` or ``None`` if no cache file
    exists OR ``DOMAIN_REBUILD=1`` (which forces a fresh relabel). The cached ids
    already encode the quality gate, so no re-filtering here.
    """
    if REBUILD or not path.exists():
        return None
    with open(path) as f:
        ids = json.load(f)
    print(f"\nFound cached label ids ({path}); skipping quality gate + overlap.")
    ds = load_commons()
    physics = set(ids["physics"])
    biology = set(ids["biology"])
    other = set(ids["other"])
    physics_ds = ds.filter(lambda x: x["id"] in physics)
    biology_ds = ds.filter(lambda x: x["id"] in biology)
    other_ds = ds.filter(lambda x: x["id"] in other)
    print(
        f"  physics : {len(physics_ds):,}   biology : {len(biology_ds):,}   "
        f"other : {len(other_ds):,}"
    )
    return physics_ds, biology_ds, other_ds


def main():
    # Reuse a previous run's labels if they were saved (unless DOMAIN_REBUILD=1).
    cached = build_from_label_ids()
    if cached is not None:
        _save_domain_datasets(*cached)
        return

    ds = apply_quality_filter(load_commons())
    texts = [t[:LABEL_CHARS] if t else "" for t in ds["text"]]

    nlp = _load_spacy()
    stop = _german_stopwords()
    per_text = _potec_domain_lemmas(str(POTEC_DIR), nlp, stop)
    cov_floor, margin_floor = calibrate_overlap(per_text)
    labels, scores = label_by_overlap(
        texts, per_text, nlp, stop, cov_floor, margin_floor
    )

    ds = ds.add_column("domain_label", labels).add_column("domain_coverage", scores)
    physics_ds = ds.filter(lambda x: x["domain_label"] == "physics")
    biology_ds = ds.filter(lambda x: x["domain_label"] == "biology")
    other_ds = select_other(ds.filter(lambda x: x["domain_label"] == "other"))

    print("\nFinal results (quality ∩ PoTeC-overlap):")
    print(
        f"physics : {len(physics_ds):,}  ({sum(r['num_tokens'] for r in physics_ds):,} tokens)"
    )
    print(
        f"biology : {len(biology_ds):,}  ({sum(r['num_tokens'] for r in biology_ds):,} tokens)"
    )
    print(f"neutral : {len(other_ds):,}  (sampled from 'other')")

    print("\n-- top physics docs --")
    for row in sorted(physics_ds, key=lambda x: x["domain_coverage"], reverse=True)[:3]:
        print(
            f"  cov={row['domain_coverage']:.3f} [{row['source']}] {row['text'][:100]!r}"
        )
    print("\n-- top biology docs --")
    for row in sorted(biology_ds, key=lambda x: x["domain_coverage"], reverse=True)[:3]:
        print(
            f"  cov={row['domain_coverage']:.3f} [{row['source']}] {row['text'][:100]!r}"
        )

    save_label_ids(physics_ds, biology_ds, other_ds)
    _save_domain_datasets(physics_ds, biology_ds, other_ds)


def _save_domain_datasets(physics_ds, biology_ds, other_ds):
    """Write the physics/biology/other datasets to disk.

    An EXISTING domain dir is normally left untouched — the DAPT checkpoints were
    trained on its exact contents and priors.py samples its held-out split, so a
    silent overwrite would desync training from the priors. ``DOMAIN_REBUILD=1``
    opts into overwriting (needed after changing the labelling recipe); it is the
    caller's job to then retrain DAPT on the new corpus. The neutral dir is always
    (re)written.
    """
    print("\nSaving datasets...")
    for ds, path in [
        (physics_ds, DOMAIN_DIRS["physics"]),
        (biology_ds, DOMAIN_DIRS["biology"]),
    ]:
        if path.exists() and not REBUILD:
            print(f"  skip {path} (exists — keeping the DAPT-trained version)")
            continue
        if path.exists():
            shutil.rmtree(path)
            print(f"  DOMAIN_REBUILD=1 → overwriting {path}")
        ds.save_to_disk(str(path))
        print(f"  saved {path}")
    if DOMAIN_OTHER_DIR.exists():
        shutil.rmtree(DOMAIN_OTHER_DIR)
    other_ds.save_to_disk(str(DOMAIN_OTHER_DIR))
    print(f"  saved {DOMAIN_OTHER_DIR}")


if __name__ == "__main__":
    main()
