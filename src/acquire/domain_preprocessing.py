"""Label the german-commons scientific corpus by domain (physics / biology).

Three-stage weak labelling:
  stage 0 — QUALITY / REGISTER gate. DAPT must adapt to CLEAN MODERN expository
            German (the PoTeC stimulus register), NOT archaic OCR. Drop pre-modern
            orthography sources, OCR-degraded scans, low-fluency docs, and
            length-outlier docs that would dominate the token budget. Applied to
            the WHOLE corpus first, so physics / biology / neutral are all gated
            identically.
  pass 1 — TF-IDF char n-gram similarity against physics/biology keyword bags
            (cheap high-recall candidate prefilter, not the final decision).
  pass 2 — zero-shot NLI (German-native GBERT-large), MULTI-LABEL so each domain hypothesis
            gets an INDEPENDENT entailment probability. A candidate is kept only
            when its best domain entailment clears an absolute threshold, so
            off-domain docs (chemistry / medicine / humanities) that pass-1 let
            through are rejected. The threshold is calibrated for domain
            MEMBERSHIP on the 12 gold PoTeC texts (in-domain probs vs
            cross-domain probs), not for the physics-vs-biology tie-break.

Also carves out a NEUTRAL pool (``DOMAIN_OTHER_DIR``): the docs that fall OUTSIDE
the pass-1 candidate set — both domain TF-IDF similarities below the candidate
threshold. Same corpus and register as the domain priors, no physics/biology
content. Feeds the neutral prompt condition (features.priors).

Run from the project root:
    python -m src.acquire.domain_preprocessing
Set ``DOMAIN_REBUILD=1`` to ignore the cached label ids AND overwrite the
existing domain dirs (needed after changing the labelling recipe).
"""

import glob
import json
import os
import shutil

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from src.config import (
    ARTIFACTS_DIR,
    BIOLOGY_SEED,
    COMMONS_DIR,
    DEVICE,
    DOMAIN_DIRS,
    DOMAIN_OTHER_DIR,
    PHYSICS_SEED,
    POTEC_DIR,
)
from src.features.potec import read_potec

# When this exists, labelling is skipped and datasets are rebuilt from it —
# UNLESS DOMAIN_REBUILD=1, which forces a full relabel + overwrite.
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"
REBUILD = os.environ.get("DOMAIN_REBUILD", "0") == "1"

# Labelling window. TF-IDF has no length limit; the NLI model truncates to its
# own max, so this only bounds NLI cost. NOTE: still head-biased — a doc whose
# opening reads on-topic but whose body drifts is judged on the opening. The
# quality gate (whole-doc metadata) is what actually removes junk bodies.
TEXT_CHARS = 2048
BATCH_SIZE = 128
TFIDF_THRESHOLD = 0.05

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
# The neutral pool is exactly the pass-1 "other" set: docs whose TF-IDF
# similarity to BOTH domain seeds falls below TFIDF_THRESHOLD, i.e. everything
# the candidate filter drops. Same corpus and register as the domain priors, no
# physics/biology content — the length-matched control for the prompt arm.
# Fluency gate: drop the worst-OCR / least-fluent docs by perplexity percentile,
# so the neutral priors read as cleanly as the domain priors.
NEUTRAL_MAX_PPL_PCTL = 60
# Pool size: a broad sample to draw K priors from, not the whole neutral mass.
N_OTHER_DOCS = 1000
OTHER_SEED = 0

# German-native zero-shot NLI (GBERT-large). Data is strictly German, so a
# monolingual German model beats multilingual XLM-R/mDeBERTa here. It is
# template-based: short label WORDS filled into a German hypothesis template
# (not full pre-built sentences under the pipeline's default English template).
NLI_MODEL = "svalabs/gbert-large-zeroshot-nli"
NLI_LABELS = {"physics": "Physik", "biology": "Biologie"}
NLI_HYPOTHESIS_TEMPLATE = "In diesem Text geht es um {}."


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


def domain_sims(texts):
    """TF-IDF char n-gram similarity of every doc to the two domain seed bags."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        max_features=100_000,
        sublinear_tf=True,
        min_df=3,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts + [PHYSICS_SEED, BIOLOGY_SEED])
    corpus_vecs = tfidf_matrix[:-2]
    sim_physics = cosine_similarity(corpus_vecs, tfidf_matrix[-2]).ravel()
    sim_biology = cosine_similarity(corpus_vecs, tfidf_matrix[-1]).ravel()
    return sim_physics, sim_biology


def select_other(ds, sim_physics, sim_biology):
    """Neutral pool: the pass-1 "other" docs — outside the candidate filter.

    A doc is "other" when both domain similarities fall below TFIDF_THRESHOLD, so
    this is exactly the complement of the pass-1 candidate set. A per-pool
    perplexity fluency gate drops the worst-OCR docs, then a shuffled sample of up
    to ``N_OTHER_DOCS`` is returned.
    """
    idx = [
        i
        for i, (sp, sb) in enumerate(zip(sim_physics, sim_biology))
        if sp < TFIDF_THRESHOLD and sb < TFIDF_THRESHOLD
    ]
    other = ds.select(idx)
    # fluency gate: keep the lower-perplexity share (missing ppl -> kept).
    ppl = np.array(
        [p if p is not None else np.nan for p in other["perplexity"]], dtype=float
    )
    finite = ppl[np.isfinite(ppl)]
    if len(finite):
        cutoff = np.percentile(finite, NEUTRAL_MAX_PPL_PCTL)
        keep = [i for i, p in enumerate(ppl) if not np.isfinite(p) or p <= cutoff]
        other = other.select(keep)
    other = other.shuffle(seed=OTHER_SEED)
    return other.select(range(min(N_OTHER_DOCS, len(other))))


def _label_scores(result, keys, hyps):
    """Map a zero-shot result to ``{domain_key: entailment_prob}``.

    The pipeline returns ``labels`` (hypothesis strings, score-sorted) and a
    parallel ``scores`` list; invert the hypothesis->key map to recover a
    per-domain score dict regardless of ordering.
    """
    hyp_to_key = {h: k for k, h in zip(keys, hyps)}
    return {hyp_to_key[lbl]: sc for lbl, sc in zip(result["labels"], result["scores"])}


# Calibrate the NLI MEMBERSHIP threshold on the 12 gold POTEC texts.
def calibrate_nli_threshold(potec_dir, classifier):
    """Absolute entailment threshold separating in-domain from off-domain.

    Multi-label NLI gives each domain an independent entailment probability. For
    every gold PoTeC text the probability of its TRUE domain is a positive; the
    probability of the OTHER domain (a physics text scored under "…von Biologie")
    is a negative. The threshold is the midpoint of the worst positive and the
    best negative — a domain-membership boundary, NOT a physics-vs-biology
    tie-break.
    """
    records = []
    for f in sorted(
        glob.glob(os.path.join(potec_dir, "stimuli/word_features/word_features_*.tsv"))
    ):
        df = read_potec(f)
        records.append(
            {
                "text_id": os.path.basename(f)
                .replace("word_features_", "")
                .replace(".tsv", ""),
                "gold": df["text_domain"].iloc[0],
                "text": " ".join(df["word"].fillna("").astype(str).tolist()),
            }
        )

    hyps = list(NLI_LABELS.values())
    keys = list(NLI_LABELS.keys())
    results = classifier(
        [r["text"][:TEXT_CHARS] for r in records],
        candidate_labels=hyps,
        hypothesis_template=NLI_HYPOTHESIS_TEMPLATE,
        multi_label=True,
        batch_size=16,
    )

    print("\nC: NLI membership scores on POTEC (gold labels):")
    pos_scores, neg_scores = [], []
    for rec, res in zip(records, results):
        sc = _label_scores(res, keys, hyps)
        pos = sc[rec["gold"]]
        negs = [sc[k] for k in keys if k != rec["gold"]]
        pos_scores.append(pos)
        neg_scores.extend(negs)
        print(
            f"  [{rec['text_id']}] gold={rec['gold']} "
            f"in-domain={pos:.3f} off-domain_max={max(negs):.3f}"
        )

    # membership boundary: midpoint of worst in-domain and best off-domain score.
    if pos_scores and neg_scores:
        threshold = (min(pos_scores) + max(neg_scores)) / 2
    else:
        threshold = min(pos_scores) if pos_scores else 0.5
    print(f"\n  in-domain scores : min={min(pos_scores):.3f} max={max(pos_scores):.3f}")
    if neg_scores:
        print(
            f"  off-domain scores: min={min(neg_scores):.3f} max={max(neg_scores):.3f}"
        )
    print(f"  → calibrated membership threshold: {threshold:.3f}")
    return threshold


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
    exists OR ``DOMAIN_REBUILD=1`` (which forces a fresh relabel). Skips the NLI
    pass entirely. The cached ids already encode the quality gate, so no
    re-filtering here.
    """
    if REBUILD or not path.exists():
        return None
    with open(path) as f:
        ids = json.load(f)
    print(f"\nFound cached label ids ({path}); skipping quality gate + TF-IDF + NLI.")
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


def _pass1_tfidf(texts):
    """Pass 1 — TF-IDF domain labels + the candidate set (everything not "other")."""
    print(f"\nPass 1 — TF-IDF on {len(texts):,} docs...")
    sim_physics, sim_biology = domain_sims(texts)

    pass1_labels = []
    for sp, sb in zip(sim_physics, sim_biology):
        if sp < TFIDF_THRESHOLD and sb < TFIDF_THRESHOLD:
            pass1_labels.append("other")
        elif sp >= sb:
            pass1_labels.append("physics")
        else:
            pass1_labels.append("biology")

    candidates_idx = [i for i, l in enumerate(pass1_labels) if l != "other"]
    print(
        f"  candidates: {len(candidates_idx):,} "
        f"({pass1_labels.count('physics'):,} physics, "
        f"{pass1_labels.count('biology'):,} biology)"
    )
    return sim_physics, sim_biology, pass1_labels, candidates_idx


def _pass2_nli(texts, candidates_idx):
    """Pass 2 — multi-label NLI membership gate over the pass-1 candidates.

    Each candidate gets an independent entailment probability per domain. It is
    assigned the ARGMAX domain and kept only when that probability clears the
    membership threshold; everything else (including off-domain docs the TF-IDF
    prefilter let through) falls back to "other". The pass-1 label is NOT used
    for the decision — NLI resolves the domain — it only bounded the candidate
    set. Returns ``(final_labels, nli_scores)`` over ALL docs.
    """
    print(f"\nLoading NLI model (device={DEVICE})...")

    classifier = pipeline(
        "zero-shot-classification",
        model=NLI_MODEL,
        device=DEVICE,
    )

    nli_threshold = calibrate_nli_threshold(str(POTEC_DIR), classifier)

    print(
        f"\nPass 2 — NLI on {len(candidates_idx):,} candidates "
        f"(membership threshold={nli_threshold:.3f})..."
    )
    hyps = list(NLI_LABELS.values())
    keys = list(NLI_LABELS.keys())
    results = classifier(
        [texts[i] for i in candidates_idx],
        candidate_labels=hyps,
        hypothesis_template=NLI_HYPOTHESIS_TEMPLATE,
        multi_label=True,
        batch_size=BATCH_SIZE,
    )

    final_labels = ["other"] * len(texts)
    nli_scores = [0.0] * len(texts)

    for idx, result in zip(candidates_idx, results):
        sc = _label_scores(result, keys, hyps)
        best = max(keys, key=lambda k: sc[k])
        if sc[best] >= nli_threshold:
            final_labels[idx] = best
            nli_scores[idx] = sc[best]
    return final_labels, nli_scores


def main():
    # Reuse a previous run's labels if they were saved (unless DOMAIN_REBUILD=1).
    cached = build_from_label_ids()
    if cached is not None:
        _save_domain_datasets(*cached)
        return

    ds = apply_quality_filter(load_commons())
    texts = [t[:TEXT_CHARS] if t else "" for t in ds["text"]]

    sim_physics, sim_biology, pass1_labels, candidates_idx = _pass1_tfidf(texts)
    final_labels, nli_scores = _pass2_nli(texts, candidates_idx)

    # ── results ──────────────────────────────────────────────────────────────
    ds = (
        ds.add_column("domain_label", final_labels)
        .add_column("sim_physics", sim_physics.tolist())
        .add_column("sim_biology", sim_biology.tolist())
        .add_column("nli_score", nli_scores)
    )

    physics_ds = ds.filter(lambda x: x["domain_label"] == "physics")
    biology_ds = ds.filter(lambda x: x["domain_label"] == "biology")
    # neutral pool for the prompt control — the pass-1 "other" docs.
    other_ds = select_other(ds, sim_physics, sim_biology)

    print("\nFinal results (quality ∩ pass 1 ∩ pass 2):")
    print(
        f"physics : {len(physics_ds):,}  ({sum(r['num_tokens'] for r in physics_ds):,} tokens)"
    )
    print(
        f"biology : {len(biology_ds):,}  ({sum(r['num_tokens'] for r in biology_ds):,} tokens)"
    )
    print(
        f"neutral : {len(other_ds):,}  (outside pass 1: both sims < {TFIDF_THRESHOLD})"
    )
    print(
        f"rejected by NLI: {len(candidates_idx) - len(physics_ds) - len(biology_ds):,}"
    )

    print("\n-- top physics docs --")
    for row in sorted(physics_ds, key=lambda x: x["nli_score"], reverse=True)[:3]:
        print(
            f"  nli={row['nli_score']:.2f} tfidf={row['sim_physics']:.3f} [{row['source']}] {row['text'][:100]!r}"
        )

    print("\n-- top biology docs --")
    for row in sorted(biology_ds, key=lambda x: x["nli_score"], reverse=True)[:3]:
        print(
            f"  nli={row['nli_score']:.2f} tfidf={row['sim_biology']:.3f} [{row['source']}] {row['text'][:100]!r}"
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
