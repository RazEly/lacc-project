"""Label the german-commons scientific corpus by domain (physics / biology).

Two-pass weak labelling:
  pass 1 — TF-IDF char n-gram similarity against physics/biology term bags built
           from the PoTeC-stimuli domain seed terms (``config.PHYSICS_SEED_TERMS``
           / ``config.BIOLOGY_SEED_TERMS``).
  pass 2 — zero-shot NLI (XLM-RoBERTa); per domain, keep the top NLI-ranked docs
           whose label agrees with pass 1 until their token count hits a tuned
           budget (``DOMAIN_TOKEN_BUDGET``) — no fixed score threshold.

Also carves out a NEUTRAL pool (``DOMAIN_OTHER_DIR``): the docs that fall OUTSIDE
the pass-1 candidate set — both domain TF-IDF similarities below the candidate
threshold. Same corpus and register as the domain priors, no physics/biology
content. Feeds the neutral prompt condition (features.priors).

Run from the project root:
    python -m src.acquire.domain_preprocessing
"""

import json

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, pipeline

from src.config import (
    ARTIFACTS_DIR,
    BIOLOGY_SEED_TERMS,
    COMMONS_DIR,
    DEVICE,
    DOMAIN_DIRS,
    DOMAIN_OTHER_DIR,
    PHYSICS_SEED_TERMS,
)

# When this exists, labelling is skipped and datasets are rebuilt from it.
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"

TEXT_CHARS = 2048
BATCH_SIZE = 128
TFIDF_THRESHOLD = 0.1

# ── document quality gate ─────────────────────────────────────────────────────
# Applied to the WHOLE corpus before labelling, so physics / biology / neutral
# are all gated identically. This is the ONLY OCR gate in the pipeline —
# download_commons saves the corpus unfiltered.
#   • OCR score: present → must clear MIN_OCR_SCORE; missing → kept for
#     born-digital sources (Wikibooks etc.), dropped for OCR_REQUIRED_SOURCES
#     where a missing score means unassessed scan quality.
#   • Length: drop stubs / redirect pages and cap huge docs so no single doc
#     dominates the token budget.
MIN_OCR_SCORE = 80.0  # ocr_score is 0-100
OCR_REQUIRED_SOURCES = {"openalex"}  # OCR'd corpora: no score → drop
MIN_DOC_TOKENS = 128

# Pass-2 token budget per domain: keep the top TF-IDF-ranked docs (among those
# that clear the NLI floor and agree with pass 1) until their num_tokens sum
# reaches this. Sizes each DAPT domain corpus directly. Tuned hyperparameter.
DOMAIN_TOKEN_BUDGET = 700_000

# Pass-2 NLI floor: a candidate must score at least this on its NLI top label to
# survive, on top of agreeing with the pass-1 label. Selection ORDER is then by
# TF-IDF similarity, not NLI score.
NLI_MIN_SCORE = 0.5

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

NLI_LABELS = {
    "physics": "Dieser Text handelt von Physik.",
    "biology": "Dieser Text handelt von Biologie.",
}


def load_commons():
    """Load the saved german-commons scientific corpus as a single Dataset.

    Raw load only; the quality gate is applied separately
    (``apply_quality_filter``) so every downstream pool shares it.
    """
    ds = load_from_disk(str(COMMONS_DIR))
    if not hasattr(ds, "column_names") or isinstance(ds.column_names, dict):
        ds = concatenate_datasets([ds[name] for name in ds.keys()])
    return ds


def apply_quality_filter(ds):
    """Drop low-OCR / length-outlier docs before labelling.

    Keeps a doc when its OCR score clears ``MIN_OCR_SCORE`` (missing score: kept
    for born-digital sources, dropped for ``OCR_REQUIRED_SOURCES``) and its token
    count sits in ``[MIN_DOC_TOKENS, MAX_DOC_TOKENS]``.
    """

    def _ocr_ok(src, ocr):
        if ocr is None or ocr != ocr:  # missing / NaN
            return src not in OCR_REQUIRED_SOURCES
        return float(ocr) >= MIN_OCR_SCORE

    def _keep(src, ocr, ntok):
        ntok = ntok or 0
        return _ocr_ok(src, ocr) and MIN_DOC_TOKENS <= ntok

    idx = [
        i
        for i, (src, ocr, ntok) in enumerate(
            zip(ds["source"], ds["ocr_score"], ds["num_tokens"])
        )
        if _keep(src, ocr, ntok)
    ]
    kept = ds.select(idx)
    print(
        f"\nQuality gate: {len(kept):,}/{len(ds):,} docs kept "
        f"({len(ds) - len(kept):,} dropped: OCR/length)."
    )
    return kept


def load_domain_seeds():
    """Space-joined physics/biology TF-IDF seed bags from the config term lists.

    The seed terms live in ``src.config`` (``PHYSICS_SEED_TERMS`` /
    ``BIOLOGY_SEED_TERMS``); this joins each into the keyword bag TF-IDF scores
    against. Returns ``(physics_seed, biology_seed)``.
    """
    return " ".join(PHYSICS_SEED_TERMS), " ".join(BIOLOGY_SEED_TERMS)


def domain_sims(texts, physics_seed, biology_seed):
    """TF-IDF char n-gram similarity of every doc to the two domain seed bags."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        max_features=100_000,
        sublinear_tf=True,
        min_df=3,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts + [physics_seed, biology_seed])
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


# ── label-id cache ───────────────────────────────────────────────────────────
def save_label_ids(physics_ds, biology_ds, other_ds, path=LABEL_IDS_PATH):
    """Persist the selected physics/biology/other document ids as JSON for reuse."""
    ids = {
        "physics": list(physics_ds["id"]),
        "biology": list(biology_ds["id"]),
        "other": list(other_ds["id"]),
    }
    path.write_text(json.dumps(ids))
    print(
        f"  wrote label ids: {path} "
        f"({len(ids['physics']):,} physics, {len(ids['biology']):,} biology, "
        f"{len(ids['other']):,} other)"
    )


def build_from_label_ids(path=LABEL_IDS_PATH):
    """Rebuild the physics/biology/other datasets from a saved id file.

    Returns ``(physics_ds, biology_ds, other_ds)`` or ``None`` if no cache file
    exists. Skips the NLI pass entirely.
    """
    if not path.exists():
        return None
    ids = json.loads(path.read_text())
    print(f"\nFound cached label ids ({path}); skipping TF-IDF + NLI.")
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


def _pass1_tfidf(texts, physics_seed, biology_seed):
    """Pass 1 — TF-IDF domain labels + the candidate set (everything not "other")."""
    print(f"\nPass 1 — TF-IDF on {len(texts):,} docs...")
    sim_physics, sim_biology = domain_sims(texts, physics_seed, biology_seed)

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


def _pass2_nli(
    texts, candidates_idx, pass1_labels, num_tokens, sim_physics, sim_biology
):
    """Pass 2 — NLI floor, then a per-domain TF-IDF-ordered token-budget cut.

    Runs zero-shot NLI on the pass-1 candidates. A candidate survives only if the
    NLI top label agrees with its pass-1 label AND its NLI score clears
    ``NLI_MIN_SCORE``. Within each domain the survivors are ranked by TF-IDF
    similarity (to that domain's seed bag) and kept from the top down until their
    ``num_tokens`` sum reaches ``DOMAIN_TOKEN_BUDGET`` — so each domain corpus is
    sized to the same token budget, NLI-gated, TF-IDF-ordered.
    """
    print(f"\nLoading NLI model (device={DEVICE})...")

    _model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    classifier = pipeline(
        "zero-shot-classification",
        model=_model_name,
        tokenizer=AutoTokenizer.from_pretrained(_model_name),
        device=DEVICE,
    )

    print(
        f"\nPass 2 — NLI on {len(candidates_idx):,} candidates "
        f"(top docs per domain up to {DOMAIN_TOKEN_BUDGET:,} tokens)..."
    )
    hyps = list(NLI_LABELS.values())
    keys = list(NLI_LABELS.keys())

    results = classifier(
        [texts[i] for i in candidates_idx],
        candidate_labels=hyps,
        multi_label=False,
        batch_size=BATCH_SIZE,
    )

    # Keep docs where NLI agrees with pass 1 and clears the floor, per domain.
    domain_sim = {"physics": sim_physics, "biology": sim_biology}
    agree = {key: [] for key in keys}
    for idx, result in zip(candidates_idx, results):
        top_label = keys[hyps.index(result["labels"][0])]
        score = result["scores"][0]
        if top_label == pass1_labels[idx] and score >= NLI_MIN_SCORE:
            agree[top_label].append((idx, score))

    final_labels = ["other"] * len(texts)
    nli_scores = [0.0] * len(texts)

    # Per domain: take the highest-TF-IDF docs until the token budget is reached.
    for domain, scored in agree.items():
        scored.sort(key=lambda t: domain_sim[domain][t[0]], reverse=True)
        tokens = kept = 0
        for idx, score in scored:
            if tokens >= DOMAIN_TOKEN_BUDGET:
                break
            final_labels[idx] = domain
            nli_scores[idx] = score
            tokens += num_tokens[idx] or 0
            kept += 1
        print(
            f"  {domain}: kept {kept:,} of {len(scored):,} NLI-passing docs "
            f"({tokens:,} tokens, budget {DOMAIN_TOKEN_BUDGET:,})"
        )

    return final_labels, nli_scores


def main():
    # Reuse a previous run's labels if they were saved.
    cached = build_from_label_ids()
    if cached is not None:
        _save_domain_datasets(*cached)
        return

    ds = apply_quality_filter(load_commons())
    texts = [t[:TEXT_CHARS] if t else "" for t in ds["text"]]

    physics_seed, biology_seed = load_domain_seeds()
    sim_physics, sim_biology, pass1_labels, candidates_idx = _pass1_tfidf(
        texts, physics_seed, biology_seed
    )
    final_labels, nli_scores = _pass2_nli(
        texts, candidates_idx, pass1_labels, ds["num_tokens"], sim_physics, sim_biology
    )

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

    print("\nFinal results (pass 1 ∩ pass 2):")
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

    print("\n-- bottom physics docs --")
    for row in sorted(physics_ds, key=lambda x: x["nli_score"], reverse=False)[:7]:
        print(
            f"  nli={row['nli_score']:.2f} tfidf={row['sim_physics']:.3f} [{row['source']}] {row['text'][:150]!r}"
        )

    print("\n-- bottom biology docs --")
    for row in sorted(biology_ds, key=lambda x: x["nli_score"], reverse=False)[:7]:
        print(
            f"  nli={row['nli_score']:.2f} tfidf={row['sim_biology']:.3f} [{row['source']}] {row['text'][:150]!r}"
        )

    save_label_ids(physics_ds, biology_ds, other_ds)
    _save_domain_datasets(physics_ds, biology_ds, other_ds)


def _save_domain_datasets(physics_ds, biology_ds, other_ds):
    """Write the physics/biology/other datasets to disk.

    An EXISTING domain dir is left untouched — the DAPT checkpoints were trained
    on its exact contents, and priors.py samples its held-out split, so silently
    overwriting it (e.g. on a neutral-only backfill) would desync training and
    the priors. The neutral dir is always (re)written.
    """
    print("\nSaving datasets...")
    for ds, path in [
        (physics_ds, DOMAIN_DIRS["physics"]),
        (biology_ds, DOMAIN_DIRS["biology"]),
    ]:
        if path.exists():
            print(f"  skip {path} (exists — keeping the DAPT-trained version)")
        else:
            ds.save_to_disk(str(path))
            print(f"  saved {path}")
    other_ds.save_to_disk(str(DOMAIN_OTHER_DIR))
    print(f"  saved {DOMAIN_OTHER_DIR}")


if __name__ == "__main__":
    main()
