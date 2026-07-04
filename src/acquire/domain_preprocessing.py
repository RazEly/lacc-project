"""Label the german-commons scientific corpus by domain (physics / biology).

Two-pass weak labelling:
  pass 1 — TF-IDF char n-gram similarity against physics/biology keyword bags.
  pass 2 — zero-shot NLI (XLM-RoBERTa) with a threshold calibrated on the 12
           gold-labelled PoTeC texts.

Also carves out a NEUTRAL pool (``DOMAIN_OTHER_DIR``): the docs that fall OUTSIDE
the pass-1 candidate set — both domain TF-IDF similarities below the candidate
threshold. Same corpus and register as the domain priors, no physics/biology
content. Feeds the neutral prompt condition (features.priors).

Run from the project root:
    python -m src.acquire.domain_preprocessing
"""

import glob
import json
import os

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import XLMRobertaTokenizer, pipeline

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

# When this exists, labelling is skipped and datasets are rebuilt from it.
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"

TEXT_CHARS = 1024
BATCH_SIZE = 128
TFIDF_THRESHOLD = 0.05

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

    OpenAlex OCR gating already happened at download time (acquire.download_commons),
    so the saved corpus is clean — no re-filtering here.
    """
    ds = load_from_disk(str(COMMONS_DIR))
    if not hasattr(ds, "column_names") or isinstance(ds.column_names, dict):
        ds = concatenate_datasets([ds[name] for name in ds.keys()])
    return ds


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


# Calibrate the NLI threshold on the 12 gold POTEC texts before applying to corpus.
def calibrate_nli_threshold(potec_dir, classifier):
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
        multi_label=False,
        batch_size=16,
    )

    print("\nC: NLI scores on POTEC (gold labels):")
    correct_scores, wrong_scores = [], []
    for rec, res in zip(records, results):
        top_label = keys[hyps.index(res["labels"][0])]
        score = res["scores"][0]
        correct = top_label == rec["gold"]
        mark = "✓" if correct else "✗"
        print(
            f"  {mark} [{rec['text_id']}] gold={rec['gold']} nli={top_label} score={score:.3f}"
        )
        (correct_scores if correct else wrong_scores).append(score)

    # threshold = midpoint between lowest correct and highest wrong score
    if wrong_scores and correct_scores:
        threshold = (min(correct_scores) + max(wrong_scores)) / 2
    else:
        threshold = min(correct_scores) if correct_scores else 0.6
    print(
        f"\n  correct scores: min={min(correct_scores):.3f} max={max(correct_scores):.3f}"
    )
    if wrong_scores:
        print(
            f"  wrong scores:   min={min(wrong_scores):.3f} max={max(wrong_scores):.3f}"
        )
    print(f"  → calibrated threshold: {threshold:.3f}")
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
    exists. Skips the NLI pass entirely.
    """
    if not path.exists():
        return None
    with open(path) as f:
        ids = json.load(f)
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


def _pass2_nli(texts, candidates_idx, pass1_labels):
    """Pass 2 — zero-shot NLI over the candidates, threshold calibrated on PoTeC.

    A candidate keeps its pass-1 label only when the NLI top label agrees AND its
    score clears the calibrated threshold; everything else falls back to "other".
    Returns ``(final_labels, nli_scores)`` over ALL docs (non-candidates "other"/0).
    """
    print(f"\nLoading NLI model (device={DEVICE})...")

    _model_name = "joeddav/xlm-roberta-large-xnli"
    classifier = pipeline(
        "zero-shot-classification",
        model=_model_name,
        tokenizer=XLMRobertaTokenizer.from_pretrained(_model_name),
        device=DEVICE,
    )

    nli_threshold = calibrate_nli_threshold(str(POTEC_DIR), classifier)

    print(
        f"\nPass 2 — NLI on {len(candidates_idx):,} candidates (threshold={nli_threshold:.3f})..."
    )
    hyps = list(NLI_LABELS.values())
    keys = list(NLI_LABELS.keys())
    results = classifier(
        [texts[i] for i in candidates_idx],
        candidate_labels=hyps,
        multi_label=False,
        batch_size=BATCH_SIZE,
    )

    final_labels = ["other"] * len(texts)
    nli_scores = [0.0] * len(texts)

    for idx, p1_label, result in zip(
        candidates_idx, [pass1_labels[i] for i in candidates_idx], results
    ):
        top_label = keys[hyps.index(result["labels"][0])]
        score = result["scores"][0]
        if top_label == p1_label and score >= nli_threshold:
            final_labels[idx] = p1_label
            nli_scores[idx] = score
    return final_labels, nli_scores


def main():
    # Reuse a previous run's labels if they were saved.
    cached = build_from_label_ids()
    if cached is not None:
        _save_domain_datasets(*cached)
        return

    ds = load_commons()
    texts = [t[:TEXT_CHARS] if t else "" for t in ds["text"]]

    sim_physics, sim_biology, pass1_labels, candidates_idx = _pass1_tfidf(texts)
    final_labels, nli_scores = _pass2_nli(texts, candidates_idx, pass1_labels)

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
