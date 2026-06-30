"""Label the german-commons scientific corpus by domain (physics / biology).

Two-pass weak labelling:
  pass 1 — TF-IDF char n-gram similarity against physics/biology keyword bags.
  pass 2 — zero-shot NLI (XLM-RoBERTa) with a threshold calibrated on the 12
           gold-labelled PoTeC texts.

Run from the project root:
    python -m src.acquire.domain_preprocessing
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import XLMRobertaTokenizer, pipeline

from src.config import (
    COMMONS_DIR,
    DOMAIN_BIO_DIR,
    DOMAIN_PHY_DIR,
    LABEL_IDS_PATH,
    POTEC_DIR,
)

TEXT_CHARS  = 1024
BATCH_SIZE  = 128   # tune to VRAM
TFIDF_THRESHOLD = 0.05

# When LABEL_IDS_PATH exists, labelling is skipped and datasets rebuilt from it.

NLI_LABELS = {
    "physics": "Dieser Text handelt von Physik.",
    "biology": "Dieser Text handelt von Biologie.",
}

# TF-IDF seeds: keyword bags, not prose — prose shares too many char n-grams with
# any German text, collapsing the similarity spread. Domain-term bags avoid that.
PHYSICS_SEED = """
Physik Quantenmechanik Thermodynamik Elektrodynamik Elektrochemie
Relativitätstheorie Optik Kernphysik Teilchenphysik Feldtheorie
Wellenmechanik Wellenlänge Frequenz Schwingung Strahlung Spektroskopie
Energie Masse Kraft Impuls Beschleunigung Geschwindigkeit Schwerkraft
Druck Temperatur Entropie Wärme Wärmeleitung Wärmekapazität
Elektron Proton Neutron Photon Atom Molekül Ion Plasma
Spannung Stromstärke Widerstand Magnetfeld Elektrisches Feld
Potential Welle Interferenz Beugung Brechung Reflexion
Planck Boltzmann Schrödinger Maxwell Newton Einstein Heisenberg Ohm
Joule Watt Volt Ampere Tesla Hertz Kelvin Pascal
"""

BIOLOGY_SEED = """
Biologie Zelle Organismus Evolution Genetik Erbgut
DNS RNS Protein Chromosom Gen Allel Mutation Genotyp Phänotyp
Art Taxonomie Ökologie Ökosystem Population Habitat Biotop
Photosynthese Stoffwechsel Zellatmung Glykolyse Enzym Rezeptor
Neuron Synapse Gehirn Nervensystem Immunsystem Antikörper
Mitose Meiose Fortpflanzung Embryo Zygote Gamete
Membran Zellkern Zytoplasma Ribosom Mitochondrium Chloroplast
Virus Bakterium Pilz Pflanze Tier Säugetier Wirbeltier
Darwin Mendel Koch Pasteur Virchow Haeckel
ATP Glucose Aminosäure Fettsäure Nukleotid
"""


# ── load corpus ──────────────────────────────────────────────────────────────
def load_commons():
    """Load the saved german-commons scientific corpus as a single Dataset."""
    ds = load_from_disk(str(COMMONS_DIR))
    # save_to_disk may yield a DatasetDict (one split per source); flatten it.
    if not hasattr(ds, "column_names") or isinstance(ds.column_names, dict):
        ds = concatenate_datasets([ds[name] for name in ds.keys()])
    return ds


# Calibrate the NLI threshold on the 12 gold POTEC texts before applying to corpus.
def calibrate_nli_threshold(potec_dir, classifier):
    records = []
    for f in sorted(glob.glob(os.path.join(potec_dir, "stimuli/word_features/word_features_*.tsv"))):
        df = pd.read_csv(f, sep="\t", keep_default_na=False,
            na_values=["#N/A","#N/A N/A","#NA","-1.#IND","-1.#QNAN","-NaN","-nan",
                       "1.#IND","1.#QNAN","<NA>","N/A","NA","NaN","None","n/a","nan",""])
        records.append({
            "text_id": os.path.basename(f).replace("word_features_","").replace(".tsv",""),
            "gold":    df["text_domain"].iloc[0],
            "text":    " ".join(df["word"].fillna("").astype(str).tolist()),
        })

    hyps = list(NLI_LABELS.values())
    keys = list(NLI_LABELS.keys())
    results = classifier([r["text"][:TEXT_CHARS] for r in records],
                         candidate_labels=hyps, multi_label=False, batch_size=16)

    print("\nC: NLI scores on POTEC (gold labels):")
    correct_scores, wrong_scores = [], []
    for rec, res in zip(records, results):
        top_label = keys[hyps.index(res["labels"][0])]
        score     = res["scores"][0]
        correct   = top_label == rec["gold"]
        mark      = "✓" if correct else "✗"
        print(f"  {mark} [{rec['text_id']}] gold={rec['gold']} nli={top_label} score={score:.3f}")
        (correct_scores if correct else wrong_scores).append(score)

    # threshold = midpoint between lowest correct and highest wrong score
    if wrong_scores and correct_scores:
        threshold = (min(correct_scores) + max(wrong_scores)) / 2
    else:
        threshold = min(correct_scores) if correct_scores else 0.6
    print(f"\n  correct scores: min={min(correct_scores):.3f} max={max(correct_scores):.3f}")
    if wrong_scores:
        print(f"  wrong scores:   min={min(wrong_scores):.3f} max={max(wrong_scores):.3f}")
    print(f"  → calibrated threshold: {threshold:.3f}")
    return threshold


# ── label-id cache ───────────────────────────────────────────────────────────
def save_label_ids(physics_ds, biology_ds, path=LABEL_IDS_PATH):
    """Persist the selected physics/biology document ids as JSON for reuse."""
    ids = {"physics": list(physics_ds["id"]), "biology": list(biology_ds["id"])}
    with open(path, "w") as f:
        json.dump(ids, f)
    print(f"  wrote label ids: {path} "
          f"({len(ids['physics']):,} physics, {len(ids['biology']):,} biology)")


def build_from_label_ids(path=LABEL_IDS_PATH):
    """Rebuild the physics/biology datasets from a saved id file.

    Returns ``(physics_ds, biology_ds)`` or ``None`` if no cache file exists.
    Skips the whole TF-IDF + NLI labelling pipeline.
    """
    if not path.exists():
        return None
    with open(path) as f:
        ids = json.load(f)
    print(f"\nFound cached label ids ({path}); skipping TF-IDF + NLI.")
    ds = load_commons()
    physics = set(ids["physics"])
    biology = set(ids["biology"])
    physics_ds = ds.filter(lambda x: x["id"] in physics)
    biology_ds = ds.filter(lambda x: x["id"] in biology)
    print(f"  physics : {len(physics_ds):,}   biology : {len(biology_ds):,}")
    return physics_ds, biology_ds


def main():
    # Reuse a previous run's labels if they were saved.
    cached = build_from_label_ids()
    if cached is not None:
        physics_ds, biology_ds = cached
        _save_domain_datasets(physics_ds, biology_ds)
        return

    # ── pass 1: TF-IDF with POTEC seeds ──────────────────────────────────────
    ds = load_commons()
    texts = [t[:TEXT_CHARS] if t else "" for t in ds["text"]]

    print(f"\nPass 1 — TF-IDF on {len(texts):,} docs...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        max_features=100_000,
        sublinear_tf=True,
        min_df=3,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts + [PHYSICS_SEED, BIOLOGY_SEED])
    corpus_vecs  = tfidf_matrix[:-2]
    sim_physics  = cosine_similarity(corpus_vecs, tfidf_matrix[-2]).ravel()
    sim_biology  = cosine_similarity(corpus_vecs, tfidf_matrix[-1]).ravel()

    pass1_labels = []
    for sp, sb in zip(sim_physics, sim_biology):
        if sp < TFIDF_THRESHOLD and sb < TFIDF_THRESHOLD:
            pass1_labels.append("other")
        elif sp >= sb:
            pass1_labels.append("physics")
        else:
            pass1_labels.append("biology")

    candidates_idx = [i for i, l in enumerate(pass1_labels) if l != "other"]
    print(f"  candidates: {len(candidates_idx):,} "
          f"({pass1_labels.count('physics'):,} physics, "
          f"{pass1_labels.count('biology'):,} biology)")

    # ── pass 2: NLI with calibrated threshold ────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    print(f"\nLoading NLI model (device={'cuda' if device==0 else 'cpu'})...")

    # AutoTokenizer wrongly tries tiktoken for XLM-RoBERTa in transformers>=4.47;
    # load SentencePiece tokenizer directly to bypass the bug.
    _model_name = "joeddav/xlm-roberta-large-xnli"
    classifier = pipeline(
        "zero-shot-classification",
        model=_model_name,
        tokenizer=XLMRobertaTokenizer.from_pretrained(_model_name),
        device=device,
    )

    nli_threshold = calibrate_nli_threshold(str(POTEC_DIR), classifier)

    print(f"\nPass 2 — NLI on {len(candidates_idx):,} candidates (threshold={nli_threshold:.3f})...")
    hyps      = list(NLI_LABELS.values())
    keys      = list(NLI_LABELS.keys())
    results   = classifier(
        [texts[i] for i in candidates_idx],
        candidate_labels=hyps,
        multi_label=False,
        batch_size=BATCH_SIZE,
    )

    final_labels = ["other"] * len(ds)
    nli_scores   = [0.0]    * len(ds)

    for idx, p1_label, result in zip(candidates_idx, [pass1_labels[i] for i in candidates_idx], results):
        top_label = keys[hyps.index(result["labels"][0])]
        score     = result["scores"][0]
        if top_label == p1_label and score >= nli_threshold:
            final_labels[idx] = p1_label
            nli_scores[idx]   = score

    # ── results ──────────────────────────────────────────────────────────────
    ds = (
        ds
        .add_column("domain_label", final_labels)
        .add_column("sim_physics",  sim_physics.tolist())
        .add_column("sim_biology",  sim_biology.tolist())
        .add_column("nli_score",    nli_scores)
    )

    physics_ds = ds.filter(lambda x: x["domain_label"] == "physics")
    biology_ds = ds.filter(lambda x: x["domain_label"] == "biology")

    print("\nFinal results (pass 1 ∩ pass 2):")
    print(f"  physics : {len(physics_ds):,}  ({sum(r['num_tokens'] for r in physics_ds):,} tokens)")
    print(f"  biology : {len(biology_ds):,}  ({sum(r['num_tokens'] for r in biology_ds):,} tokens)")
    print(f"  rejected by NLI: {len(candidates_idx) - len(physics_ds) - len(biology_ds):,}")

    print("\n-- top physics docs --")
    for row in sorted(physics_ds, key=lambda x: x["nli_score"], reverse=True)[:3]:
        print(f"  nli={row['nli_score']:.2f} tfidf={row['sim_physics']:.3f} [{row['source']}] {row['text'][:100]!r}")

    print("\n-- top biology docs --")
    for row in sorted(biology_ds, key=lambda x: x["nli_score"], reverse=True)[:3]:
        print(f"  nli={row['nli_score']:.2f} tfidf={row['sim_biology']:.3f} [{row['source']}] {row['text'][:100]!r}")

    # Cache the selected ids so a rerun can skip the labelling above.
    save_label_ids(physics_ds, biology_ds)
    _save_domain_datasets(physics_ds, biology_ds)


def _save_domain_datasets(physics_ds, biology_ds):
    """Write the physics/biology datasets to their on-disk locations."""
    print("\nSaving datasets...")
    physics_ds.save_to_disk(str(DOMAIN_PHY_DIR))
    biology_ds.save_to_disk(str(DOMAIN_BIO_DIR))
    print(f"  saved: {DOMAIN_PHY_DIR}, {DOMAIN_BIO_DIR}")
    print(f"  load with: load_from_disk('{DOMAIN_PHY_DIR}')")


if __name__ == "__main__":
    main()
