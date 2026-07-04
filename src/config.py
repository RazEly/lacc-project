"""Central paths and dataset identifiers. Paths resolve from the project root.

Knobs used by a single module live in that module (e.g. the DAPT hyper-parameter
tables in main, the PoTeC sub-paths in features.potec). This file holds only what
more than one module shares.
"""

from pathlib import Path

import torch

# Single compute-device decision for the whole pipeline: pick CUDA when present,
# else CPU. Modules move models/inputs onto this and gate CUDA-only flags on it,
# so a run never splits work across devices.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# PoTeC (eye-tracking corpus). Sub-paths (word_features / reading measures /
# eyetracking) are derived where they're read — acquire.download_potec, features.potec.
POTEC_DIR = DATA_DIR / "potec"

# german-commons (fine-tuning corpus). The HF repo/config id and the OpenAlex
# OCR gate live in acquire.download_commons (its only reader).
COMMONS_DIR = DATA_DIR / "commons_scientific"

# The two DAPT domains. The full name is the ONLY spelling anywhere — dataset
# labels, cache-column suffixes, run dirs, data dirs — no shorthand aliases.
DOMAINS = ("physics", "biology")

# TF-IDF domain seeds: keyword bags, not prose — prose shares too many char
# n-grams with any German text, collapsing the similarity spread. Domain-term
# bags avoid that. Consumed by acquire.domain_preprocessing.domain_sims.
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

# domain_preprocessing outputs — training-domain corpora by domain name
# (finetune loads these; priors adds "neutral").
DOMAIN_DIRS = {domain: DATA_DIR / f"domain_{domain}" for domain in DOMAINS}
# Neutral control pool: german-commons docs OUTSIDE the pass-1 candidate set
# (both domain TF-IDF sims below threshold) — same corpus and register as the
# domain pools, no physics/biology content. See
# acquire.domain_preprocessing.select_other.
DOMAIN_OTHER_DIR = DATA_DIR / "domain_other"

# Default decoder LM; loaders are model-agnostic within the decoder family.
DEFAULT_MODEL = "dbmdz/german-gpt2"

# Run artifacts — anchored to the project root like every other path, so runs
# from any cwd hit the same DAPT/surprisal caches (a cwd-relative path made a
# wrong-dir run silently retrain everything).
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# Wide per-word surprisal cache: computed once, reloaded to skip model forwards.
# Delete the file (and the DAPT run dirs) by hand to force a recompute.
SURPRISAL_CACHE_PATH = ARTIFACTS_DIR / "surprisal.csv"

# Per-word join key for every word-level table.
WORD_KEY = ["text_id", "word_index_in_text"]

# Prompts averaged per prior condition. Each condition's surprisal is the mean
# per-word surprisal across this many DISTINCT priors (K distinct held-out docs
# per domain, K distinct neutral passages) so no single idiosyncratic passage
# drives the estimate. 5 sits at the variance-reduction elbow (1/sqrt(K) shrinks
# fast to K≈5, then flattens); raise for the final run if a K-sensitivity check
# still drifts. The off-domain pool (DOMAIN_OTHER_DIR) needs >= this many docs.
N_PRIOR_PASSAGES = 5
