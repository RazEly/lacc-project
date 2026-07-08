"""Central paths and dataset identifiers. Paths resolve from the project root.

Knobs used by a single module live in that module (e.g. the DAPT hyper-parameter
tables in main, the PoTeC sub-paths in features.potec). This file holds only what
more than one module shares.
"""

from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# PoTeC corpus path
POTEC_DIR = DATA_DIR / "potec"

DOMAINS = ("physics", "biology")

# fine-tuning corpora dirs
DOMAIN_DIRS = {domain: DATA_DIR / f"wiki_{domain}" for domain in DOMAINS}

# Off-domain neutral corpus — source of the neutral prior passage (the single
# length-matched prompt pseudo-test replacing the physics/biology-for-all ones).
# NOT a DAPT domain and NOT a reader discipline; kept out of DOMAINS on purpose.
NEUTRAL_DIR = DATA_DIR / "wiki_neutral"


# Default decoder LM; loaders are model-agnostic within the decoder family.
DEFAULT_MODEL = "benjamin/gerpt2"

# Decoder LMs the study runs over (slug -> HF repo). Both German: german-gpt2
# (124M) + LLäMmlein 1B (German-only, from scratch). GPT and LLaMA archs supported.
MODELS = {
    "german-gpt2": "benjamin/gerpt2",
    "llammlein-1b": "LSX-UniWue/LLaMmlein_1B",
}

# DAPT checkpoint schedule: the adapter is saved at exactly these optimiser steps
# (checkpoint indices 1..; index 0 = un-fine-tuned base). Shared so a checkpoint
# index means the same training-token count for every model.
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4_096]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SURPRISAL_CACHE_PATH = ARTIFACTS_DIR / "surprisal.csv"

# Per-word join key for every word-level table.
WORD_KEY = ["text_id", "word_index_in_text"]

# Prompts averaged per prior condition. Each condition's surprisal is the mean
# per-word surprisal across this many distinct priors
N_PRIOR_PASSAGES = 20
