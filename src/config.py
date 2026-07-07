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


DOMAIN_COS_THRESHOLD = {"physics": 0.12, "biology": 0.20}

# Default decoder LM; loaders are model-agnostic within the decoder family.
DEFAULT_MODEL = "benjamin/gerpt2"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SURPRISAL_CACHE_PATH = ARTIFACTS_DIR / "surprisal.csv"

# Per-word join key for every word-level table.
WORD_KEY = ["text_id", "word_index_in_text"]

# Prompts averaged per prior condition. Each condition's surprisal is the mean
# per-word surprisal across this many distinct priors
N_PRIOR_PASSAGES = 20
