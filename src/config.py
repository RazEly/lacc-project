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

# PoTeC (eye-tracking corpus). Sub-paths (word_features / reading measures /
# eyetracking) are derived where they're read — acquire.download_potec, features.potec.
POTEC_DIR = DATA_DIR / "potec"

# The two DAPT domains. The full name is the ONLY spelling anywhere — dataset
# labels, cache-column suffixes, run dirs, data dirs — no shorthand aliases.
DOMAINS = ("physics", "biology")

# Fine-tuning corpora: the term-targeted German-Wikipedia scrape (acquire.scrape),
# one per domain — seeded from the PoTeC level-2 expert terms, the paper's own
# approach. Replaces the german-commons doc-classification selection (which lacked
# the expert terms DAPT needs; see notes). Built by `python -m src.acquire.scrape`.
DOMAIN_DIRS = {domain: DATA_DIR / f"wiki_{domain}" for domain in DOMAINS}

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
# per domain) so no single idiosyncratic passage drives the estimate. 5 sits at
# the variance-reduction elbow (1/sqrt(K) shrinks fast to K≈5, then flattens);
# raise for the final run if a K-sensitivity check still drifts. Each domain's
# held-out split (features.priors) needs >= this many docs.
N_PRIOR_PASSAGES = 20

