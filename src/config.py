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
# approach. Built by `python -m src.acquire.scrape`.
DOMAIN_DIRS = {domain: DATA_DIR / f"wiki_{domain}" for domain in DOMAINS}

# Per-domain doc-level cutoff for acquire.cluster_filter: a scraped article whose
# SBERT embedding scores below this cosine to the domain's curriculum-term anchor
# is dropped as off-domain. Per domain BY NECESSITY — the anchor-cosine baseline
# differs (physics docs sit at median ~0.43, biology ~0.35), so one global cut
# means different things in each. Physics contamination is small but diffuse
# (~3%, doesn't form droppable clusters), so a low cut catches its tail; biology
# contamination is large and clustered (~18%), so its cut runs higher. Tuned off
# the inspect-pass cosine table — re-judge if the scrape or EMBED_MODEL changes.
DOMAIN_COS_THRESHOLD = {"physics": 0.12, "biology": 0.20}

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
# per-word surprisal across this many distinct priors
N_PRIOR_PASSAGES = 20
