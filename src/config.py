"""Central paths and dataset identifiers. Paths resolve from the project root.

Knobs used by a single module live in that module (e.g. the DAPT hyper-parameter
tables in main, the PoTeC sub-paths in features.data). This file holds only what
more than one module shares.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# PoTeC (eye-tracking corpus). Sub-paths (word_features / reading measures /
# eyetracking) are derived where they're read — acquire.download_potec, features.data.
POTEC_DIR = DATA_DIR / "potec"

# german-commons (fine-tuning corpus). The HF repo/config id and the OpenAlex
# OCR gate live in acquire.download_commons (its only reader).
COMMONS_DIR = DATA_DIR / "commons_scientific"

# domain_preprocessing outputs
DOMAIN_PHY_DIR = DATA_DIR / "domain_phy"
DOMAIN_BIO_DIR = DATA_DIR / "domain_bio"
# Neutral control pool: german-commons docs OUTSIDE the pass-1 candidate set
# (both domain TF-IDF sims below threshold) — same corpus and register as the
# domain pools, no physics/biology content. See
# acquire.domain_preprocessing.select_other.
DOMAIN_OTHER_DIR = DATA_DIR / "domain_other"

# Default decoder LM; loaders are model-agnostic within the decoder family.
DEFAULT_MODEL = "dbmdz/german-gpt2"

# Run artifacts (relative -> resolved against cwd at run time).
ARTIFACTS_DIR = Path("artifacts")
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
