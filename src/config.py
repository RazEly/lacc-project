"""Central paths and dataset identifiers for the project.

All paths are resolved relative to the project root (the parent of src/),
so scripts work regardless of the current working directory.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── PoTeC (eye-tracking corpus) ──────────────────────────────────────────────
POTEC_DIR = DATA_DIR / "potec"
POTEC_REPO_URL = "https://github.com/DiLi-Lab/PoTeC"
POTEC_EYETRACKING_DIR = POTEC_DIR / "eyetracking_data"

# ── german-commons (fine-tuning corpus) ──────────────────────────────────────
COMMONS_DIR = DATA_DIR / "commons_scientific"
COMMONS_HF_REPO = "coral-nlp/german-commons"
COMMONS_HF_CONFIG = "scientific"

# ── domain_preprocessing outputs ─────────────────────────────────────────────
DOMAIN_PHY_DIR = DATA_DIR / "domain_phy"
DOMAIN_BIO_DIR = DATA_DIR / "domain_bio"

# ── PoTeC sub-paths used by the analysis pipeline ────────────────────────────
POTEC_WORD_FEATURES_DIR = POTEC_DIR / "stimuli" / "word_features"
POTEC_READING_MEASURES_DIR = POTEC_EYETRACKING_DIR / "reading_measures_merged"
POTEC_PARTICIPANTS = POTEC_DIR / "participants" / "participant_data.tsv"

# ── pipeline model + outputs ─────────────────────────────────────────────────
# Default decoder-only causal LM (German). The loaders stay model-agnostic
# within the decoder family, so any HF name or local checkpoint can be swapped in.
DEFAULT_MODEL = "dbmdz/german-gpt2"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"          # surprisal/attention/analysis tables
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"  # fine-tuned model weights (DAPT)

# Eye-tracking measure name mapping: plan name -> PoTeC column.
ET_MEASURE_MAP = {
    "GD": "FPRT",   # gaze duration       == first-pass reading time
    "TRT": "TFT",   # total reading time  == total fixation time
    "FFD": "FFD",   # first fixation duration
    "SFD": "SFD",   # single fixation duration
    "GPT": "RPD_inc",  # go-past time     == inclusive regression-path duration
    "F": "TFC",     # fixation count      == total fixation count
}

# The 4 informative measures kept for PCA (drop SFD, GPT per plan / Mouratidi & Poesio 2025).
PCA_MEASURES = ["FPRT", "TFT", "FFD", "TFC"]
