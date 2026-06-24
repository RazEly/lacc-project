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
