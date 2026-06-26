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

# Decoder LMs the whole pipeline is run over, slug -> HF repo. Every model runs
# the same workflow independently (surprisal, attention, DAPT, model comparison),
# then the three are compared. All German: german-gpt2 (124M) plus the LLäMmlein
# family (LSX-UniWue), German-ONLY decoders trained from scratch on one dataset at
# 1B and 7B — a clean parameter-scaling comparison at the same training data.
# Weights are NOT downloaded here; the GPU run pulls them on first use.
MODELS = {
    "german-gpt2": "dbmdz/german-gpt2",
    "llammlein-1b": "LSX-UniWue/LLaMmlein_1B",
    "llammlein-7b": "LSX-UniWue/LLaMmlein_7B",
}

# Per-model DAPT (step 4) train batch size: bigger models need a smaller batch to
# fit VRAM. Tune on the GPU machine; grad-accum in finetune_dapt lifts the
# effective batch back up without more memory.
DAPT_BATCH_SIZE = {
    "german-gpt2": 8,
    "llammlein-1b": 4,
    "llammlein-7b": 1,
}
# German encoder for the attention experiment (src/experiment). Mouratidi &
# Poesio (2025) find attention flow on an ENCODER aligns best with gaze; their
# bert-base-uncased is English, so we use a German BERT for the German corpus.
ENCODER_MODEL = "deepset/gbert-base"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"          # surprisal/attention/analysis tables
# Run artifacts (checkpoints + the domain label-id map). Relative -> resolved
# against the cwd at run time, so artifacts land in ./artifacts/ wherever the
# pipeline is launched.
ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR                 # fine-tuned model weights (DAPT)
# domain_preprocessing's expensive NLI labelling output (tracked in git).
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"

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

# ── reader-discipline system prompt (prompted-surprisal variant) ─────────────
# Baseline-model surprisal under a discipline-matched system prompt — the
# prompting analogue of the fine-tuned "aligned" model. The prompt is chosen by
# the READER's discipline (a physicist gets the physics prompt on any text), so
# the variant needs both columns per word; the per-reader mix is built in
# model_comparison._prep_models. German to match the german-gpt2 base model.
GRAD_STUDENT_PROMPTS = {
    "physics": "Du bist Doktorand der Physik. ",
    "biology": "Du bist Doktorand der Biologie. ",
}
