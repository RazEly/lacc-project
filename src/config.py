"""Central paths and dataset identifiers. Paths resolve from the project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# PoTeC (eye-tracking corpus)
POTEC_DIR = DATA_DIR / "potec"
POTEC_REPO_URL = "https://github.com/DiLi-Lab/PoTeC"
POTEC_EYETRACKING_DIR = POTEC_DIR / "eyetracking_data"

# german-commons (fine-tuning corpus)
COMMONS_DIR = DATA_DIR / "commons_scientific"
COMMONS_HF_REPO = "coral-nlp/german-commons"
COMMONS_HF_CONFIG = "scientific"

# domain_preprocessing outputs
DOMAIN_PHY_DIR = DATA_DIR / "domain_phy"
DOMAIN_BIO_DIR = DATA_DIR / "domain_bio"

# PoTeC sub-paths used by the analysis pipeline
POTEC_WORD_FEATURES_DIR = POTEC_DIR / "stimuli" / "word_features"
POTEC_READING_MEASURES_DIR = POTEC_EYETRACKING_DIR / "reading_measures_merged"
POTEC_PARTICIPANTS = POTEC_DIR / "participants" / "participant_data.tsv"

# Default decoder LM; loaders are model-agnostic within the decoder family.
DEFAULT_MODEL = "dbmdz/german-gpt2"

# Decoder LMs the pipeline runs over (slug -> HF repo), then compares. Both German:
# german-gpt2 (124M) + LLäMmlein 1B (German-only, from scratch). Pulled at run time.
MODELS = {
    "german-gpt2": "dbmdz/german-gpt2",
    "llammlein-1b": "LSX-UniWue/LLaMmlein_1B",
}

# Short column prefix per model for the wide surprisal cache (surprisal.csv):
# <prefix>_0 baseline, <prefix>_<i>_<phys|bio> checkpoints, <prefix>_prompt_* prompted.
MODEL_PREFIX = {
    "german-gpt2": "gpt",
    "llammlein-1b": "llama",
}

# Per-model DAPT train batch (LoRA + bf16 + block_size=512, ~16 GB VRAM).
DAPT_BATCH_SIZE = {
    "german-gpt2": 8,
    "llammlein-1b": 2,
}
# Effective batch = batch_size × grad_accum (recovers batch after VRAM cuts).
DAPT_GRAD_ACCUM = {
    "llammlein-1b": 3,  # 2 × 3 = 6 effective
}

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# Run artifacts (relative -> resolved against cwd at run time).
ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"
# Wide per-word surprisal cache: computed once, reloaded to skip model forwards.
SURPRISAL_CACHE_PATH = ARTIFACTS_DIR / "surprisal.csv"

# Per-word join key for every word-level table.
WORD_KEY = ["text_id", "word_index_in_text"]

# Prompted-surprisal baseline: discipline-matched system prompt, mixed per reader
# by discipline in model_comparison._prep_models. German (matches german-gpt2).
GRAD_STUDENT_PROMPTS = {
    "physics": "Du bist Doktorand der Physik. ",
    "biology": "Du bist Doktorand der Biologie. ",
}

# Richer variant: prompt matches discipline AND study level. Keyed
# (level_of_studies_numeric, reader_discipline_numeric): level 0 undergrad / 1
# graduate; discipline 1 physics / 0 biology. "fortgeschritten" since PoTeC
# graduate spans Master/Diplom/PhD.
FIELD_LEVEL_PROMPTS = {
    (0, 1): "Du bist Physikstudent im Grundstudium. ",
    (1, 1): "Du bist fortgeschrittener Physikstudent. ",
    (0, 0): "Du bist Biologiestudent im Grundstudium. ",
    (1, 0): "Du bist fortgeschrittener Biologiestudent. ",
}
