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

# ── general-domain validation corpus (cross-domain perplexity) ───────────────
# German Wikipedia: general, non-scientific German prose — the out-of-domain
# reference for cross-domain perplexity (how DAPT for physics/biology shifts the
# model's fit on plain general text). Streamed, so only the first slice is
# pulled, never the full dump. 20231101.de = the 2023-11-01 German dump.
GENERAL_HF_REPO = "wikimedia/wikipedia"
GENERAL_HF_CONFIG = "20231101.de"

# ── PoTeC sub-paths used by the analysis pipeline ────────────────────────────
POTEC_WORD_FEATURES_DIR = POTEC_DIR / "stimuli" / "word_features"
POTEC_READING_MEASURES_DIR = POTEC_EYETRACKING_DIR / "reading_measures_merged"
POTEC_PARTICIPANTS = POTEC_DIR / "participants" / "participant_data.tsv"

# ── pipeline model + outputs ─────────────────────────────────────────────────
# Default decoder-only causal LM (German). The loaders stay model-agnostic
# within the decoder family, so any HF name or local checkpoint can be swapped in.
DEFAULT_MODEL = "dbmdz/german-gpt2"

# Decoder LMs the whole pipeline is run over, slug -> HF repo. Every model runs
# the same workflow independently (surprisal, DAPT, model comparison),
# then they are compared. All German: german-gpt2 (124M) plus the LLäMmlein 1B
# (LSX-UniWue), a German-ONLY decoder trained from scratch on one dataset.
# Weights are NOT downloaded here; the GPU run pulls them on first use.
MODELS = {
    "german-gpt2": "dbmdz/german-gpt2",
    "llammlein-1b": "LSX-UniWue/LLaMmlein_1B",
}

# Per-model DAPT (step 4) train batch size, sized for ~16 GB VRAM with the
# LoRA path + bf16 + block_size=512. Bigger models need a smaller batch to fit:
# frozen base weights eat the budget (1B≈2 GB in bf16), the rest is activations.
# 1B's 6 is close to the 16 GB limit (drop it / raise grad-accum if you OOM).
# grad-accum in finetune_dapt lifts the effective batch without more memory.
DAPT_BATCH_SIZE = {
    "german-gpt2": 8,
    "llammlein-1b": 2,  # 12 GB VRAM: batch 6 OOMs; grad_accum 3 keeps effective batch=6
}
# Gradient accumulation per model. Effective batch = batch_size × grad_accum. Use to
# recover effective batch size after reducing batch_size for VRAM, without extra memory.
DAPT_GRAD_ACCUM = {
    "llammlein-1b": 3,  # 2 × 3 = 6 effective (matches original intent)
}

OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # surprisal / analysis tables
# Run artifacts (checkpoints + the domain label-id map). Relative -> resolved
# against the cwd at run time, so artifacts land in ./artifacts/ wherever the
# pipeline is launched.
ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR  # fine-tuned model weights (DAPT)
# domain_preprocessing's expensive NLI labelling output (tracked in git).
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"

# Per-word join key for every word-level table (surprisal, reading times).
WORD_KEY = ["text_id", "word_index_in_text"]

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

# ── field × level system prompts (prompted-surprisal, second variant) ────────
# A richer prompted baseline than GRAD_STUDENT_PROMPTS: the system prompt matches
# BOTH the reader's discipline AND level of studies (undergraduate vs graduate),
# i.e. the prompting analogue of "an undergrad-/graduate-level physicist/biologist".
# Keyed (level_of_studies_numeric, reader_discipline_numeric): level 0 undergrad,
# 1 graduate (reading_time.LEVEL_LABELS); discipline 1 physics, 0 biology. Mixed
# per reader by both columns in model_comparison._prep_models. German to match the
# german-gpt2 base model. Level phrasing is degree-neutral: PoTeC ``graduate``
# spans Master/Diplom/PhD (participant_data.semester), so the graduate prompt says
# "fortgeschritten" rather than naming a specific degree.
FIELD_LEVEL_PROMPTS = {
    (0, 1): "Du bist Physikstudent im Grundstudium. ",
    (1, 1): "Du bist fortgeschrittener Physikstudent. ",
    (0, 0): "Du bist Biologiestudent im Grundstudium. ",
    (1, 0): "Du bist fortgeschrittener Biologiestudent. ",
}
