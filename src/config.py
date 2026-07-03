"""Central paths and dataset identifiers. Paths resolve from the project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# PoTeC (eye-tracking corpus)
POTEC_DIR = DATA_DIR / "potec"
POTEC_EYETRACKING_DIR = POTEC_DIR / "eyetracking_data"

# german-commons (fine-tuning corpus)
COMMONS_DIR = DATA_DIR / "commons_scientific"
COMMONS_HF_REPO = "coral-nlp/german-commons"
COMMONS_HF_CONFIG = "scientific"
# OpenAlex is OCR'd; keep only its high-quality docs. ocr_score is 0-100.
OPENALEX_SOURCE = "openalex"
OPENALEX_MIN_OCR = 90.0

# domain_preprocessing outputs
DOMAIN_PHY_DIR = DATA_DIR / "domain_phy"
DOMAIN_BIO_DIR = DATA_DIR / "domain_bio"
# Neutral control pool: german-commons docs OUTSIDE the pass-1 candidate set
# (both domain TF-IDF sims below threshold) — same corpus and register as the
# domain pools, no physics/biology content. See
# acquire.domain_preprocessing.select_other.
DOMAIN_OTHER_DIR = DATA_DIR / "domain_other"

# PoTeC sub-paths used by the analysis pipeline
POTEC_WORD_FEATURES_DIR = POTEC_DIR / "stimuli" / "word_features"
POTEC_READING_MEASURES_DIR = POTEC_EYETRACKING_DIR / "reading_measures_merged"

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
# Effective batch = batch_size × grad_accum. Both models MUST land on the same
# effective batch (8 -> 4096 tokens/step at block_size=512) so a checkpoint step
# means the same number of training tokens for every model.
DAPT_GRAD_ACCUM = {
    "llammlein-1b": 4,  # 2 × 4 = 8 effective — matches german-gpt2's 8 × 1
}
# Per-model DAPT learning rate: the 1B model gets a smaller LR than the 124M one.
DAPT_LEARNING_RATE = {
    "german-gpt2": 2e-4,
    "llammlein-1b": 1e-4,
}
# Training seeds; per-word surprisal is averaged over the seeds' checkpoints
# (Škrjanec & Demberg average 3 random seeds per method × domain × step). Each
# extra seed is a FULL extra DAPT run per domain — (0, 1, 2) = 3× training. Use a
# single seed while iterating; widen to (0, 1, 2) for the final seed-averaged run.
DAPT_SEEDS = (0,)

# Run artifacts (relative -> resolved against cwd at run time).
ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR
LABEL_IDS_PATH = ARTIFACTS_DIR / "domain_label_ids.json"
# Wide per-word surprisal cache: computed once, reloaded to skip model forwards.
# Delete the file (and the DAPT run dirs) by hand to force a recompute.
SURPRISAL_CACHE_PATH = ARTIFACTS_DIR / "surprisal.csv"

# Per-word join key for every word-level table.
WORD_KEY = ["text_id", "word_index_in_text"]

# Prompted-surprisal arm: a PRIOR-READING passage (not a persona instruction —
# base LMs don't obey personas) joined to the stimulus by the native document
# boundary in surprisal.score_words. The reader's domain lives in the context the
# same way it lives in the DAPT weights: physics prior for physicists, applied to
# any text (mixed per reader in model_comparison._prep_models). All three prior
# conditions are sampled from german-commons by the SAME code path
# (features.priors): physics/biology from the held-out DAPT split, neutral from
# the off-domain pool (DOMAIN_OTHER_DIR) — scientific register, no physics or
# biology content. Register-matched, so the domain-vs-neutral contrast isolates
# domain content rather than "sounds like science".
PRIOR_MAX_TOKENS = 64  # context budget shared by every prompt condition
# how many leading sentences of a held-out domain doc to use as one domain prior
# (token-truncated to PRIOR_MAX_TOKENS at scoring time).
PRIOR_PASSAGE_SENTENCES = 4
# Prompts averaged per condition. Each condition's surprisal is the mean per-word
# surprisal across this many DISTINCT priors (K distinct held-out docs per domain,
# K distinct neutral passages) so no single idiosyncratic passage drives the
# estimate. 5 sits at the variance-reduction elbow (1/sqrt(K) shrinks fast to
# K≈5, then flattens); raise for the final run if a K-sensitivity check still
# drifts. The off-domain pool (DOMAIN_OTHER_DIR) needs >= this many docs.
N_PRIOR_PASSAGES = 5
