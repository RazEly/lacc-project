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

# domain_preprocessing outputs
DOMAIN_PHY_DIR = DATA_DIR / "domain_phy"
DOMAIN_BIO_DIR = DATA_DIR / "domain_bio"

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
# any text (mixed per reader in model_comparison._prep_models). Domain priors are
# pulled from the held-out german-commons split (features.priors); the neutral
# prior is a length-matched non-domain control.
PRIOR_MAX_TOKENS = 64  # context budget shared by every prompt condition
# how many leading sentences of a held-out domain doc to use as the domain prior
# (token-truncated to PRIOR_MAX_TOKENS at scoring time).
PRIOR_PASSAGE_SENTENCES = 4
# Non-domain control prior: generic German prose, no physics/biology content.
# Long enough to exceed PRIOR_MAX_TOKENS so the neutral condition matches length.
NEUTRAL_PRIOR = (
    "Am Wochenende gehen viele Menschen gern spazieren oder treffen sich mit "
    "Freunden. Manche kochen zusammen, andere sehen einen Film oder hören Musik. "
    "In der Stadt gibt es Cafés, Parks und Märkte, auf denen man den Nachmittag "
    "verbringen kann. Das Wetter spielt dabei oft eine große Rolle, denn bei "
    "Sonnenschein zieht es die Leute nach draußen. Am Abend kehren die meisten "
    "wieder nach Hause zurück und lassen den Tag ruhig ausklingen."
)
