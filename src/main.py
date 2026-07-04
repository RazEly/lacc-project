"""Pipeline driver — full end-to-end run.

Per LM: one DAPT run per domain, then the mixed-model sweep for the reading-time
measure (TFT). Every surprisal source (baseline, aligned, prompted, …) enters as
a single plain main effect and is scored against the SAME shared no-surprisal
baseline (paper Eq. 2/4/5): LL, ΔLL, χ², LRT p, AIC. A second per-LM table
reports Vuong tests of the baseline-surprisal model against each
reader-conditioned arm (aligned per checkpoint, prompted, neutral prompt).
Along the way: figures (viz.md) — perplexity grid, example-sentence surprisal,
mean surprisal over steps, ΔLL curves, RT–surprisal.

    python -m src.main
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis import model_comparison as mc
from src.analysis import viz
from src.config import WORD_KEY
from src.features import potec
from src.features import priors as pr
from src.features import reading_time as rt
from src.features import surprisal as su
from src.features import surprisal_cache as sc
from src.modeling import finetune as ft
from src.modeling import lm

# reading-time measure (TFT); the effect is expected on the late measure.
MEASURES = ("TFT",)
# delete the artifacts/ run dirs + surprisal cache by hand to force a retrain
DAPT_MAX_STEPS = 4_096
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096]
SLIM_MODELS = ("baseline", "prompted", "prompt_neutral", "aligned")
# every checkpoint: indices 1..N pair to DAPT_CHECKPOINT_STEPS (0 = baseline).
SLIM_INDICES = list(range(1, len(DAPT_CHECKPOINT_STEPS) + 1))

# Decoder LMs the pipeline runs over (slug -> HF repo), then compares. Both German:
# german-gpt2 (124M) + LLäMmlein 1B (German-only, from scratch). Pulled at run time.
MODELS = {
    "german-gpt2": "dbmdz/german-gpt2",
    "llammlein-1b": "LSX-UniWue/LLaMmlein_1B",
}
# Short column prefix per model for the wide surprisal cache (surprisal.csv):
# <prefix>_0 baseline, <prefix>_<i>_<domain> checkpoints, <prefix>_prompt_* prompted.
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
# Context budget (tokens) shared by every prompt condition.
PRIOR_MAX_TOKENS = 128

# Example sentence for the fig-2 surprisal walk-through: (text_id, sent_index).
# Baseline GPT-2 surprisal + Δ over checkpoints/prompts on one PoTeC sentence.
EXAMPLE_SENTENCE = ("b0", 1)
# The LM whose baseline drives the single-model figures (fig 2 + fig 5).
FIGURE_LM = "german-gpt2"


def compute_model_surprisal(slug: str, name: str, words, priors: dict) -> dict:
    """Phase 1 — all surprisal for one model. No linear-model fitting here."""
    print(f"\n=== surprisal: {slug} ({name}) ===")
    prefix = MODEL_PREFIX[slug]

    print("Step 2 — load LM")
    model, tok = lm.load_causal_lm(name)

    # Prompted arm: prior-reading passage + native document boundary, truncated to
    # a shared context budget. physics/biology = domain priors (reader-aligned per
    # reader downstream); neutral = off-domain scientific control (same corpus,
    # same register, no domain content). Each arg is a LIST of N_PRIOR_PASSAGES
    # priors — compute_surprisal averages per-word surprisal across them, so the
    # stored column is the mean over priors.
    def _prior_surp(prior_texts: list[str], col: str) -> pd.DataFrame:
        return su.compute_surprisal(
            words,
            model,
            tok,
            prompt=prior_texts,
            max_prompt_tokens=PRIOR_MAX_TOKENS,
        ).rename(columns={"surprisal": col})

    prompt_surp = (
        _prior_surp(priors["physics"], "s_prompt_physics")
        .merge(_prior_surp(priors["biology"], "s_prompt_biology"), on=WORD_KEY)
        .merge(_prior_surp(priors["neutral"], "s_prompt_neutral"), on=WORD_KEY)
    )

    # Step 4 — DAPT checkpoints, one run per domain (cached / Hub-mirrored)
    print("Step 4 — DAPT checkpoints")

    batch_size = DAPT_BATCH_SIZE.get(slug, 8)
    grad_accum = DAPT_GRAD_ACCUM.get(slug, 1)
    learning_rate = DAPT_LEARNING_RATE.get(slug, 2e-4)
    manifest = pd.concat(
        [
            ft.finetune_dapt(
                domain,
                base_model=name,
                max_steps=DAPT_MAX_STEPS,
                checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                batch_size=batch_size,
                grad_accum=grad_accum,
                learning_rate=learning_rate,
            )
            for domain in config.DOMAINS
        ],
        ignore_index=True,
    )

    surp_versions = su.recompute_surprisal_over_checkpoints(words, manifest).drop(
        columns=["checkpoint"]
    )

    # Wide surprisal columns for this model.
    wide = sc.build_wide(prefix, surp_versions, prompt_surp)
    return {
        "slug": slug,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "wide": wide,
        "manifest": manifest,
    }


def fit_model(
    bundle: dict, rm, measure: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phase 2 — mixed-model comparison for one model × measure.

    Returns ``(results, vuong, reader_ll)``, all tagged with ``model_lm`` /
    ``measure``.
    """
    slug = bundle["slug"]
    print(f"\n=== fit: {slug} / {measure} ===")

    # drop the neutral-prompt arm when the cache lacks it (older surprisal.csv).
    has_neutral = "s_prompt_neutral" in bundle["prompt_surp"].columns
    models = tuple(m for m in SLIM_MODELS if has_neutral or m != "prompt_neutral")

    # Model comparison — every source scored against the shared no-surprisal
    # baseline (LRT + AIC), plus Vuong tests of the baseline-surprisal model
    # vs the reader-conditioned arms. All checkpoints.
    cmp, vuong, reader_ll = mc.model_comparison_over_epochs(
        bundle["surp_versions"],
        rm,
        bundle["prompt_surp"],
        measure=measure,
        models=models,
        indices=SLIM_INDICES,
    )
    for df in (cmp, vuong, reader_ll):
        df.insert(0, "model_lm", slug)
        df.insert(1, "measure", measure)
    print(cmp.to_string(index=False))
    print("\nVuong — baseline+surprisal vs reader-conditioned arms:")
    print(vuong.to_string(index=False))
    return cmp, vuong, reader_ll


def main() -> None:
    print("Step 1 — load PoTeC")
    rm_raw = potec.load_reading_measures()
    words = potec.load_word_features()
    print(f"  raw_rows={len(rm_raw)}  words={len(words)}")

    # Phase 1 — surprisal. If the wide cache exists, reload it straight from CSV
    # (no model load, no finetune) and go to fitting; trust its contents. Otherwise
    # compute + cache each model.
    cache = sc.load_cache()
    if cache is not None:
        print(f"Surprisal cache hit ({config.SURPRISAL_CACHE_PATH}) — skipping compute")
        bundles = [
            sc.bundle_from_cache(
                cache, slug, MODEL_PREFIX[slug], SLIM_INDICES, DAPT_CHECKPOINT_STEPS
            )
            for slug in MODELS
        ]
    else:
        priors = pr.load_prior_passages()
        print(
            f"  priors: {config.N_PRIOR_PASSAGES}× physics/biology from held-out "
            f"german-commons + {config.N_PRIOR_PASSAGES}× neutral (off-domain "
            f"pool), averaged per word (budget {PRIOR_MAX_TOKENS} tokens)"
        )
        bundles = []
        for slug, name in MODELS.items():
            b = compute_model_surprisal(slug, name, words, priors)
            bundles.append(b)
            cache = sc.merge_model(cache, b["wide"])
            sc.save_cache(cache)
            print(f"  wrote surprisal cache -> {config.SURPRISAL_CACHE_PATH}")

    # Surprisal-only figures — perplexity, example sentence, mean surprisal.
    # Perplexity needs the training manifests (None on a cache reload).
    print("\nStep 4b — surprisal figures -> figures/")
    manifests = [
        b["manifest"].assign(model=b["slug"])
        for b in bundles
        if b.get("manifest") is not None
    ]
    if manifests:
        # NOTE: the DAPT manifest evals each run on its OWN held-out split only,
        # so eval_domain == ft_domain (one line per panel). Cross-domain eval
        # (both test sets per panel, per viz.md fig 1) needs a finetune change.
        m = pd.concat(manifests, ignore_index=True).rename(
            columns={"domain": "ft_domain"}
        )
        m["eval_domain"] = m["ft_domain"]
        viz.perplexity_grid(m)

    for b in bundles:
        viz.mean_surprisal_over_steps(
            b["surp_versions"], b["prompt_surp"], words, b["slug"]
        )

    fig_lm = next((b for b in bundles if b["slug"] == FIGURE_LM), bundles[0])
    text_id, sent_index = EXAMPLE_SENTENCE
    viz.sentence_surprisal_example(
        fig_lm["surp_versions"],
        fig_lm["prompt_surp"],
        words,
        text_id,
        sent_index,
        slug=fig_lm["slug"],
    )

    # Phase 2 — mixed models per measure (early + late) per LM.
    all_cmp, all_vuong = [], []
    rm_by_measure = {}
    for measure in MEASURES:
        rm = rt.clean_reading_times(rm_raw, measure)
        rm_by_measure[measure] = rm
        print(f"\nStep 5 — {measure}: cleaned={len(rm)} ({len(rm) / len(rm_raw):.1%})")
        for b in bundles:
            cmp, vuong, _ = fit_model(b, rm, measure)
            all_cmp.append(cmp)
            all_vuong.append(vuong)

    results = pd.concat(all_cmp, ignore_index=True)
    results_path = config.PROJECT_ROOT / "results_slim.csv"
    results.to_csv(results_path, index=False)
    vuong = pd.concat(all_vuong, ignore_index=True)
    vuong_path = config.PROJECT_ROOT / "results_vuong.csv"
    vuong.to_csv(vuong_path, index=False)
    print(f"\n  wrote {results_path.name}, {vuong_path.name}")

    # Fit-dependent figures — ΔLL curves (all LMs) + RT vs baseline surprisal
    # (baseline of FIGURE_LM, first measure only).
    print("\nStep 5b — fit figures -> figures/")
    viz.delta_ll_curves(results)
    measure = MEASURES[0]
    baseline_surp = fig_lm["surp_versions"].query("index == 0")
    viz.rt_vs_surprisal(
        rm_by_measure[measure], baseline_surp, measure=measure, slug=fig_lm["slug"]
    )


if __name__ == "__main__":
    main()
