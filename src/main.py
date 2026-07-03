"""Pipeline driver — full end-to-end run.

Per LM: DAPT over ``config.DAPT_SEEDS`` (per-word surprisal averaged across
seeds, as in Škrjanec & Demberg), then the mixed-model sweep for the reading-time
measure (TFT). Every surprisal source (baseline, aligned, prompted, …) is scored
by ΔLL over the SAME shared no-surprisal baseline (paper Eq. 2/4/5); ``p_lrt`` is
the nested 1-df LRT for adding that source's surprisal.
Afterwards: adaptation diagnostics (perplexity + terminology-surprisal
trajectories) and the claim-level tests (best-checkpoint single LRT + direct
cross-LM comparisons).

    python -m src.main
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis import adaptation_diagnostics as diag
from src.analysis import claim_tests as ct
from src.analysis import model_comparison as mc
from src.config import WORD_KEY
from src.features import data
from src.features import priors as pr
from src.features import reading_time as rt
from src.features import surprisal as su
from src.modeling import finetune as ft

# reading-time measure (TFT); the effect is expected on the late measure.
MEASURES = ("TFT",)
# delete the artifacts/ run dirs + surprisal cache by hand to force a retrain
DAPT_MAX_STEPS = 4_096
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096]
SLIM_MODELS = ("baseline", "prompted", "prompt_neutral", "aligned")
# every checkpoint: indices 1..N pair to DAPT_CHECKPOINT_STEPS (0 = baseline).
SLIM_INDICES = list(range(1, len(DAPT_CHECKPOINT_STEPS) + 1))


def compute_model_surprisal(slug: str, name: str, words, priors: dict) -> dict:
    """Phase 1 — all surprisal for one model. No linear-model fitting here."""
    print(f"\n=== surprisal: {slug} ({name}) ===")
    prefix = config.MODEL_PREFIX[slug]

    print("Step 2 — load LM")
    model, tok = su.load_causal_lm(name)

    # Prompted arm: prior-reading passage + native document boundary, truncated to
    # a shared context budget. physics/biology = domain priors (reader-aligned per
    # reader downstream); neutral = length-matched non-domain control. Each arg is
    # a LIST of N_PRIOR_PASSAGES priors — compute_surprisal averages per-word
    # surprisal across them, so the stored column is the mean over priors.
    def _prior_surp(prior_texts: list[str], col: str) -> pd.DataFrame:
        return su.compute_surprisal(
            words, model, tok,
            prompt=prior_texts, max_prompt_tokens=config.PRIOR_MAX_TOKENS,
        ).rename(columns={"surprisal": col})

    prompt_surp = (
        _prior_surp(priors["physics"], "s_prompt_phys")
        .merge(_prior_surp(priors["biology"], "s_prompt_bio"), on=WORD_KEY)
        .merge(_prior_surp(priors["neutral"], "s_prompt_neutral"), on=WORD_KEY)
    )

    # Step 4 — DAPT checkpoints, one run per domain × seed (cached / Hub-mirrored)
    print(f"Step 4 — DAPT checkpoints (seeds {config.DAPT_SEEDS})")

    batch_size = config.DAPT_BATCH_SIZE.get(slug, 8)
    grad_accum = config.DAPT_GRAD_ACCUM.get(slug, 1)
    learning_rate = config.DAPT_LEARNING_RATE.get(slug, 2e-4)
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
                seed=seed,
            )
            for domain in ("physics", "biology")
            for seed in config.DAPT_SEEDS
        ],
        ignore_index=True,
    )

    # baseline (index 0) weights are seed-independent: score once per domain.
    first_seed = config.DAPT_SEEDS[0]
    to_score = manifest[(manifest["index"] > 0) | (manifest["seed"] == first_seed)]
    per_seed = ft.recompute_surprisal_over_checkpoints(words, to_score)
    # per-word surprisal averaged over seeds (paper averages 3 seeds).
    surp_versions = per_seed.groupby(
        WORD_KEY + ["index", "domain"], as_index=False
    ).agg(surprisal=("surprisal", "mean"), epoch=("epoch", "first"))

    # Wide surprisal columns for this model.
    wide = su.build_wide(prefix, surp_versions, prompt_surp)
    return {
        "slug": slug,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "wide": wide,
        "manifest": manifest,
    }


def fit_model(bundle: dict, rm, measure: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 2 — mixed-model comparison for one model × measure.

    Returns ``(results, reader_ll)``, both tagged with ``model_lm`` / ``measure``.
    """
    slug = bundle["slug"]
    print(f"\n=== fit: {slug} / {measure} ===")

    # drop the neutral-prompt arm when the cache lacks it (older surprisal.csv).
    has_neutral = "s_prompt_neutral" in bundle["prompt_surp"].columns
    models = tuple(m for m in SLIM_MODELS if has_neutral or m != "prompt_neutral")

    # Model comparison — every source scored by ΔLL over the shared no-surprisal
    # baseline (paper Eq. 2/4/5): surprisal enters as a single plain main effect,
    # nested 1-df LRT. All checkpoints.
    cmp, reader_ll = mc.model_comparison_over_epochs(
        bundle["surp_versions"],
        rm,
        bundle["prompt_surp"],
        measure=measure,
        models=models,
        indices=SLIM_INDICES,
    )
    for df in (cmp, reader_ll):
        df.insert(0, "model_lm", slug)
        df.insert(1, "measure", measure)
    print(cmp.to_string(index=False))
    return cmp, reader_ll


def bundle_from_cache(cache: pd.DataFrame, slug: str) -> dict:
    """Rebuild a fit bundle straight from the wide cache — no model load, no finetune.

    Pulls only what ``fit_model`` needs for the slim run: baseline surprisal, the
    prompted columns, and the slim checkpoint(s) for both domains. ``epoch`` in
    surp_versions is the checkpoint step (display only). ``manifest`` is None —
    perplexity diagnostics need the training manifests.
    """
    prefix = config.MODEL_PREFIX[slug]
    print(f"\n=== reload from cache: {slug} ({prefix}) ===")

    def _col(name: str, new: str) -> pd.DataFrame:
        return (
            cache[WORD_KEY + [name]]
            .dropna(subset=[name])
            .rename(columns={name: new})
            .reset_index(drop=True)
        )

    prompt_surp = _col(f"{prefix}_prompt_phys", "s_prompt_phys")
    prompt_surp = prompt_surp.merge(
        _col(f"{prefix}_prompt_bio", "s_prompt_bio"), on=WORD_KEY
    )
    # neutral prior is optional: older caches predate it. Merge only if present.
    if f"{prefix}_prompt_neutral" in cache.columns:
        prompt_surp = prompt_surp.merge(
            _col(f"{prefix}_prompt_neutral", "s_prompt_neutral"), on=WORD_KEY
        )

    # Long per-checkpoint table for model_comparison: index 0 (baseline, shared by
    # both domains) + the slim checkpoint(s), each domain.
    frames = []
    for idx in [0, *SLIM_INDICES]:
        step = 0 if idx == 0 else DAPT_CHECKPOINT_STEPS[idx - 1]
        for domain, short in [("physics", "phys"), ("biology", "bio")]:
            col = f"{prefix}_0" if idx == 0 else f"{prefix}_{idx}_{short}"
            sv = _col(col, "surprisal")
            sv["index"], sv["domain"], sv["epoch"] = idx, domain, step
            frames.append(sv)
    surp_versions = pd.concat(frames, ignore_index=True)

    return {
        "slug": slug,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "manifest": None,
    }


def main() -> None:
    print("Step 1 — load PoTeC")
    rm_raw = data.load_reading_measures()
    words = data.load_word_features()
    print(f"  raw_rows={len(rm_raw)}  words={len(words)}")

    # Phase 1 — surprisal. If the wide cache exists, reload it straight from CSV
    # (no model load, no finetune) and go to fitting; trust its contents. Otherwise
    # compute + cache each model.
    cache = su.load_cache()
    if cache is not None:
        print(f"Surprisal cache hit ({config.SURPRISAL_CACHE_PATH}) — skipping compute")
        bundles = [bundle_from_cache(cache, slug) for slug in config.MODELS]
    else:
        priors = pr.load_prior_passages()
        print(f"  priors: {config.N_PRIOR_PASSAGES}× physics/biology from held-out "
              f"german-commons + {config.N_PRIOR_PASSAGES}× neutral, averaged "
              f"per word (budget {config.PRIOR_MAX_TOKENS} tokens)")
        bundles = []
        for slug, name in config.MODELS.items():
            b = compute_model_surprisal(slug, name, words, priors)
            bundles.append(b)
            cache = su.merge_model(cache, b["wide"])
            su.save_cache(cache)
            print(f"  wrote surprisal cache -> {config.SURPRISAL_CACHE_PATH}")

    # Adaptation diagnostics — establish DAPT moved each model before reading
    # anything into the reading-time fits (perplexity needs a fresh manifest).
    print("\nStep 4b — adaptation diagnostics -> figures/")
    for b in bundles:
        diag.surprisal_trajectories(b["surp_versions"], words, b["slug"])
        if b.get("manifest") is not None:
            diag.perplexity_curves(b["manifest"], b["slug"])

    # Phase 2 — mixed models per measure (early + late) per LM.
    all_cmp, all_rll = [], []
    for measure in MEASURES:
        rm = rt.clean_reading_times(rm_raw, measure)
        print(f"\nStep 5 — {measure}: cleaned={len(rm)} ({len(rm) / len(rm_raw):.1%})")
        for b in bundles:
            cmp, reader_ll = fit_model(b, rm, measure)
            all_cmp.append(cmp)
            all_rll.append(reader_ll)

    results = pd.concat(all_cmp, ignore_index=True)
    results_path = config.PROJECT_ROOT / "results_slim.csv"
    results.to_csv(results_path, index=False)
    print(f"\n  wrote {results_path.name}")

    # Step 6 — claim tests: single LRT at the ΔLL-selected checkpoint per LM ×
    # measure, plus direct cross-LM comparisons (slope-difference z + paired
    # per-reader ΔLL). A cross-LM difference claim needs these, not two separate
    # significance verdicts.
    reader_ll = pd.concat(all_rll, ignore_index=True)
    best, cross = ct.run_claim_tests(results, reader_ll)
    best_path = config.PROJECT_ROOT / "results_best.csv"
    cross_path = config.PROJECT_ROOT / "results_cross_lm.csv"
    best.to_csv(best_path, index=False)
    cross.to_csv(cross_path, index=False)
    print("\nStep 6 — claim tests")
    print(best.to_string(index=False))
    print(cross.to_string(index=False))
    print(f"  wrote {best_path.name}, {cross_path.name}")


if __name__ == "__main__":
    main()
