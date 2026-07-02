"""Pipeline driver — full end-to-end run.

Per LM: DAPT over ``config.DAPT_SEEDS`` (per-word surprisal averaged across
seeds, as in Škrjanec & Demberg), then the mixed-model sweep per reading
measure (early FPRT + late GP/TFT). baseline reports the standard LRT vs the
no-surprisal null; the rest are residual-split relative to base surprisal.
Afterwards: adaptation diagnostics (perplexity + terminology-surprisal
trajectories) and the claim-level tests (best-checkpoint single LRT + direct
cross-LM comparisons).

    python -m src.main
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.analysis import adaptation_diagnostics as diag
from src.analysis import claim_tests as ct
from src.analysis import model_comparison as mc
from src.config import WORD_KEY
from src.features import data
from src.features import reading_time as rt
from src.features import surprisal as su
from src.modeling import finetune as ft

# early / late reading measures (paper: FPRT early; GP == PoTeC's RPD_inc and
# TFT late — the effect is expected on the late ones).
MEASURES = ("FPRT", "RPD_inc", "TFT")
# changing the schedule invalidates cached runs — bump config.DAPT_RUN_VERSION
DAPT_MAX_STEPS = 4_096
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096]
SLIM_MODELS = ("baseline", "prompted", "aligned")
# every checkpoint: indices 1..N pair to DAPT_CHECKPOINT_STEPS (0 = baseline).
SLIM_INDICES = list(range(1, len(DAPT_CHECKPOINT_STEPS) + 1))
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_model_surprisal(slug: str, name: str, words) -> dict:
    """Phase 1 — all surprisal for one model. No linear-model fitting here."""
    print(f"\n=== surprisal: {slug} ({name}) ===")
    prefix = config.MODEL_PREFIX[slug]

    print("Step 2 — load LM")
    model, tok = su.load_causal_lm(name)

    # Discipline-matched prompted baseline (physics / biology)
    s_pp = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["physics"]
    ).rename(columns={"surprisal": "s_prompt_phys"})
    s_pb = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["biology"]
    ).rename(columns={"surprisal": "s_prompt_bio"})
    prompt_surp = s_pp.merge(s_pb, on=["text_id", "word_index_in_text"])

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

    # Model comparison — baseline: standard LRT vs null; others: split signal,
    # base LM surprisal + each model's residual block (D = S_model - S_base,
    # with expertise × terminology interactions). All checkpoints.
    cmp, reader_ll = mc.model_comparison_over_epochs(
        bundle["surp_versions"],
        rm,
        bundle["prompt_surp"],
        measure=measure,
        models=SLIM_MODELS,
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
        bundles = []
        for slug, name in config.MODELS.items():
            b = compute_model_surprisal(slug, name, words)
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
    results_path = PROJECT_ROOT / "results_slim.csv"
    results.to_csv(results_path, index=False)
    print(f"\n  wrote {results_path.relative_to(PROJECT_ROOT)}")

    # Step 6 — claim tests: single LRT at the ΔLL-selected checkpoint per LM ×
    # measure, plus direct cross-LM comparisons (slope-difference z + paired
    # per-reader ΔLL). A cross-LM difference claim needs these, not two separate
    # significance verdicts.
    reader_ll = pd.concat(all_rll, ignore_index=True)
    best, cross = ct.run_claim_tests(results, reader_ll)
    best_path = PROJECT_ROOT / "results_best.csv"
    cross_path = PROJECT_ROOT / "results_cross_lm.csv"
    best.to_csv(best_path, index=False)
    cross.to_csv(cross_path, index=False)
    print("\nStep 6 — claim tests")
    print(best.to_string(index=False))
    print(cross.to_string(index=False))
    print(f"  wrote {best_path.name}, {cross_path.name}")


if __name__ == "__main__":
    main()
