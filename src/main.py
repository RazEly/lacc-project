"""Pipeline driver — fast end-to-end run.

Runs only the 1024-step DAPT checkpoint, baseline and basic-prompted models.

    python -m src.main
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.analysis import model_comparison as mc
from src.config import WORD_KEY
from src.features import data
from src.features import reading_time as rt
from src.features import surprisal as su
from src.modeling import finetune as ft

MEASURE = "TFT"
DAPT_LR = 2e-4
# must match the stored run_signature.json exactly to get a cache hit
DAPT_MAX_STEPS = 4_096
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096]
SLIM_MODELS = ("baseline", "prompted", "aligned")
# index 5 = step 1024 in the 6-checkpoint schedule above
SLIM_INDICES = [5]
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

    # Step 4 — load cached DAPT checkpoints (no retraining)
    print("Step 4 — DAPT checkpoints (loading from cache)")

    batch_size = config.DAPT_BATCH_SIZE.get(slug, 8)
    grad_accum = config.DAPT_GRAD_ACCUM.get(slug, 1)
    manifest = pd.concat(
        [
            ft.finetune_dapt(
                "physics",
                base_model=name,
                max_steps=DAPT_MAX_STEPS,
                checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                batch_size=batch_size,
                grad_accum=grad_accum,
                learning_rate=DAPT_LR,
            ),
            ft.finetune_dapt(
                "biology",
                base_model=name,
                max_steps=DAPT_MAX_STEPS,
                checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                batch_size=batch_size,
                grad_accum=grad_accum,
                learning_rate=DAPT_LR,
            ),
        ],
        ignore_index=True,
    )

    surp_versions = ft.recompute_surprisal_over_checkpoints(words, manifest)

    # Wide surprisal columns for this model.
    wide = su.build_wide(prefix, surp_versions, prompt_surp)
    return {
        "slug": slug,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "wide": wide,
    }


def fit_model(bundle: dict, rm) -> pd.DataFrame:
    """Phase 2 — mixed-model comparison for one model's precomputed surprisal."""
    slug = bundle["slug"]
    prompt_surp = bundle["prompt_surp"]
    surp_versions = bundle["surp_versions"]
    print(f"\n=== fit: {slug} ===")

    # Model comparison — split signal: base LM surprisal + each model's residual
    # (D = S_model - S_base). p_surprisal tests the residual; delta_ll = gain over
    # base surprisal. baseline auto-skipped (D == 0). Step 1024 only.
    print("Step 5 — model comparison (residual split, slim models, step 1024)")
    cmp = mc.model_comparison_over_epochs(
        surp_versions,
        rm,
        prompt_surp,
        measure=MEASURE,
        models=SLIM_MODELS,
        indices=SLIM_INDICES,
        residual=True,
    )
    cmp.insert(0, "model_lm", slug)
    print(cmp.to_string(index=False))
    return cmp


def bundle_from_cache(cache: pd.DataFrame, slug: str) -> dict:
    """Rebuild a fit bundle straight from the wide cache — no model load, no finetune.

    Pulls only what ``fit_model`` needs for the slim run: baseline surprisal, the
    prompted columns, and the slim checkpoint(s) for both domains. ``epoch`` in
    surp_versions is the checkpoint step (display only).
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

    return {"slug": slug, "prompt_surp": prompt_surp, "surp_versions": surp_versions}


def main() -> None:
    print("Step 1 — reading time")
    rm_raw = data.load_reading_measures()
    rm = rt.clean_reading_times(rm_raw, MEASURE)
    print(f"  raw_rows={len(rm_raw)}  cleaned={len(rm)} ({len(rm) / len(rm_raw):.1%})")

    # Phase 1 — surprisal. If the wide cache exists, reload it straight from CSV
    # (no model load, no finetune) and go to fitting; trust its contents. Otherwise
    # compute + cache each model.
    cache = su.load_cache()
    if cache is not None:
        print(f"Surprisal cache hit ({config.SURPRISAL_CACHE_PATH}) — skipping compute")
        bundles = [bundle_from_cache(cache, slug) for slug in config.MODELS]
    else:
        words = data.load_word_features()
        print(f"  words={len(words)}")
        bundles = []
        for slug, name in config.MODELS.items():
            b = compute_model_surprisal(slug, name, words)
            bundles.append(b)
            cache = su.merge_model(cache, b["wide"])
            su.save_cache(cache)
            print(f"  wrote surprisal cache -> {config.SURPRISAL_CACHE_PATH}")

    # Phase 2 — fit mixed models now that all surprisal exists.
    cmps = [fit_model(b, rm) for b in bundles]

    # Both models' comparisons in one CSV (model_lm column distinguishes them).
    all_cmp = pd.concat(cmps, ignore_index=True)
    results_path = PROJECT_ROOT / "results_slim.csv"
    all_cmp.to_csv(results_path, index=False)
    print(f"  wrote {results_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
