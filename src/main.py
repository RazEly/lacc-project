"""Pipeline driver — fast end-to-end run.

Runs only the 1024-step DAPT checkpoint, baseline and basic-prompted models.
Skips field×level prompts, physics/biology/aligned comparisons.

    python -m src.main
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis import correlation as co
from src.config import WORD_KEY
from src.features import data
from src.analysis import model_comparison as mc
from src.features import reading_time as rt
from src.features import surprisal as su
from src.analysis import viz

MEASURE = "TFT"
DAPT_LR = 2e-4
# must match the stored run_signature.json exactly to get a cache hit
DAPT_MAX_STEPS = 4_096
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096]
SLIM_MODELS = ("baseline", "prompted", "aligned", "graded_aligned")
# index 5 = step 1024 in the 6-checkpoint schedule above
SLIM_INDICES = [5]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"


def save_fig(ax, name: str) -> None:
    fig = ax.get_figure()
    fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def compute_model_surprisal(slug: str, name: str, words) -> dict:
    """Phase 1 — all surprisal for one model. No linear-model fitting here."""
    print(f"\n=== surprisal: {slug} ({name}) ===")
    prefix = config.MODEL_PREFIX[slug]

    # Step 2 — baseline surprisal
    print("Step 2 — surprisal")
    model, tok = su.load_causal_lm(name)
    surp = su.compute_surprisal(words, model, tok)
    print(surp.head().to_string())

    # Basic prompted baseline only (no field×level variants)
    s_pp = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["physics"]
    ).rename(columns={"surprisal": "s_prompt_phys"})
    s_pb = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["biology"]
    ).rename(columns={"surprisal": "s_prompt_bio"})
    prompt_surp = s_pp.merge(s_pb, on=["text_id", "word_index_in_text"])

    # Step 4 — load cached DAPT checkpoints (no retraining)
    print("Step 4 — DAPT checkpoints (loading from cache)")
    from src.modeling import finetune as ft

    batch_size = config.DAPT_BATCH_SIZE.get(slug, 8)
    grad_accum = config.DAPT_GRAD_ACCUM.get(slug, 1)
    manifest = pd.concat(
        [
            ft.finetune_dapt("physics", base_model=name, max_steps=DAPT_MAX_STEPS,
                             checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                             batch_size=batch_size, grad_accum=grad_accum,
                             learning_rate=DAPT_LR),
            ft.finetune_dapt("biology", base_model=name, max_steps=DAPT_MAX_STEPS,
                             checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                             batch_size=batch_size, grad_accum=grad_accum,
                             learning_rate=DAPT_LR),
        ],
        ignore_index=True,
    )

    surp_versions = ft.recompute_surprisal_over_checkpoints(words, manifest)

    # Pull ck1 (index 1, step 4) and ck2 (index 2, step 16) for both domains
    # and attach to prompt_surp so build_index_df carries them into _prep_models.
    def _ck_surp(idx, domain, col):
        sel = surp_versions[(surp_versions["index"] == idx) & (surp_versions["domain"] == domain)]
        return sel[WORD_KEY + ["surprisal"]].rename(columns={"surprisal": col})

    for df_ck in [
        _ck_surp(1, "physics", "s_phys_ck1"),
        _ck_surp(2, "physics", "s_phys_ck2"),
        _ck_surp(1, "biology", "s_bio_ck1"),
        _ck_surp(2, "biology", "s_bio_ck2"),
    ]:
        prompt_surp = prompt_surp.merge(df_ck, on=WORD_KEY)

    # Wide surprisal columns for this model (build_wide keeps only schema prompt
    # columns, so the ck scratch cols merged into prompt_surp above are dropped).
    wide = su.build_wide(prefix, surp_versions, prompt_surp)
    return {
        "slug": slug,
        "surp": surp,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "wide": wide,
    }


def fit_model(bundle: dict, rm) -> dict:
    """Phase 2 — fit linear models for one model's precomputed surprisal."""
    slug = bundle["slug"]
    surp = bundle["surp"]
    prompt_surp = bundle["prompt_surp"]
    surp_versions = bundle["surp_versions"]
    print(f"\n=== fit: {slug} ===")

    # Step 5 — correlation
    print("Step 5 — analysis")
    merged = co.merge_surprisal_rt(surp, rm)
    fit = co.regress_rt(merged, participants="all")
    print(f"  slope={fit.params['surprisal']:.2f} ms/bit  R2={fit.rsquared:.3f}")

    # Model comparison — split signal: base LM surprisal + each model's residual
    # (D = S_model - S_base). p_surprisal tests the residual; delta_ll = gain over
    # base surprisal. baseline auto-skipped (D == 0). Step 1024 only.
    print("Step 5 — model comparison (paper_full, residual split, slim models, step 1024)")
    cmp = mc.model_comparison_over_epochs(
        surp_versions, rm, prompt_surp, measure=MEASURE,
        spec="paper_full", models=SLIM_MODELS, indices=SLIM_INDICES,
        residual=True,
    )
    cmp["spec"] = "paper_full"
    cmp.insert(0, "model_lm", slug)
    print(cmp.to_string(index=False))

    return {
        "summary": {
            "model": slug,
            "slope_ms_per_bit": float(fit.params["surprisal"]),
            "r2": float(fit.rsquared),
        },
        "cmp": cmp,
    }


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("Step 1 — reading time")
    words = data.load_word_features()
    rm_raw = data.load_reading_measures()
    rm = rt.clean_reading_times(rm_raw, MEASURE)
    print(
        f"  words={len(words)}  raw_rows={len(rm_raw)}  "
        f"cleaned={len(rm)} ({len(rm) / len(rm_raw):.1%})"
    )

    # Phase 1 — compute + cache surprisal for ALL models before any fitting.
    bundles = []
    cache = su.load_cache()
    for slug, name in config.MODELS.items():
        b = compute_model_surprisal(slug, name, words)
        bundles.append(b)
        cache = su.merge_model(cache, b["wide"])
        su.save_cache(cache)
        print(f"  wrote surprisal cache -> {config.SURPRISAL_CACHE_PATH}")

    # Phase 2 — fit linear models now that all surprisal exists.
    summaries, cmps = [], []
    for b in bundles:
        out = fit_model(b, rm)
        summaries.append(out["summary"])
        cmps.append(out["cmp"])

    # Both models' comparisons in one CSV (model_lm column distinguishes them).
    all_cmp = pd.concat(cmps, ignore_index=True)
    results_path = PROJECT_ROOT / "results_slim.csv"
    all_cmp.to_csv(results_path, index=False)
    print(f"  wrote {results_path.relative_to(PROJECT_ROOT)}")

    summary_df = pd.DataFrame(summaries)
    print("\n=== slim summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
