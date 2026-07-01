"""Slim pipeline driver — fast end-to-end test only.

Runs only the 1024-step DAPT checkpoint, baseline and basic-prompted models.
Skips field×level prompts, physics/biology/aligned comparisons.
Use for pipeline validation; run full src/main.py for results.

    python -m src.main_slim
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


def run_model(slug: str, name: str, words, rm) -> dict:
    print(f"\n=== model: {slug} ({name}) ===")

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

    # Step 5 — correlation
    print("Step 5 — analysis")
    merged = co.merge_surprisal_rt(surp, rm)
    fit = co.regress_rt(merged, participants="all")
    print(f"  slope={fit.params['surprisal']:.2f} ms/bit  R2={fit.rsquared:.3f}")

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

    # Model comparison — baseline + prompted only, index 5 (step 1024) only
    print("Step 5 — model comparison (paper_full, slim models, step 1024)")
    cmp = mc.model_comparison_over_epochs(
        surp_versions, rm, prompt_surp, measure=MEASURE,
        spec="paper_full", models=SLIM_MODELS, indices=SLIM_INDICES,
    )
    cmp["spec"] = "paper_full"
    print(cmp.to_string(index=False))

    cmp.insert(0, "model_lm", slug)
    cmp.to_csv(PROJECT_ROOT / f"results_slim_{slug}.csv", index=False)

    return {
        "summary": {
            "model": slug,
            "slope_ms_per_bit": float(fit.params["surprisal"]),
            "r2": float(fit.rsquared),
        },
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

    summaries = []
    for slug, name in config.MODELS.items():
        out = run_model(slug, name, words, rm)
        summaries.append(out["summary"])

    summary_df = pd.DataFrame(summaries)
    print("\n=== slim summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
