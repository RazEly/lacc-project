"""Driver for the PoTeC decoder-LM pipeline (steps 1-6).

Reading-time cleaning, surprisal, the surprisal-vs-RT analysis, DAPT fine-tuning,
and the reader-aligned model comparison. Steps 2-6 run per model in
``config.MODELS`` (``<slug>_*`` figures + ``results_<slug>.csv``), then a
cross-model block compares them. Weights pull from the Hub on first use; step 4
DAPT runs end to end (GPU recommended). Run from the project root:

    python -m src.main
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis import correlation as co
from src.features import data
from src.analysis import model_comparison as mc
from src.features import reading_time as rt
from src.features import surprisal as su
from src.analysis import viz

MEASURE = "TFT"  # total fixation time == TRT
DAPT_LR = 2e-4  # LoRA needs a high LR (zero-init, few rank-r params)
# Škrjanec et al. (papers/07) schedule: ≤16,384 steps, checkpoints at 4ⁿ steps.
DAPT_MAX_STEPS = 16_384
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096, 16384]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"


def save_fig(ax, name: str) -> None:
    """Save the figure owning ``ax`` to ``figures/<name>.png`` and close it."""
    fig = ax.get_figure()
    fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def run_model(slug: str, name: str, words, rm) -> dict:
    """Run steps 2-6 for one model; write ``<slug>_*`` figures + csv; return a summary.

    The summary row feeds the cross-model comparison: the all-readers surprisal-RT
    correlation, the regression slope, and the best reader-aligned ΔLL.
    """
    print(f"\n=== model: {slug} ({name}) ===")

    # ── Step 2 — model surprisal (baseline) ──────────────────────────────────
    print("Step 2 — surprisal")
    model, tok = su.load_causal_lm(name)
    surp = su.compute_surprisal(words, model, tok)  # prompt=None
    print(surp.head().to_string())

    # prompted-baseline surprisal: un-adapted model + discipline-matched prompt
    # (prompting analogue of fine-tuning), one column per discipline.
    s_pp = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["physics"]
    ).rename(columns={"surprisal": "s_prompt_phys"})
    s_pb = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["biology"]
    ).rename(columns={"surprisal": "s_prompt_bio"})
    prompt_surp = s_pp.merge(s_pb, on=["text_id", "word_index_in_text"])

    # field × level prompted baseline: prompt matches discipline AND study level,
    # one column per (level, discipline) cell (ug = undergrad, grad = graduate).
    _fl_cols = {
        (0, 1): "s_prompt_phys_ug",
        (1, 1): "s_prompt_phys_grad",
        (0, 0): "s_prompt_bio_ug",
        (1, 0): "s_prompt_bio_grad",
    }
    for key, col in _fl_cols.items():
        prompt_surp = prompt_surp.merge(
            su.compute_surprisal(
                words, model, tok, prompt=config.FIELD_LEVEL_PROMPTS[key]
            ).rename(columns={"surprisal": col}),
            on=["text_id", "word_index_in_text"],
        )

    # ── Step 5 — analysis ────────────────────────────────────────────────────
    print("Step 5 — analysis")
    merged = co.merge_surprisal_rt(surp, rm)
    rows = []
    for grp in ["all", "experts", "novices"]:
        for dom_only in (False, True):
            r = co.correlate_surprisal(
                merged, participants=grp, domain_only=dom_only, mode="mean"
            )
            rows.append({"group": grp, "domain_only": dom_only, **r})
    corr_df = pd.DataFrame(rows)[
        ["group", "domain_only", "n", "pearson", "spearman"]
    ]
    print(corr_df.to_string(index=False))

    fit = co.regress_rt(merged, participants="all")
    print(
        f"  slope={fit.params['surprisal']:.2f} ms/bit  R2={fit.rsquared:.3f}"
    )

    # ── Figures (per model) ──────────────────────────────────────────────────
    print("Figures")
    # surprisal vs reading time (binned scatter + OLS fit)
    agg_words = co._aggregate_words(
        co._filter_participants(merged, "all"), MEASURE, "mean"
    )
    fig, ax = plt.subplots()
    viz.surprisal_scatter(agg_words, ax=ax)
    save_fig(ax, f"{slug}_surprisal_scatter")

    # ── Step 4 — DAPT fine-tuning (GPU recommended) ──────────────────────────
    print("Step 4 — DAPT fine-tuning")
    from src.modeling import finetune as ft

    # Fine-tune both domains (the comparison needs both + the shared baseline) on
    # the same fixed-step schedule, so checkpoint indices line up across domains.
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
    fig, ax = plt.subplots()
    viz.perplexity_curve(manifest, ax=ax)
    save_fig(ax, f"{slug}_perplexity_curve")

    surp_versions = ft.recompute_surprisal_over_checkpoints(words, manifest)
    curve = co.correlation_over_epochs(
        surp_versions, rm, domain_only=True, mode="mean"
    )
    fig, ax = plt.subplots()
    viz.finetune_correlation_curve(curve, metric="pearson", ax=ax)
    save_fig(ax, f"{slug}_finetune_correlation_curve")

    # Surprisal-model comparison, repeated under each fixed-effects spec.
    cmps = []
    for spec in mc.MODEL_SPECS:
        print(f"Step 5 — model comparison (spec={spec})")
        cmp = mc.model_comparison_over_epochs(
            surp_versions, rm, prompt_surp, measure=MEASURE, spec=spec
        )
        cmp["spec"] = spec
        cmps.append(cmp)
        print(cmp.to_string(index=False))

        fig, ax = plt.subplots()
        viz.model_comparison_curve(cmp, metric="delta_ll", ax=ax)
        save_fig(ax, f"{slug}_finetune_model_comparison_{spec}")

    cmp_all = pd.concat(cmps, ignore_index=True)
    cmp_all.insert(0, "model_lm", slug)
    cmp_all.to_csv(PROJECT_ROOT / f"results_{slug}.csv", index=False)

    # Cross-model summary row: all-readers correlation, slope, best aligned ΔLL.
    base = corr_df[(corr_df["group"] == "all") & (~corr_df["domain_only"])].iloc[0]
    aligned_cov = cmp_all[(cmp_all["spec"] == "covariates") & (cmp_all["model"] == "aligned")]
    return {
        "summary": {
            "model": slug,
            "pearson": base["pearson"],
            "spearman": base["spearman"],
            "slope_ms_per_bit": float(fit.params["surprisal"]),
            "r2": float(fit.rsquared),
            "aligned_delta_ll": float(aligned_cov["delta_ll"].max()),
        },
    }


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    # ── Step 1 — reading time (shared across models) ─────────────────────────
    print("Step 1 — reading time")
    words = data.load_word_features()
    rm_raw = data.load_reading_measures()
    rm = rt.clean_reading_times(rm_raw, MEASURE)
    print(
        f"  words={len(words)}  raw_rows={len(rm_raw)}  "
        f"cleaned={len(rm)} ({len(rm) / len(rm_raw):.1%})"
    )
    agg = rt.aggregate_rt(rm, MEASURE)
    print(
        "  ",
        {k: (len(v), round(v[f"mean_{MEASURE}"].mean())) for k, v in agg.items()},
    )

    # ── Steps 2-6 — run the whole workflow per model ─────────────────────────
    summaries = []
    for slug, name in config.MODELS.items():
        out = run_model(slug, name, words, rm)
        summaries.append(out["summary"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(PROJECT_ROOT / "results_models_summary.csv", index=False)
    print("\n=== cross-model summary ===")
    print(summary_df.to_string(index=False))

    # ── Cross-model figures ──────────────────────────────────────────────────
    print("Cross-model figures")
    fig, ax = plt.subplots()
    viz.across_models_correlation_bars(summary_df, ax=ax)
    save_fig(ax, "across_models_correlation")

    fig, ax = plt.subplots()
    viz.across_models_bar(summary_df, "slope_ms_per_bit", ylabel="slope (ms/bit)", ax=ax)
    save_fig(ax, "across_models_slope")

    fig, ax = plt.subplots()
    viz.across_models_bar(
        summary_df, "aligned_delta_ll", ylabel="reader-aligned ΔLL (best ckpt)", ax=ax
    )
    save_fig(ax, "across_models_aligned_delta_ll")

    print(f"\nDone. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
