"""Driver for the PoTeC decoder-LM pipeline (steps 1-6).

Reading-time cleaning, causal-LM surprisal, raw attention, the
surprisal/attention vs gaze analysis, DAPT fine-tuning, and the reader-aligned
model comparison with its significance tests. Prints a summary per step and
writes every plot to ``figures/``.

The whole workflow (steps 2-6) is run independently for every model in
``config.MODELS`` — german-gpt2 plus the German-only LLäMmlein 1B decoder
— each writing its own ``<slug>_*`` figures and ``results_<slug>.csv``. A final
cross-model block compares them on one axes (surprisal-RT correlation,
regression slope, attention-vs-gaze layer curve, reader-aligned ΔLL).

Runs all steps end to end (step 4 DAPT included; GPU recommended). Weights are
pulled from the Hub on first use — nothing is pre-downloaded here.
Run from the project root:

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
# DAPT (step 4) is LoRA-only. LoRA needs ~10x a full-FT LR — its adapters are
# zero-initialised and only a few rank-r params train, so a low LR barely moves them.
DAPT_LR = 2e-4
# DAPT checkpoint schedule, matching Škrjanec et al. (papers/07): train ≤16,384
# steps and save checkpoints at 4ⁿ steps for n∈{1..7}. include_baseline still
# prepends the un-fine-tuned model as checkpoint index 0.
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
    correlation, the regression slope, and the best reader-aligned ΔLL, plus the
    attention-correlation table for the combined layer curve.
    """
    print(f"\n=== model: {slug} ({name}) ===")

    # ── Step 2 — model surprisal (baseline) ──────────────────────────────────
    print("Step 2 — surprisal")
    model, tok = su.load_causal_lm(name)
    surp = su.compute_surprisal(words, model, tok)  # prompt=None
    print(surp.head().to_string())

    # prompted-baseline surprisal: the un-adapted model with a discipline-matched
    # system prompt prepended (the prompting analogue of fine-tuning). One column
    # per discipline; the comparison mixes them by reader discipline (S_prompted).
    s_pp = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["physics"]
    ).rename(columns={"surprisal": "s_prompt_phys"})
    s_pb = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["biology"]
    ).rename(columns={"surprisal": "s_prompt_bio"})
    prompt_surp = s_pp.merge(s_pb, on=["text_id", "word_index_in_text"])

    # field × level prompted baseline: same un-adapted model, but the prompt
    # matches BOTH discipline AND study level (undergrad/grad). One column per
    # (level, discipline) cell; the comparison mixes by reader discipline AND
    # level (S_prompted_level). Column suffix: ug = undergraduate, grad = graduate.
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

    # Fine-tune both domains — the model comparison needs physics- and
    # biology-adapted surprisal (plus the shared step-0 baseline). Budget by a fixed
    # step count, not tokens: Škrjanec et al. (papers/07) train ≤16,384 steps and
    # checkpoint at 4ⁿ steps (n∈{1..7}). Matching that schedule here makes the
    # checkpoint indices line up with the paper. Both domains share the schedule, so
    # they see the same step at the same checkpoint index.
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

    # Five-model surprisal comparison on the whole corpus (all readers × words):
    # does reader-aligned surprisal (physics LM for physicists, biology LM for
    # biologists) fit better than any single model? Repeated under each
    # fixed-effects spec (covariates / expertise-only / full), since the right
    # control structure is itself an open question.
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

    # Cross-model summary: all-readers, both-domains surprisal-RT correlation,
    # the regression slope, and the strongest reader-aligned ΔLL (covariates spec).
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

    # ── Cross-model figures (all three on one axes) ──────────────────────────
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
