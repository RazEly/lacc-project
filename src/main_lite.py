"""Lightweight driver for the PoTeC decoder-LM pipeline (steps 1-6).

A trimmed copy of ``src/main.py`` for fast iteration / low-VRAM boxes:

* only german-gpt2 (124M) — the LLäMmlein 1B decoder is dropped, so no
  multi-GB weights download and DAPT fits a small GPU;
* only the first 4 DAPT checkpoints per domain (not the full 7), halving the
  fine-tuning + recompute time.

It adds one figure the full pipeline does not have: a **cross-domain perplexity**
bar chart. The baseline, the physics-DAPT and the biology-DAPT model are each
scored on the SAME general / physics / biology validation sets, so you can read
off what fine-tuning for physics costs on biology and on plain general text. The
general-domain validation set is a streamed German-Wikipedia slice (config
``GENERAL_HF_REPO``); physics/biology reuse each domain's held-out DAPT split.

Run from the project root:

    uv run python -m src.main_lite
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
# Lightweight overrides vs main.py ───────────────────────────────────────────
# Only german-gpt2 (no LLäMmlein 1B).
MODELS_LITE = {"german-gpt2": config.MODELS["german-gpt2"]}
# Only the first 4 DAPT checkpoints per domain (main.py uses 7).
N_CHECKPOINTS = 4

# DAPT checkpoints are reused from artifacts/ (LoRA runs), not re-trained. The
# cross-domain perplexity baseline reuses the step-0 checkpoint (grown
# embeddings) so every model loads the same way.
FINETUNE_LORA = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"


def load_existing_manifest(base_model: str, domain: str, n_checkpoints: int) -> pd.DataFrame:
    """Load a DAPT run's saved manifest from artifacts/ and keep the first N checkpoints.

    Reuses the checkpoints already on disk (no re-training): reads
    ``<CHECKPOINTS_DIR>/<base>_<domain>_lora/manifest.csv`` and returns its first
    ``n_checkpoints`` rows (index 0 = baseline … up). The full run has 7; the lite
    slice is the first 4 evenly-spaced ones.
    """
    suffix = "_lora" if FINETUNE_LORA else ""
    run_dir = config.CHECKPOINTS_DIR / f"{Path(base_model).name}_{domain}{suffix}"
    manifest = pd.read_csv(run_dir / "manifest.csv")
    manifest = manifest.sort_values("index").head(n_checkpoints).reset_index(drop=True)
    missing = [c for c in manifest["checkpoint"] if not Path(c).exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints for {domain}: {missing}")
    print(f"  [{domain}] reusing {len(manifest)} checkpoints from {run_dir}")
    return manifest


def save_fig(ax, name: str) -> None:
    """Save the figure owning ``ax`` to ``figures/<name>.png`` and close it."""
    fig = ax.get_figure()
    fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def run_model(slug: str, name: str, words, rm) -> dict:
    """Run steps 2-6 for one model; write ``<slug>_*`` figures + csv; return a summary."""
    print(f"\n=== model: {slug} ({name}) ===")

    # ── Step 2 — model surprisal (baseline) ──────────────────────────────────
    print("Step 2 — surprisal")
    model, tok = su.load_causal_lm(name)
    surp = su.compute_surprisal(words, model, tok)  # prompt=None

    # prompted-baseline surprisal: the un-adapted model with a discipline-matched
    # system prompt prepended (the prompting analogue of fine-tuning).
    s_pp = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["physics"]
    ).rename(columns={"surprisal": "s_prompt_phys"})
    s_pb = su.compute_surprisal(
        words, model, tok, prompt=config.GRAD_STUDENT_PROMPTS["biology"]
    ).rename(columns={"surprisal": "s_prompt_bio"})
    prompt_surp = s_pp.merge(s_pb, on=["text_id", "word_index_in_text"])

    # field × level prompted baseline: prompt matches BOTH discipline AND level.
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
    corr_df = pd.DataFrame(rows)[["group", "domain_only", "n", "pearson", "spearman"]]
    print(corr_df.to_string(index=False))

    fit = co.regress_rt(merged, participants="all")
    print(f"  slope={fit.params['surprisal']:.2f} ms/bit  R2={fit.rsquared:.3f}")

    # ── Figures (per model) ──────────────────────────────────────────────────
    print("Figures")
    agg_words = co._aggregate_words(
        co._filter_participants(merged, "all"), MEASURE, "mean"
    )
    fig, ax = plt.subplots()
    viz.surprisal_scatter(agg_words, ax=ax)
    save_fig(ax, f"{slug}_lite_surprisal_scatter")

    # ── Step 4 — reuse existing DAPT checkpoints (no re-training) ─────────────
    # Load the manifests already in artifacts/ and keep the FIRST 4 checkpoints
    # per domain (index 0-3) — the lite slice of the full 7. No GPU training.
    print("Step 4 — reuse DAPT checkpoints (first 4)")
    from src.modeling import finetune as ft

    phys_manifest = load_existing_manifest(name, "physics", N_CHECKPOINTS)
    bio_manifest = load_existing_manifest(name, "biology", N_CHECKPOINTS)
    manifest = pd.concat([phys_manifest, bio_manifest], ignore_index=True)

    fig, ax = plt.subplots()
    viz.perplexity_curve(manifest, ax=ax)
    save_fig(ax, f"{slug}_lite_perplexity_curve")

    # ── Cross-domain perplexity (the new lite figure) ────────────────────────
    # Baseline vs physics-DAPT vs biology-DAPT, each scored on the SAME general /
    # physics / biology validation sets — the effect of fine-tuning for physics
    # on biology and on general text.
    print("Cross-domain perplexity (general / physics / biology)")
    ppl_df = ft.cross_domain_perplexity(phys_manifest, bio_manifest, base_model=name)
    ppl_df.insert(0, "model_lm", slug)
    ppl_df.to_csv(PROJECT_ROOT / f"results_cross_domain_ppl_{slug}.csv", index=False)
    fig, ax = plt.subplots()
    viz.cross_domain_perplexity_bars(ppl_df, ax=ax)
    save_fig(ax, f"{slug}_lite_cross_domain_perplexity")

    surp_versions = ft.recompute_surprisal_over_checkpoints(words, manifest)
    curve = co.correlation_over_epochs(
        surp_versions, rm, domain_only=True, mode="mean"
    )
    fig, ax = plt.subplots()
    viz.finetune_correlation_curve(curve, metric="pearson", ax=ax)
    save_fig(ax, f"{slug}_lite_finetune_correlation_curve")

    cmps = []
    # Our specs (base + full), every surprisal source, all checkpoints. Each is run
    # twice: raw RT and log RT (LLs only comparable WITHIN a response, not across).
    for spec in ("covariates", "full"):
        for log_rt in (False, True):
            resp = "log" if log_rt else "raw"
            print(f"Step 5 — model comparison (spec={spec}, response={resp}_rt)")
            cmp = mc.model_comparison_over_epochs(
                surp_versions, rm, prompt_surp, measure=MEASURE, spec=spec,
                log_rt=log_rt,
            )
            cmp["response"] = f"{resp}_rt"
            cmps.append(cmp)
            print(cmp.to_string(index=False))

            fig, ax = plt.subplots()
            viz.model_comparison_curve(cmp, metric="delta_ll", ax=ax)
            save_fig(ax, f"{slug}_lite_model_comparison_{spec}_{resp}")

    # Škrjanec et al. (2023) full model: general (baseline) surprisal only, first 2
    # checkpoints, three-way surprisal × expertise × terminology. Raw + log RT.
    for log_rt in (False, True):
        resp = "log" if log_rt else "raw"
        print(f"Step 5 — paper full model (baseline, first 2 ckpts, {resp}_rt)")
        cmp = mc.model_comparison_over_epochs(
            surp_versions, rm, prompt_surp, measure=MEASURE, spec="paper_full",
            models=["baseline"], indices=[0, 1], log_rt=log_rt,
        )
        cmp["response"] = f"{resp}_rt"
        cmps.append(cmp)
        print(cmp.to_string(index=False))

    cmp_all = pd.concat(cmps, ignore_index=True)
    cmp_all.insert(0, "model_lm", slug)
    cmp_all.to_csv(PROJECT_ROOT / f"results_lite_{slug}.csv", index=False)

    base = corr_df[(corr_df["group"] == "all") & (~corr_df["domain_only"])].iloc[0]
    # raw-RT covariates spec only: log-RT LLs are on a different scale.
    aligned_cov = cmp_all[
        (cmp_all["spec"] == "covariates")
        & (cmp_all["model"] == "aligned")
        & (cmp_all["response"] == "raw_rt")
    ]
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

    # ── Steps 2-6 — german-gpt2 only ─────────────────────────────────────────
    summaries = []
    for slug, name in MODELS_LITE.items():
        out = run_model(slug, name, words, rm)
        summaries.append(out["summary"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(PROJECT_ROOT / "results_lite_models_summary.csv", index=False)
    print("\n=== summary ===")
    print(summary_df.to_string(index=False))

    print(f"\nDone. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
