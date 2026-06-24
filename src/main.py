"""Non-notebook driver for the PoTeC decoder-LM pipeline (steps 1-6).

Mirrors ``notebooks/pipeline.ipynb``: reading-time cleaning, causal-LM
surprisal, attention (raw + flow), the surprisal/attention vs gaze analysis, and
(optional) DAPT fine-tuning. Prints the same summaries the notebook shows and
writes every plot to ``figures/``.

Runs all steps end to end (step 4 DAPT included; GPU recommended).
Run from the project root:

    python -m src.main
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src import analysis as an
from src import attention as at
from src import data
from src import reading_time as rt
from src import surprisal as su
from src import viz

MEASURE = "TFT"  # total fixation time == TRT
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


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    # ── Step 1 — reading time ────────────────────────────────────────────────
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

    # ── Step 2 — model surprisal (baseline) ──────────────────────────────────
    print("Step 2 — surprisal")
    model, tok = su.load_causal_lm()
    surp = su.compute_surprisal(words, model, tok)  # prompt=None
    print(surp.head().to_string())

    # ── Step 3 — attention (raw on full corpus) ──────────────────────────────
    print("Step 3 — attention (raw)")
    amodel, atok = su.load_causal_lm(attn=True)
    attn_raw = at.extract_attention(words, amodel, atok, method="raw")
    et = an.build_et_table(rm_raw, domain="all", participants="all")
    print(f"  attn_raw={attn_raw.shape}  et={et.shape}")

    # ── Step 5 — analysis ────────────────────────────────────────────────────
    print("Step 5 — analysis")
    merged = an.merge_surprisal_rt(surp, rm)
    rows = []
    for grp in ["all", "experts", "novices"]:
        for dom_only in (False, True):
            r = an.correlate_surprisal(
                merged, participants=grp, domain_only=dom_only, mode="mean"
            )
            rows.append({"group": grp, "domain_only": dom_only, **r})
    corr_df = pd.DataFrame(rows)[
        ["group", "domain_only", "n", "pearson", "spearman"]
    ]
    print(corr_df.to_string(index=False))

    fit = an.regress_rt(merged, participants="all")
    print(
        f"  slope={fit.params['surprisal']:.2f} ms/bit  R2={fit.rsquared:.3f}"
    )

    # ── Figures ──────────────────────────────────────────────────────────────
    print("Figures")
    # surprisal vs reading time (binned scatter + OLS fit)
    agg_words = an._aggregate_words(
        an._filter_participants(merged, "all"), MEASURE, "mean"
    )
    fig, ax = plt.subplots()
    viz.surprisal_scatter(agg_words, ax=ax)
    save_fig(ax, "surprisal_scatter")

    # regression slope per reader group
    slopes = pd.DataFrame(
        [
            {
                "group": grp,
                "slope": an.regress_rt(merged, participants=grp).params["surprisal"],
            }
            for grp in ["all", "experts", "novices"]
        ]
    )
    fig, ax = plt.subplots()
    viz.expert_novice_slopes(slopes, ax=ax)
    save_fig(ax, "expert_novice_slopes")

    # attention layer curve (raw)
    attn_corr = an.correlate_attention(attn_raw, et, attention_method="raw")
    fig, ax = plt.subplots()
    viz.attention_layer_curve(attn_corr, feature="pca", ax=ax)
    save_fig(ax, "attention_layer_curve")

    # ── Step 4 — DAPT fine-tuning (GPU recommended) ──────────────────────────
    print("Step 4 — DAPT fine-tuning")
    from src import finetune as ft

    manifest = ft.finetune_dapt("physics", epochs=10, save_every=2)
    fig, ax = plt.subplots()
    viz.perplexity_curve(manifest, ax=ax)
    save_fig(ax, "perplexity_curve")

    surp_versions = ft.recompute_surprisal_over_checkpoints(words, manifest)
    curve = an.correlation_over_epochs(
        surp_versions, rm, domain_only=True, mode="mean"
    )
    fig, ax = plt.subplots()
    viz.finetune_correlation_curve(curve, metric="pearson", ax=ax)
    save_fig(ax, "finetune_correlation_curve")

    print(f"Done. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
