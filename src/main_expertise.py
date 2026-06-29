"""Expertise-aligned variant of the PoTeC decoder-LM model comparison.

A copy of ``src/main.py`` with two deliberate changes:

  1. **No training.** The DAPT step is skipped entirely; the first 3 fine-tuning
     checkpoints (index 0 = un-adapted baseline, 1, 2) are read straight from the
     manifests already saved under ``artifacts/`` — see ``load_manifest``.
  2. **A new reader-aligned surprisal source, ``expertise_aligned``.** Where
     ``src/main.py``'s ``aligned`` model routes by the reader's bare *discipline*
     (every physicist gets the physics LM, every biologist the biology LM), this
     variant routes by *demonstrated* domain expertise — the reader's
     background-question accuracy (PoTeC ``mean_acc_bq``) on that text, per trial:

         physics text + mean_acc_bq > 0.9  -> physics LM   (physics expert)
         biology text + mean_acc_bq > 0.9  -> biology LM   (biology expert)
         mean_acc_bq <= 0.9 (either task)  -> baseline LM  (non-expert)

     0.9 is the EyeBench PoTeC Domain-Expertise (DE) threshold: a reading is
     "expert" iff the reader aced that text's background questions. Routing is per
     (reader×text) by the TEXT's domain, so the three indicators are mutually
     exclusive; readings below threshold fall back to the un-adapted baseline.

The point of this script is the head-to-head: does ``expertise_aligned`` (which
withholds the adapted LM from non-experts) fit reading time better or worse than
the discipline-only ``aligned`` (``src/main.py``'s current model)? Both are fit in
the same mixed model as ``baseline`` / ``physics`` / ``biology`` so their ΔLLs are
directly comparable.

Run from the project root (no GPU needed — surprisal only, no training):

    python -m src.main_expertise
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.features import data
from src.features import surprisal as su
from src.analysis import model_comparison as mc
from src.features import reading_time as rt
from src.analysis import viz

MEASURE = "TFT"  # total fixation time == TRT
# Only the first N DAPT checkpoints are used this run (index 0 = baseline, then 1,
# 2). They are loaded from artifacts/, never retrained.
N_CHECKPOINTS = 3
# Surprisal sources fit in the comparison. No prompted models here (prompt_surp is
# skipped); the two reader-aligned sources are the headline: ``aligned``
# (discipline-only, src/main.py) vs ``expertise_aligned`` (the new variant).
COMPARE_MODELS = ["baseline", "physics", "biology", "aligned", "expertise_aligned"]
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


def load_manifest(name: str, domain: str, n: int = N_CHECKPOINTS) -> pd.DataFrame:
    """Load the first ``n`` saved DAPT checkpoints for one (model, domain) from disk.

    Reads ``artifacts/<base>_<domain>_lora/manifest.csv`` (written by a previous
    ``finetune.finetune_dapt`` run) and keeps the ``n`` lowest checkpoint indices
    — index 0 is the un-adapted baseline. No training and no Hub round-trip: the
    checkpoint dirs must already exist locally. Returns the same columns
    ``finetune_dapt`` would (``domain``, ``checkpoint``, ``index``, ``epoch`` …).
    """
    run_dir = config.CHECKPOINTS_DIR / f"{Path(name).name}_{domain}_lora"
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no saved checkpoints for {domain} at {manifest_path} — "
            "run src.main once to produce them, or restore artifacts/"
        )
    df = pd.read_csv(manifest_path).sort_values("index").head(n).reset_index(drop=True)
    missing = [c for c in df["checkpoint"] if not (PROJECT_ROOT / c).exists()]
    if missing:
        raise FileNotFoundError(f"manifest references missing checkpoints: {missing}")
    print(f"  [{domain}] loaded {len(df)} checkpoints from {run_dir} (no training)")
    return df


def recompute_surprisal(words_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Recompute per-word surprisal with each manifest checkpoint (no training).

    A training-free re-implementation of
    ``finetune.recompute_surprisal_over_checkpoints`` so this script never imports
    the DAPT module (and its heavy ``datasets`` dependency). ``surprisal.load_causal_lm``
    reattaches each saved LoRA adapter to its base model. Tags every row with
    ``checkpoint`` / ``index`` / ``epoch`` / ``domain`` for index-paired comparison.
    """
    frames = []
    for _, row in manifest.iterrows():
        model, tok = su.load_causal_lm(row.checkpoint)
        sup = su.compute_surprisal(words_df, model, tok)
        sup["checkpoint"] = row.checkpoint
        sup["index"] = row["index"]
        sup["epoch"] = row.epoch
        sup["domain"] = row.domain
        frames.append(sup)
    return pd.concat(frames, ignore_index=True)


def run_model(slug: str, name: str, words, rm) -> dict:
    """Compare surprisal sources for one LM over the first 3 loaded checkpoints.

    Skips DAPT: pairs the pre-saved physics + biology checkpoints by index, recomputes
    surprisal at each, and fits the ``COMPARE_MODELS`` sources (incl. the new
    ``expertise_aligned``) per fixed-effects spec. Writes ``<slug>_*`` figures +
    ``results_<slug>_expertise.csv``; returns a cross-model summary row.
    """
    print(f"\n=== model: {slug} ({name}) ===")

    # ── Load the first 3 checkpoints per domain (no training) ────────────────
    print(f"Loading first {N_CHECKPOINTS} checkpoints from artifacts/")
    manifest = pd.concat(
        [load_manifest(name, "physics"), load_manifest(name, "biology")],
        ignore_index=True,
    )
    fig, ax = plt.subplots()
    viz.perplexity_curve(manifest, ax=ax)
    save_fig(ax, f"{slug}_expertise_perplexity_curve")

    # ── Recompute surprisal at each loaded checkpoint ────────────────────────
    print("Recomputing surprisal over checkpoints")
    surp_versions = recompute_surprisal(words, manifest)

    # ── Model comparison: discipline-aligned vs expertise-aligned ────────────
    # prompt_surp=None -> the prompted-baseline models are skipped (not part of this
    # comparison). Each spec is its own mixed-effects control structure.
    cmps = []
    for spec in mc.MODEL_SPECS:
        print(f"Model comparison (spec={spec})")
        cmp = mc.model_comparison_over_epochs(
            surp_versions, rm, None, measure=MEASURE, spec=spec,
            models=COMPARE_MODELS,
        )
        cmp["spec"] = spec
        cmps.append(cmp)
        print(cmp.to_string(index=False))

        fig, ax = plt.subplots()
        viz.model_comparison_curve(cmp, metric="delta_ll", ax=ax)
        save_fig(ax, f"{slug}_expertise_model_comparison_{spec}")

    cmp_all = pd.concat(cmps, ignore_index=True)
    cmp_all.insert(0, "model_lm", slug)
    cmp_all.to_csv(PROJECT_ROOT / f"results_{slug}_expertise.csv", index=False)

    # Cross-model summary: best ΔLL of each reader-aligned source (covariates spec),
    # and the gap (expertise_aligned - aligned). Positive gap => withholding the
    # adapted LM from non-experts helps.
    cov = cmp_all[cmp_all["spec"] == "covariates"]

    def best(model):
        return float(cov[cov["model"] == model]["delta_ll"].max())

    aligned_best = best("aligned")
    expertise_best = best("expertise_aligned")
    return {
        "summary": {
            "model": slug,
            "baseline_delta_ll": best("baseline"),
            "aligned_delta_ll": aligned_best,
            "expertise_aligned_delta_ll": expertise_best,
            "expertise_minus_aligned": expertise_best - aligned_best,
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
    # Trial counts behind the three expertise indicators (per reader×text, by the
    # text's domain and the mean_acc_bq > 0.9 DE threshold; see expertise_aligned).
    by_trial = rm.groupby(["reader_id", "text_id"])[
        ["text_domain_numeric", "mean_acc_bq"]
    ].first()
    expert = by_trial["mean_acc_bq"] > 0.9
    de_phys = (expert & (by_trial["text_domain_numeric"] == 1)).sum()
    de_bio = (expert & (by_trial["text_domain_numeric"] == 0)).sum()
    non_exp = (~expert).sum()
    print(f"  trials: DE-physics={de_phys}  DE-biology={de_bio}  non-expert={non_exp}")

    # ── Compare the two reader-aligned variants per model ────────────────────
    summaries = []
    for slug, name in config.MODELS.items():
        out = run_model(slug, name, words, rm)
        summaries.append(out["summary"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(PROJECT_ROOT / "results_expertise_summary.csv", index=False)
    print("\n=== expertise vs discipline summary (covariates spec, best ckpt) ===")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots()
    viz.across_models_bar(
        summary_df, "expertise_minus_aligned",
        ylabel="ΔLL(expertise_aligned) − ΔLL(aligned)", ax=ax,
    )
    save_fig(ax, "across_models_expertise_minus_aligned")

    print(f"\nDone. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
