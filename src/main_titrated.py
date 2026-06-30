"""Driver for the knowledge-titrated personal-surprisal follow-up experiment.

Implements ``potec-knowledge-titrated-surprisal-plan.md`` — a sharper test than
``src.main``'s binary reader-aligned comparison. Where ``src.main`` adapts the LM
to a *group* (physics vs biology) and routes surprisal by discipline, this driver
makes the *degree* of domain adaptation a continuous function of each reader's
measured background-knowledge score ``bg`` (PoTeC ``mean_acc_bq``).

Per model (``config.MODELS``) and per reading measure (FPRT placebo, GP, TFT) it:

  Step 1  reading time           reused per-measure PoTeC cleaning (src.features)
  Step 2  DAPT checkpoint ladder reused 4ⁿ-step schedule (shared with src.main)
  Step 3  per-reader ΔLL curves  domain-matched surprisal at each checkpoint,
                                 per-reader OLS ΔLL over the ladder
  Step 4  k* + monotonicity      within-discipline Spearman ρ(k*_p, score_p) ± CI
  Step 5  titrated head-to-head  reader-clustered Vuong: titrated vs binary
                                 reader-aligned (paper's best) vs population-k

Steps 4-5 run for two reader-score axes (``AXES``), both LM-domain-matched to the
reader, differing only in *which* score sets each reader's checkpoint depth:
  • ``grade``      — background-quiz knowledge ``bg`` (``mean_acc_bq``, original).
  • ``studytime``  — study level ``seniority`` (``level_of_studies_numeric``:
                     undergrad → less-trained checkpoint, graduate → more-trained).

Central hypothesis: ρ(k*, score) > 0 within group (higher score → fit peaks at a
more adapted checkpoint), strongest on the late measures (GP, TFT), null on FPRT.

Writes ``figures/titrated_*`` plots and ``results_titrated_<slug>.csv`` tables.
Run from the project root (GPU recommended — DAPT runs unless checkpoints cache):

    python -m src.main_titrated
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis import titration as ti
from src.features import data
from src.features import reading_time as rt

# DAPT method + checkpoint ladder — identical to src.main so the cached
# checkpoints (local artifacts/ + Hub mirror) are shared, not retrained.
FINETUNE_LORA = True
DAPT_LR = 2e-4 if FINETUNE_LORA else 2e-5
DAPT_MAX_STEPS = 16_384
DAPT_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096, 16384]
# Monotone score→k map families fit under nested CV (plan §7); both pre-registered.
MAP_KINDS = ("tercile", "isotonic")

# Reader-score axes that drive checkpoint depth. Both route the LM to the reader's
# own discipline (``_reader_matched``) — they differ only in *which* reader score
# sets the depth. ``grade`` = background-quiz knowledge (``bg``, the original axis);
# ``studytime`` = study level (``seniority``: undergrad→shallow, graduate→deep).
# slug -> (reader_table column, human-readable label).
AXES = {"grade": ("bg", "background knowledge"),
        "studytime": ("seniority", "study level (undergrad→grad)")}

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


def _plot_delta_ll_curves(dll_df: pd.DataFrame, ax, score_col: str, label: str) -> None:
    """One ΔLL-over-checkpoints line per reader, colored by the reader score.

    The headline figure (plan §10.2): x = training steps (log), y = ΔLL, hue =
    ``score_col``. If the hypothesis holds, high-score readers (warm) peak further
    right.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=dll_df[score_col].min(), vmax=dll_df[score_col].max())
    for rid, g in dll_df.groupby("reader_id"):
        g = g.sort_values("step")
        ax.plot(
            g["step"].clip(lower=1), g["delta_ll"],
            color=cm.viridis(norm(g[score_col].iloc[0])), alpha=0.5, lw=0.8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("DAPT training steps (k)")
    ax.set_ylabel("per-reader ΔLL")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    ax.get_figure().colorbar(sm, ax=ax, label=label)


def _plot_kstar_scatter(
    kstar: pd.DataFrame, rho_df: pd.DataFrame, ax, score_col: str, label: str
) -> None:
    """k* vs reader-score scatter, one panel-color per discipline, ρ annotated."""
    for grp, g in kstar.groupby("group"):
        ax.scatter(g[score_col], g["step"].clip(lower=1), label=grp, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("k*  (steps at peak ΔLL)")
    txt = "  ".join(
        f"{r.group}: ρ={r.rho:.2f} [{r.ci_lo:.2f},{r.ci_hi:.2f}]"
        for r in rho_df.itertuples()
    )
    ax.set_title(txt, fontsize=8)
    ax.legend()


def run_model(slug: str, name: str, words, rm_raw) -> None:
    """Run the titration experiment for one model; write figures + a results CSV."""
    print(f"\n=== model: {slug} ({name}) ===")

    # ── Step 2 — DAPT checkpoint ladder (shared schedule with src.main) ───────
    print("Step 2 — DAPT checkpoint ladder")
    from src.modeling import finetune as ft

    batch_size = config.DAPT_BATCH_SIZE.get(slug, 8)
    grad_accum = config.DAPT_GRAD_ACCUM.get(slug, 1)
    manifest = pd.concat(
        [
            ft.finetune_dapt("physics", base_model=name, max_steps=DAPT_MAX_STEPS,
                             checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                             batch_size=batch_size, grad_accum=grad_accum,
                             learning_rate=DAPT_LR, lora=FINETUNE_LORA),
            ft.finetune_dapt("biology", base_model=name, max_steps=DAPT_MAX_STEPS,
                             checkpoint_steps=DAPT_CHECKPOINT_STEPS,
                             batch_size=batch_size, grad_accum=grad_accum,
                             learning_rate=DAPT_LR, lora=FINETUNE_LORA),
        ],
        ignore_index=True,
    )
    surp_versions = ft.recompute_surprisal_over_checkpoints(words, manifest)
    idx_to_step = manifest.groupby("index")["step"].first().astype(int).to_dict()

    # ── Steps 3-5 — per reading measure ──────────────────────────────────────
    rho_rows, vuong_rows, kstar_rows = [], [], []
    for label, col in ti.MEASURES.items():
        print(f"\n--- measure {label} ({col}) ---")
        rm = rt.clean_reading_times(rm_raw, col)  # rm_raw already has is_expert
        reader_tbl = ti.reader_table(rm)

        # Step 3 — per-reader ΔLL over the checkpoint ladder (score-independent).
        dll = ti.per_reader_delta_ll(surp_versions, rm, reader_tbl, col, idx_to_step)
        # Step 4 — each reader's k* (peak ΔLL); also score-independent.
        kstar = ti.find_kstar(dll)
        kstar_rows.append(kstar.assign(measure=label))

        # Step 5 prep — population-k and binary reader-aligned k (axis-independent).
        pop_index = ti.population_optimal_index(dll)
        aligned_index = ti.aligned_optimal_index(surp_versions, rm, col)

        # Steps 4-5 per reader-score axis (grade=bg, studytime=seniority). The LM
        # is reader-domain-matched in every axis; only the depth-setting score
        # changes. FPRT stays the placebo channel for the monotonicity ρ.
        for axis, (score_col, score_label) in AXES.items():
            print(f"\n  ·· axis {axis} ({score_col}) ··")

            fig, ax = plt.subplots()
            _plot_delta_ll_curves(dll, ax, score_col, score_label)
            ax.set_title(f"{slug} — per-reader ΔLL ({label}, {axis})")
            save_fig(ax, f"titrated_{slug}_{axis}_delta_ll_{label}")

            rho = ti.within_group_rho(kstar, score_col=score_col)
            rho.insert(0, "measure", label)
            rho.insert(1, "axis", axis)
            print(rho.to_string(index=False))
            rho_rows.append(rho)
            fig, ax = plt.subplots()
            _plot_kstar_scatter(kstar, rho, ax, score_col, score_label)
            save_fig(ax, f"titrated_{slug}_{axis}_kstar_{label}")

            # Titrated vs binary-aligned vs population-k (Vuong, nested-CV map).
            for kind in MAP_KINDS:
                oof = ti.titrated_oof_index(
                    kstar, sorted(idx_to_step), kind=kind, score_col=score_col
                )
                d = ti.build_headtohead(
                    surp_versions, rm, oof, pop_index, aligned_index, col
                )
                v = ti.head_to_head(d, col)
                v.insert(0, "measure", label)
                v.insert(1, "axis", axis)
                v.insert(2, "map", kind)
                v["pop_index"] = pop_index
                v["aligned_index"] = aligned_index
                print(f"  [{axis}/{kind}] pop_k={idx_to_step.get(pop_index)} "
                      f"aligned_k={idx_to_step.get(aligned_index)}")
                print(v.to_string(index=False))
                vuong_rows.append(v)

    # ── Results CSVs ──────────────────────────────────────────────────────────
    pd.concat(kstar_rows, ignore_index=True).to_csv(
        PROJECT_ROOT / f"results_titrated_kstar_{slug}.csv", index=False
    )
    pd.concat(rho_rows, ignore_index=True).to_csv(
        PROJECT_ROOT / f"results_titrated_rho_{slug}.csv", index=False
    )
    pd.concat(vuong_rows, ignore_index=True).to_csv(
        PROJECT_ROOT / f"results_titrated_vuong_{slug}.csv", index=False
    )


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("Step 1 — load words + reading measures")
    words = data.load_word_features()
    rm_raw = data.load_reading_measures()
    print(f"  words={len(words)}  reading_rows={len(rm_raw)}")

    for slug, name in config.MODELS.items():
        run_model(slug, name, words, rm_raw)

    print(f"\nDone. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
