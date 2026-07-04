"""Project figures (see viz.md). Every function saves a PNG under ``figures/``
(never ``.show()``) and returns the plotted frame.

fig1  perplexity_grid            held-out perplexity vs step, 2×2 (model × ft-domain)
fig2  sentence_surprisal_example one sentence: baseline surprisal, Δ per checkpoint, Δ per prompt
fig3  mean_surprisal_over_steps  mean aligned surprisal vs step; prompted arms as intercepts
fig4  delta_ll_curves            ΔLL vs step per LM; checkpoint-independent arms as intercepts
fig5  rt_vs_surprisal            log RT vs baseline-GPT-2 surprisal

"aligned" here is text-level (checkpoint / prompt domain matches ``text_domain``);
the reader-level alignment lives in the mixed models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, WORD_KEY

FIGURES_DIR = PROJECT_ROOT / "figures"
MODEL_LABELS = {"german-gpt2": "GPT-2", "llammlein-1b": "Llama"}
FT_DOMAINS = ("biology", "physics")  # fig1 row order
PROMPT_COLORS = {"physics": "tab:purple", "biology": "tab:green", "neutral": "gray"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)


def _slugs(df: pd.DataFrame, col: str) -> list[str]:
    """Model slugs in ``col``, ordered like MODEL_LABELS (unknowns appended)."""
    present = set(df[col])
    return [s for s in MODEL_LABELS if s in present] + sorted(
        present - set(MODEL_LABELS)
    )


# def perplexity_grid(manifests: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> pd.DataFrame:
#     """Fig 1 — held-out perplexity vs training step.
#
#     2×2 grid: columns = LM (GPT-2 / Llama), rows = fine-tune domain (biology /
#     physics); one line per held-out test set. ``manifests`` is long with columns
#     ``model``, ``ft_domain``, ``eval_domain``, ``step``, ``perplexity``
#     (cross-domain eval — every run scored on BOTH test sets).
#     """
#     plt = _plt()
#     models = _slugs(manifests, "model")
#     fig, axes = plt.subplots(
#         len(FT_DOMAINS), len(models), figsize=(5 * len(models), 3.5 * len(FT_DOMAINS)),
#         sharex=True, squeeze=False,
#     )
#     for j, model in enumerate(models):
#         for i, ft in enumerate(FT_DOMAINS):
#             ax = axes[i][j]
#             sub = manifests[(manifests["model"] == model) & (manifests["ft_domain"] == ft)]
#             for ev, g in sub.groupby("eval_domain"):
#                 g = g.sort_values("step")
#                 ax.plot(g["step"].clip(lower=1), g["perplexity"],
#                         color=PROMPT_COLORS.get(ev, "gray"), marker="o", label=f"{ev} test")
#             ax.set_xscale("log")
#             ax.set_title(f"{MODEL_LABELS.get(model, model)} — {ft} fine-tune")
#             if i == len(FT_DOMAINS) - 1:
#                 ax.set_xlabel("training step (log)")
#             if j == 0:
#                 ax.set_ylabel("held-out perplexity")
#     axes[0][0].legend()
#     _save(fig, "fig1_perplexity", out_dir)
#     plt.close(fig)
#     return manifests
#


def sentence_surprisal_example(
    surp_versions: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    words: pd.DataFrame,
    text_id: str,
    sent_index: int = 0,
    slug: str = "german-gpt2",
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Fig 2 — one sentence of one PoTeC text (``text_id`` = "b0", "p2", …), 3×1 stack:

    1. baseline surprisal per word;
    2. Δ surprisal vs baseline, one line per DAPT checkpoint (fine-tune domain
       matches the text's domain);
    3. Δ surprisal vs baseline, one line per prompt (physics / biology / neutral).
    """
    sent = words[
        (words["text_id"] == text_id) & (words["sent_index_in_text"] == sent_index)
    ].sort_values("word_index_in_text")[WORD_KEY + ["word", "text_domain"]]
    if sent.empty:
        raise ValueError(f"no words for text_id={text_id!r} sent_index={sent_index}")
    domain = sent["text_domain"].iloc[0]

    def _surp(sel: pd.DataFrame, col: str) -> np.ndarray:
        m = sent.merge(sel.drop_duplicates(WORD_KEY), on=WORD_KEY, how="left")
        return m.sort_values("word_index_in_text")[col].to_numpy()

    base = _surp(surp_versions[surp_versions["index"] == 0], "surprisal")

    plt = _plt()
    x = np.arange(len(sent))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.45 * len(sent)), 9), sharex=True)

    axes[0].plot(x, base, color="k", marker="o")
    axes[0].set_title(f"baseline surprisal — {slug}, {text_id} sentence {sent_index}")
    axes[0].set_ylabel("surprisal (bits)")

    ckpts = surp_versions[
        (surp_versions["domain"] == domain) & (surp_versions["index"] > 0)
    ]
    for (_, step), g in sorted(ckpts.groupby(["index", "epoch"])):
        axes[1].plot(
            x, _surp(g, "surprisal") - base, marker="o", label=f"step {int(step)}"
        )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title(f"Δ surprisal per DAPT checkpoint ({domain} fine-tune)")
    axes[1].set_ylabel("Δ surprisal (bits)")
    axes[1].legend(fontsize=8)

    for cond, color in PROMPT_COLORS.items():
        col = f"s_prompt_{cond}"
        if col in prompt_surp.columns:
            axes[2].plot(
                x,
                _surp(prompt_surp, col) - base,
                color=color,
                marker="o",
                label=f"{cond} prompt",
            )
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Δ surprisal per prompt")
    axes[2].set_ylabel("Δ surprisal (bits)")
    axes[2].set_xticks(x, sent["word"], rotation=60, ha="right")

    _save(fig, f"fig2_sentence_{slug}_{text_id}", out_dir)
    plt.close(fig)
    return sent


def mean_surprisal_over_steps(
    surp_versions: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    words: pd.DataFrame,
    slug: str,
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Fig 3 — mean surprisal of the text-aligned model over all texts vs
    fine-tune step, plus intercept lines for the (step-independent) aligned and
    neutral prompts."""
    dom = words[WORD_KEY + ["text_domain"]].drop_duplicates(WORD_KEY)
    aligned = (
        surp_versions.merge(dom, on=WORD_KEY)
        .loc[lambda x: x["domain"] == x["text_domain"]]
        .groupby("epoch")["surprisal"]
        .mean()
        .reset_index()
    )

    p = prompt_surp.merge(dom, on=WORD_KEY)
    aligned_prompt = float(
        np.where(
            p["text_domain"] == "physics", p["s_prompt_physics"], p["s_prompt_biology"]
        ).mean()
    )

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        aligned["epoch"].clip(lower=1),
        aligned["surprisal"],
        color="tab:blue",
        marker="o",
        label="aligned model",
    )
    ax.axhline(aligned_prompt, color="tab:orange", ls="--", label="aligned prompt")
    if "s_prompt_neutral" in p.columns:
        ax.axhline(
            p["s_prompt_neutral"].mean(), color="gray", ls="--", label="neutral prompt"
        )
    ax.set_xscale("log")
    ax.set_xlabel("fine-tune step (log)")
    ax.set_ylabel("mean surprisal (bits)")
    ax.set_title(f"mean surprisal, all texts — {slug}")
    ax.legend()
    _save(fig, f"fig3_mean_surprisal_{slug}", out_dir)
    plt.close(fig)
    return aligned


def delta_ll_curves(results: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    """Fig 4 — ΔLL vs training step, one panel per LM (1×2).

    ``results`` is the model-comparison table (``model_lm``, ``model``,
    ``epoch``, ``delta_ll``). Checkpoint-dependent sources plot as curves over
    ``epoch``; checkpoint-independent ones (baseline / prompted / neutral, NA
    epoch) as intercept lines. y-limits span the global ΔLL range.
    """
    plt = _plt()
    slugs = _slugs(results, "model_lm")
    fig, axes = plt.subplots(
        1, len(slugs), figsize=(5.5 * len(slugs), 4), sharey=True, squeeze=False
    )
    lo, hi = results["delta_ll"].min(), results["delta_ll"].max()
    pad = 0.05 * (hi - lo) or 1.0
    for ax, slug in zip(axes[0], slugs):
        sub = results[results["model_lm"] == slug]
        for model, g in sub.groupby("model"):
            if g["epoch"].isna().all():  # checkpoint-independent: intercept only
                ax.axhline(g["delta_ll"].iloc[0], ls="--", label=model)
            else:
                g = g.dropna(subset=["epoch"]).sort_values("epoch")
                ax.plot(
                    g["epoch"].clip(lower=1), g["delta_ll"], marker="o", label=model
                )
        ax.set_xscale("log")
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("training step (log)")
        ax.set_title(MODEL_LABELS.get(slug, slug))
    axes[0][0].set_ylabel("ΔLL vs no-surprisal baseline")
    axes[0][0].legend(fontsize=8)
    _save(fig, "fig4_delta_ll", out_dir)
    plt.close(fig)
    return results


def rt_vs_surprisal(
    rm: pd.DataFrame,
    baseline_surp: pd.DataFrame,
    measure: str = "TFT",
    slug: str = "german-gpt2",
    n_bins: int = 20,
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Fig 5 — log reading time vs surprisal, baseline model only (GPT-2).

    ``baseline_surp``: WORD_KEY + ``surprisal`` (the un-adapted checkpoint).
    Scatter of all reader×word points plus quantile-binned means.
    """
    df = rm.merge(
        baseline_surp[WORD_KEY + ["surprisal"]].drop_duplicates(WORD_KEY), on=WORD_KEY
    )
    df = df[df[measure] > 0].assign(log_rt=lambda x: np.log(x[measure]))

    binned = df.groupby(
        pd.qcut(df["surprisal"], n_bins, duplicates="drop"), observed=True
    ).agg(surprisal=("surprisal", "mean"), log_rt=("log_rt", "mean"))

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["surprisal"], df["log_rt"], s=3, alpha=0.05, color="tab:blue")
    ax.plot(
        binned["surprisal"],
        binned["log_rt"],
        color="tab:red",
        marker="o",
        label=f"binned mean (n={len(binned)})",
    )
    ax.set_xlabel("surprisal (bits)")
    ax.set_ylabel(f"log {measure} (ms)")
    ax.set_title(f"log RT vs baseline surprisal — {slug}")
    ax.legend()
    _save(fig, f"fig5_logrt_surprisal_{slug}", out_dir)
    plt.close(fig)
    return df
