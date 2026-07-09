"""Project figures (see viz.md). Every function saves a PNG under ``figures/``
(never ``.show()``) and returns the plotted frame.

fig1  perplexity_grid            held-out perplexity vs step, 2×2 (model × ft-domain); optional PoTeC-stimuli lines
fig2  sentence_surprisal_overlay one sentence, one panel: raw surprisal — baseline, one checkpoint, one prompt
fig3  mean_surprisal_over_steps  mean surprisal vs step, 2×2 (text domain × ft-domain); prompted arms as intercepts; optional expert-term-only filter
fig4  delta_ll_curves            ΔLL vs step per LM; checkpoint-independent arms as intercepts
fig5  rt_vs_surprisal            log RT vs baseline-GPT-2 surprisal

"aligned" here is text-level (checkpoint / prompt domain matches ``text_domain``);
the reader-level alignment lives in the mixed models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DAPT_CHECKPOINT_STEPS, PROJECT_ROOT, WORD_KEY

FIGURES_DIR = PROJECT_ROOT / "figures"
# checkpoint index -> optimiser step; index 0 is the un-fine-tuned base model.
# Surprisal tables carry index/epoch only, and epoch is NOT comparable across
# domains (corpus sizes differ), so figures plot by index mapped to step.
STEP_OF_INDEX = {i: s for i, s in enumerate([0, *DAPT_CHECKPOINT_STEPS])}
MODEL_LABELS = {"german-gpt2": "GPT-2", "llammlein-1b": "Llama 1B"}
FT_DOMAINS = ("biology", "physics")  # fig1 row order
# Okabe-Ito colorblind-safe palette; also survives grayscale print. Domains are
# additionally distinguished by linestyle/marker so color is never load-bearing.
OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)
PROMPT_COLORS = {"physics": "#CC79A7", "biology": "#009E73", "neutral": "#999999"}
# fig4 checkpoint-independent intercepts (curves use the default cycle). The
# neutral fixed-domain prompt pseudo-test reuses the PROMPT_COLORS palette.
INTERCEPT_COLORS = {
    "baseline": "#000000",
    "prompted": "#E69F00",
    "prompt_neutral": "#999999",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Publication defaults: larger fonts (matplotlib's ~10pt renders tiny in a
    # two-column figure), colorblind-safe default cycle, 300-dpi tight PNGs.
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        }
    )
    return plt


def _panel_letters(axes) -> None:
    """Tag each panel (a), (b), … top-left, for caption cross-reference."""
    for k, ax in enumerate(np.asarray(axes).flat):
        ax.text(
            -0.08,
            1.04,
            f"({chr(97 + k)})",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
            ha="right",
        )


def _shared_legend(fig, axes, ncol: int = 4) -> None:
    """One figure-level legend (deduped by label) below the grid, replacing the
    per-panel legends that otherwise repeat the same entries in every axis."""
    handles: list = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in np.asarray(axes).flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen.add(lab)
                handles.append(h)
                labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )


def _grid(plt, nrows: int, ncols: int, **kwargs):
    """2-D subplot grid, ~5×3.5in per panel; always returns a 2-D axes array."""
    return plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False, **kwargs
    )


def _save(fig, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png")


def _step_ticks(ax, steps) -> None:
    """Label a log step-axis with the fixed training-step counts themselves.

    Step 0 (base model) is drawn at x=1 (log axis can't show 0) but keeps its
    "0" label. Ticks are the discrete checkpoint steps, not auto log decades.
    """
    vals = sorted({int(s) for s in steps})
    ax.set_xticks([max(s, 1) for s in vals], [str(s) for s in vals])
    ax.minorticks_off()


def _slugs(df: pd.DataFrame, col: str) -> list[str]:
    """Model slugs in ``col``, ordered like MODEL_LABELS (unknowns appended)."""
    present = set(df[col])
    return [s for s in MODEL_LABELS if s in present] + sorted(
        present - set(MODEL_LABELS)
    )


def perplexity_grid(
    manifests: pd.DataFrame,
    stimuli: pd.DataFrame | None = None,
    show_stimuli: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Fig 1 — held-out perplexity vs training step.

    2×2 grid: columns = LM (GPT-2 / Llama), rows = fine-tune domain (biology /
    physics); one line per held-out test set. ``manifests`` is long with columns
    ``model``, ``ft_domain``, ``eval_domain``, ``step``, ``perplexity``. In-domain
    (``eval_domain == ft_domain``) draws solid, out-of-domain dotted; manifests
    from runs before the cross-domain eval carry the in-domain line only.

    ``stimuli`` (optional, drawn when given unless ``show_stimuli=False``) adds
    two dotted square-marker lines per panel: the checkpoint's perplexity on the
    PoTeC stimulus texts of each domain (``surprisal.stimuli_perplexity`` output,
    columns ``model``, ``ft_domain``, ``index``, ``text_domain``, ``perplexity``).
    Colors follow the eval domain; markers separate corpus (o = held-out
    Wikipedia split, s = PoTeC stimuli).
    """
    plt = _plt()
    models = _slugs(manifests, "model")
    fig, axes = _grid(plt, len(FT_DOMAINS), len(models), sharex=True, sharey=True)
    for j, model in enumerate(models):
        for i, ft in enumerate(FT_DOMAINS):
            ax = axes[i, j]
            sub = manifests[
                (manifests["model"] == model) & (manifests["ft_domain"] == ft)
            ]
            for ev, g in sub.groupby("eval_domain"):
                g = g.sort_values("step")
                in_domain = ev == ft
                ax.plot(
                    g["step"].clip(lower=1),
                    g["perplexity"],
                    color=PROMPT_COLORS.get(ev, "gray"),
                    marker="o",
                    ls="-" if in_domain else ":",
                    label=f"{ev} test" + ("" if in_domain else " (out-of-domain)"),
                )
            tick_steps = sub["step"]
            if show_stimuli and stimuli is not None:
                st = stimuli[
                    (stimuli["model"] == model) & (stimuli["ft_domain"] == ft)
                ].assign(step=lambda x: x["index"].astype(int).map(STEP_OF_INDEX))
                for td, g in st.groupby("text_domain"):
                    g = g.sort_values("step")
                    ax.plot(
                        g["step"].clip(lower=1),
                        g["perplexity"],
                        color=PROMPT_COLORS.get(td, "gray"),
                        marker="s",
                        ls=":",
                        ms=4,
                        label=f"{td} PoTeC stimuli"
                        + ("" if td == ft else " (out-of-domain)"),
                    )
                tick_steps = pd.concat([tick_steps, st["step"]])
            ax.set_xscale("log")
            if len(tick_steps):
                _step_ticks(ax, tick_steps)
            # column header = LM (top row only); row header folded into the
            # leftmost ylabel — avoids repeating both in all four panel titles.
            if i == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            if i == len(FT_DOMAINS) - 1:
                ax.set_xlabel("training steps (log)")
            if j == 0:
                # stimuli lines are not a held-out split — generalise the label.
                ybase = (
                    "perplexity"
                    if show_stimuli and stimuli is not None
                    else "held-out perplexity"
                )
                ax.set_ylabel(f"{ft} fine-tune\n{ybase}")
    _panel_letters(axes)
    _shared_legend(fig, axes)
    _save(fig, "fig1_perplexity", out_dir)
    plt.close(fig)
    return manifests


def sentence_surprisal_overlay(
    surp_versions: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    words: pd.DataFrame,
    text_id: str,
    sent_index: int = 0,
    step: int = 4096,
    slug: str = "german-gpt2",
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Fig 2 — one sentence, one panel, raw surprisal (not Δ): three overlaid lines
    — baseline, the ``step`` DAPT checkpoint (fine-tune domain = text domain), and
    the domain-matching prompt. Word labels of PoTeC expert terms are drawn in red.
    """
    sent = words[
        (words["text_id"] == text_id) & (words["sent_index_in_text"] == sent_index)
    ].sort_values("word_index_in_text")[
        WORD_KEY + ["word", "text_domain", "is_expert_technical_term"]
    ]
    if sent.empty:
        raise ValueError(f"no words for text_id={text_id!r} sent_index={sent_index}")
    domain = sent["text_domain"].iloc[0]
    index_of_step = {s: i for i, s in STEP_OF_INDEX.items()}
    if step not in index_of_step:
        raise ValueError(f"step {step} not a checkpoint; have {sorted(index_of_step)}")

    def _surp(sel: pd.DataFrame, col: str) -> np.ndarray:
        m = sent.merge(sel.drop_duplicates(WORD_KEY), on=WORD_KEY, how="left")
        return m.sort_values("word_index_in_text")[col].to_numpy()

    plt = _plt()
    x = np.arange(len(sent))
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(sent)), 4.5))

    ax.plot(
        x,
        _surp(surp_versions[surp_versions["index"] == 0], "surprisal"),
        color="k",
        marker="o",
        label="baseline",
    )
    ckpt = surp_versions[
        (surp_versions["domain"] == domain)
        & (surp_versions["index"] == index_of_step[step])
    ]
    ax.plot(
        x,
        _surp(ckpt, "surprisal"),
        color="tab:blue",
        marker="o",
        label=f"{step} training steps ({domain} fine-tune)",
    )
    col = f"s_prompt_{domain}"
    if col in prompt_surp.columns:
        ax.plot(
            x,
            _surp(prompt_surp, col),
            color=PROMPT_COLORS.get(domain, "tab:orange"),
            marker="o",
            label=f"{domain} prompt",
        )
    ax.set_title(f"surprisal — {slug}, {text_id} sentence {sent_index}")
    ax.set_ylabel("surprisal (bits)")
    ax.set_xticks(x, sent["word"], rotation=60, ha="right")
    # PoTeC expert technical terms in red (all other words stay black).
    for lab, is_term in zip(ax.get_xticklabels(), sent["is_expert_technical_term"]):
        if is_term == 1:
            lab.set_color("red")
    # Shade expert-term columns (redundant to the red labels for grayscale print).
    for k, is_term in enumerate(sent["is_expert_technical_term"].to_numpy()):
        if is_term == 1:
            ax.axvspan(k - 0.5, k + 0.5, color="0.9", zorder=0)
    ax.legend()

    _save(fig, f"fig2_sentence_overlay_{slug}_{text_id}", out_dir)
    plt.close(fig)
    return sent


def mean_surprisal_over_steps(
    bundles: list[dict],
    words: pd.DataFrame,
    expert_terms_only: bool = False,
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Fig 3 — mean surprisal vs fine-tune step, 2×2 grid: rows = stimulus text
    domain (physics top, biology bottom), columns = LM (one per bundle, e.g.
    GPT-2 / Llama). Each panel overlays BOTH fine-tune domains on that stimulus
    set — the in-domain fine-tune (model domain = stimulus domain) solid, the
    out-of-domain fine-tune dotted — plus intercept lines for the neutral
    fixed-domain prompt pseudo-test (dashed) and the domain-matched prompt
    (dotted, e.g. physics prompt on physics stimuli). ``expert_terms_only`` restricts the mean to
    words annotated as expert technical terms in PoTeC."""
    w = words[WORD_KEY + ["text_domain", "is_expert_technical_term"]].drop_duplicates(
        WORD_KEY
    )
    if expert_terms_only:
        w = w[w["is_expert_technical_term"] == 1]
    dom = w[WORD_KEY + ["text_domain"]]

    model_domains = ("physics", "biology")  # fine-tune domains overlaid per panel
    text_domains = ("physics", "biology")  # grid rows
    # in-/out-of-domain roles get their own colour+style so the shared legend
    # stays consistent across panels (label no longer names a fixed domain).
    ROLE = {
        True: ("#0072B2", "-", "in-domain fine-tune"),
        False: ("#D55E00", ":", "out-of-domain fine-tune"),
    }
    plt = _plt()
    # sharey="col": each model keeps its own surprisal scale (GPT-2 and Llama
    # sit at different absolute bits) while its two stimulus rows stay comparable.
    fig, axes = _grid(
        plt, len(text_domains), len(bundles), sharex=True, sharey="col"
    )
    all_means: list[pd.DataFrame] = []
    for j, b in enumerate(bundles):
        # group by index, not epoch: epoch differs across domains at the same
        # checkpoint, so an epoch groupby would split the paired domains apart.
        means = (
            b["surp_versions"].merge(dom, on=WORD_KEY)
            .groupby(["domain", "text_domain", "index"])["surprisal"]
            .mean()
            .reset_index()
            .assign(
                step=lambda x: x["index"].astype(int).map(STEP_OF_INDEX),
                model=b["slug"],
            )
        )
        all_means.append(means)
        p = b["prompt_surp"].merge(dom, on=WORD_KEY)
        for i, td in enumerate(text_domains):
            ax = axes[i, j]
            for md in model_domains:
                g = means[
                    (means["domain"] == md) & (means["text_domain"] == td)
                ].sort_values("step")
                color, ls, label = ROLE[md == td]
                ax.plot(
                    g["step"].clip(lower=1),  # step 0 (base model) shown at x=1
                    g["surprisal"],
                    color=color,
                    ls=ls,
                    marker="o",
                    label=label,
                )
                if len(g):
                    _step_ticks(ax, g["step"])
            pt = p[p["text_domain"] == td]
            # the neutral fixed-domain prompt pseudo-test as an intercept reference.
            if "s_prompt_neutral" in pt.columns and len(pt):
                ax.axhline(
                    pt["s_prompt_neutral"].mean(),
                    color=PROMPT_COLORS["neutral"],
                    ls="--",
                    label="neutral prompt",
                )
            # the domain-matched prompt (physics prompt on physics stimuli, etc.)
            # as a dotted intercept — color follows the stimulus domain.
            aligned_col = f"s_prompt_{td}"
            if aligned_col in pt.columns and len(pt):
                ax.axhline(
                    pt[aligned_col].mean(),
                    color=PROMPT_COLORS.get(td, "tab:orange"),
                    ls=":",
                    label=f"{td} prompt",
                )
            ax.set_xscale("log")
            # column header = model (top row); stimulus domain folded into the
            # leftmost ylabel; scope goes to the suptitle.
            if i == 0:
                ax.set_title(MODEL_LABELS.get(b["slug"], b["slug"]))
            if i == len(text_domains) - 1:
                ax.set_xlabel("training steps (log)")
            if j == 0:
                ax.set_ylabel(f"{td} stimuli\nmean surprisal (bits)")
    scope = "expert terms" if expert_terms_only else "all words"
    # y>1 keeps the suptitle clear of the top-row (a) letters / column titles.
    fig.suptitle(f"mean surprisal — {scope}", y=1.02)
    _panel_letters(axes)
    _shared_legend(fig, axes, ncol=3)
    suffix = "_expert-terms" if expert_terms_only else ""
    _save(fig, f"fig3_mean_surprisal{suffix}", out_dir)
    plt.close(fig)
    return pd.concat(all_means, ignore_index=True)


def delta_ll_curves(results: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    """Fig 4 — ΔLL vs training step, one panel per LM (1×2).

    ``results`` is the model-comparison table (``model_lm``, ``model``,
    ``index``, ``delta_ll``). Checkpoint-dependent sources plot as curves over
    the step of each checkpoint index; checkpoint-independent ones (baseline /
    prompted / prompt_neutral, NA index) as intercept lines.
    y-limits span the global ΔLL range.
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
            if g["index"].isna().all():  # checkpoint-independent: intercept only
                ax.axhline(
                    g["delta_ll"].iloc[0],
                    ls="--",
                    color=INTERCEPT_COLORS.get(model, "tab:brown"),
                    label=model,
                )
            else:
                g = g.dropna(subset=["index"]).sort_values("index")
                steps = g["index"].astype(int).map(STEP_OF_INDEX)
                ax.plot(steps.clip(lower=1), g["delta_ll"], marker="o", label=model)
        ax.set_xscale("log")
        dep_idx = sub["index"].dropna().astype(int)
        if len(dep_idx):
            _step_ticks(ax, dep_idx.map(STEP_OF_INDEX))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("training steps (log)")
        ax.set_title(MODEL_LABELS.get(slug, slug))
    axes[0, 0].set_ylabel("ΔLL vs no-surprisal baseline")
    axes[0, 0].legend()
    _panel_letters(axes)
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

    # OLS fit (linear in log RT) — the surprisal main effect the mixed model uses.
    coef = np.polyfit(df["surprisal"], df["log_rt"], 1)
    xs = np.linspace(df["surprisal"].min(), df["surprisal"].max(), 100)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    # rasterize the dense point cloud only (tens of thousands of reader×word
    # points) — keeps the PNG small and axes/text crisp.
    ax.scatter(
        df["surprisal"], df["log_rt"], s=3, alpha=0.05, color="#0072B2", rasterized=True
    )
    ax.plot(
        xs,
        np.polyval(coef, xs),
        color="k",
        lw=1.5,
        label=f"OLS fit (slope={coef[0]:.3f})",
    )
    ax.plot(
        binned["surprisal"],
        binned["log_rt"],
        color="#D55E00",
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
