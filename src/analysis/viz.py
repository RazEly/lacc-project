"""Project visualizations. Home for every figure the pipeline produces.

Adaptation diagnostics — did DAPT actually move the model? (paper Figs 4–5).
Before reading anything into reading-time fits — especially a null result for
one LM — establish that domain adaptation happened at all:

* ``perplexity_curves``: held-out perplexity per checkpoint from the training
  manifests (per domain).
* ``surprisal_trajectories``: mean PoTeC surprisal per checkpoint, split by
  terminology level (common / T1 / T2) and in- vs out-of-domain text. Successful
  adaptation shows technical-term surprisal dropping on in-domain texts while
  common-word surprisal stays put.

Both write a CSV + PNG under ``figures/`` and return the tidy frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, WORD_KEY

FIGURES_DIR = PROJECT_ROOT / "figures"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def perplexity_curves(
    manifest: pd.DataFrame, slug: str, out_dir: Path = FIGURES_DIR
) -> pd.DataFrame:
    """Held-out perplexity vs training step, per domain."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = manifest[["domain", "index", "step", "perplexity"]].copy()
    df.to_csv(out_dir / f"diag_{slug}_perplexity.csv", index=False)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"physics": "tab:purple", "biology": "tab:green"}
    for domain, g in df.groupby("domain"):
        g = g.sort_values("index")
        ax.plot(
            g["step"].clip(lower=1),
            g["perplexity"],
            color=colors.get(domain, "gray"),
            lw=2,
            marker="o",
            label=domain,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("held-out perplexity")
    ax.set_title(f"DAPT perplexity — {slug}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"diag_{slug}_perplexity.png", dpi=150)
    plt.close(fig)
    return df


def surprisal_trajectories(
    surp_versions: pd.DataFrame,
    words: pd.DataFrame,
    slug: str,
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Mean PoTeC surprisal per checkpoint × terminology × in/out-of-domain.

    ``surp_versions`` is the long per-checkpoint table (both domains, index 0 =
    baseline); ``words`` the PoTeC word-features frame (terminology labels +
    ``text_domain``). x-axis is the checkpoint index (0 = baseline, then the 4ⁿ
    step schedule).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w = words[
        WORD_KEY
        + ["text_domain", "is_expert_technical_term", "is_general_technical_term"]
    ].drop_duplicates(WORD_KEY)
    df = surp_versions.merge(w, on=WORD_KEY, how="inner")
    df["terminology"] = np.select(
        [df["is_expert_technical_term"] == 1, df["is_general_technical_term"] == 1],
        ["technical T2", "technical T1"],
        default="common",
    )
    df["text_match"] = np.where(
        df["domain"] == df["text_domain"], "in-domain", "out-of-domain"
    )
    agg = (
        df.groupby(["domain", "index", "text_match", "terminology"])["surprisal"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )
    agg.to_csv(out_dir / f"diag_{slug}_surprisal.csv", index=False)

    plt = _plt()
    domains = sorted(agg["domain"].unique())
    matches = ["in-domain", "out-of-domain"]
    colors = {
        "common": "tab:blue",
        "technical T1": "tab:orange",
        "technical T2": "tab:red",
    }
    fig, axes = plt.subplots(
        len(matches),
        len(domains),
        figsize=(5 * len(domains), 3.2 * len(matches)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, match in enumerate(matches):
        for j, domain in enumerate(domains):
            ax = axes[i][j]
            sub = agg[(agg["domain"] == domain) & (agg["text_match"] == match)]
            for term, g in sub.groupby("terminology"):
                g = g.sort_values("index")
                ax.errorbar(
                    g["index"],
                    g["mean"],
                    yerr=g["sem"],
                    color=colors.get(term, "gray"),
                    marker="o",
                    label=term,
                )
            ax.set_title(f"{domain} LM — {match}")
            if i == len(matches) - 1:
                ax.set_xlabel("checkpoint index (0 = baseline, then 4ⁿ steps)")
            if j == 0:
                ax.set_ylabel("mean surprisal (bits)")
    axes[0][0].legend()
    fig.suptitle(f"PoTeC surprisal over DAPT — {slug}")
    fig.tight_layout()
    fig.savefig(out_dir / f"diag_{slug}_surprisal.png", dpi=150)
    plt.close(fig)
    return agg
