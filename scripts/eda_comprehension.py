"""EDA: PoTeC comprehension / background-knowledge accuracy by reader domain.

Question driving the main figure: on BIOLOGY texts, of the readers who scored
each accuracy level (0.0, 0.1, … 1.0), what is the biology- vs physics-student
split — and the same on PHYSICS texts. PoTeC asks 3 text-comprehension (TQ) and 3
background-knowledge (BQ) questions per text, so the per-reader×text accuracy is
effectively one of {0, ⅓, ⅔, 1}; we round to the nearest 0.1 decile for binning.

All figures + a summary CSV are written to ``figures/`` / project root. Run:

    uv run python -m scripts.eda_comprehension
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
DECILES = np.round(np.arange(0.0, 1.01, 0.1), 1)  # 0.0, 0.1, … 1.0
MAJORS = {0: "biology students", 1: "physics students"}
MAJOR_COLORS = {0: "#2ca02c", 1: "#1f77b4"}


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def load_reader_text_accuracy() -> pd.DataFrame:
    """One row per reader×text: comprehension (TQ) + background (BQ) accuracy.

    The reading-measures table repeats the per-text accuracy on every word, so we
    take the first row per (reader, text). Adds ``major`` (reader discipline) and
    ``is_expert`` (major == text domain).
    """
    rm = data.load_reading_measures()
    key = ["reader_id", "text_id", "text_domain",
           "reader_discipline_numeric", "text_domain_numeric"]
    t = rm.groupby(key, as_index=False)[["mean_acc_tq", "mean_acc_bq"]].first()
    t["is_expert"] = (
        t["reader_discipline_numeric"] == t["text_domain_numeric"]
    ).astype(int)
    t["major"] = t["reader_discipline_numeric"].map(MAJORS)
    return t


def decile_distribution(sub: pd.DataFrame, score: str) -> pd.DataFrame:
    """Per-major decile distribution of ``score`` for one text domain.

    Returns a DataFrame indexed by decile (0.0…1.0) with one column per reader
    major, normalised WITHIN each major (column sums to 1) so the two majors'
    accuracy shapes are comparable despite unequal group sizes.
    """
    d = sub.copy()
    d["decile"] = np.round(d[score], 1)
    counts = (
        d.groupby(["reader_discipline_numeric", "decile"]).size()
        .unstack("reader_discipline_numeric", fill_value=0)
        .reindex(DECILES, fill_value=0)
    )
    prop = counts.div(counts.sum(axis=0), axis=1)  # normalise within major
    return prop


def plot_decile_by_major(acc: pd.DataFrame, score: str, title: str, fname: str):
    """Grouped decile bars per reader major, side by side for bio & physics texts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (domain_num, domain_name) in zip(axes, [(0, "biology"), (1, "physics")]):
        sub = acc[acc["text_domain_numeric"] == domain_num]
        prop = decile_distribution(sub, score)
        x = np.arange(len(DECILES))
        width = 0.4
        for i, major_num in enumerate((0, 1)):
            vals = prop.get(major_num, pd.Series(0, index=DECILES))
            ax.bar(x + (i - 0.5) * width, vals.values, width,
                   label=MAJORS[major_num], color=MAJOR_COLORS[major_num])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:.1f}" for d in DECILES], rotation=0, fontsize=8)
        ax.set_title(f"{domain_name} texts")
        ax.set_xlabel(f"{score.replace('mean_acc_', '').upper()} accuracy")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("proportion within major")
    axes[0].legend()
    fig.suptitle(title)
    save_fig(fig, fname)


def plot_major_share(acc: pd.DataFrame, score: str, title: str, fname: str):
    """Per-decile reader-major SHARE: of all readings at this accuracy on a
    domain's texts, what fraction came from each major (stacked to 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (domain_num, domain_name) in zip(axes, [(0, "biology"), (1, "physics")]):
        sub = acc[acc["text_domain_numeric"] == domain_num].copy()
        sub["decile"] = np.round(sub[score], 1)
        counts = (
            sub.groupby(["decile", "reader_discipline_numeric"]).size()
            .unstack("reader_discipline_numeric", fill_value=0)
            .reindex(DECILES, fill_value=0)
        )
        total = counts.sum(axis=1).replace(0, np.nan)
        share = counts.div(total, axis=0).fillna(0)
        x = np.arange(len(DECILES))
        bottom = np.zeros(len(DECILES))
        for major_num in (0, 1):
            vals = share.get(major_num, pd.Series(0, index=DECILES)).values
            ax.bar(x, vals, bottom=bottom, label=MAJORS[major_num],
                   color=MAJOR_COLORS[major_num])
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:.1f}" for d in DECILES], fontsize=8)
        ax.set_title(f"{domain_name} texts")
        ax.set_xlabel(f"{score.replace('mean_acc_', '').upper()} accuracy")
    axes[0].set_ylabel("reader-major share")
    axes[0].legend(loc="lower left")
    fig.suptitle(title)
    save_fig(fig, fname)


def plot_mean_grid(acc: pd.DataFrame, fname: str):
    """Mean TQ & BQ accuracy by reader major × text domain (PoTeC Table-2 style)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, score in zip(axes, ["mean_acc_tq", "mean_acc_bq"]):
        piv = acc.pivot_table(index="reader_discipline_numeric",
                              columns="text_domain_numeric",
                              values=score, aggfunc="mean")
        x = np.arange(2)  # text domains: 0 bio, 1 phys
        width = 0.4
        for i, major_num in enumerate((0, 1)):
            vals = [piv.loc[major_num, 0], piv.loc[major_num, 1]]
            ax.bar(x + (i - 0.5) * width, vals, width,
                   label=MAJORS[major_num], color=MAJOR_COLORS[major_num])
        ax.set_xticks(x)
        ax.set_xticklabels(["biology texts", "physics texts"])
        ax.set_title(score.replace("mean_acc_", "").upper()
                     + (" (text comprehension)" if "tq" in score else " (background)"))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mean accuracy")
    axes[0].legend()
    fig.suptitle("Mean accuracy by reader major × text domain")
    save_fig(fig, fname)


def plot_expert_novice_box(acc: pd.DataFrame, fname: str):
    """TQ & BQ accuracy: in-domain (expert) vs out-of-domain (novice) readings."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, score in zip(axes, ["mean_acc_tq", "mean_acc_bq"]):
        groups = [acc.loc[acc["is_expert"] == 1, score],
                  acc.loc[acc["is_expert"] == 0, score]]
        ax.boxplot(groups, tick_labels=["expert\n(in-domain)", "novice\n(out-domain)"],
                   showmeans=True)
        ax.set_title(score.replace("mean_acc_", "").upper())
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Expert vs novice accuracy")
    save_fig(fig, fname)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    acc = load_reader_text_accuracy()

    n_readers = acc.groupby("reader_discipline_numeric")["reader_id"].nunique()
    print(f"reader×text rows={len(acc)}  "
          f"bio-students={n_readers.get(0)} phys-students={n_readers.get(1)}")
    print("\nMean accuracy by major × text domain:")
    for score in ["mean_acc_tq", "mean_acc_bq"]:
        print(f"  {score}:")
        print(acc.pivot_table(index="major", columns="text_domain",
                              values=score, aggfunc="mean").round(3).to_string())

    # ── Requested core: decile distribution by major, per text domain ─────────
    plot_decile_by_major(
        acc, "mean_acc_tq",
        "Text-comprehension (TQ) accuracy distribution by reader major",
        "eda_comprehension_tq_decile_by_major")
    plot_decile_by_major(
        acc, "mean_acc_bq",
        "Background-knowledge (BQ) accuracy distribution by reader major",
        "eda_comprehension_bq_decile_by_major")
    # The literal bio/phys ratio at each accuracy level (stacked share).
    plot_major_share(
        acc, "mean_acc_tq",
        "Reader-major share at each TQ accuracy level",
        "eda_comprehension_tq_major_share")

    # ── Supporting EDA ────────────────────────────────────────────────────────
    plot_mean_grid(acc, "eda_comprehension_mean_grid")
    plot_expert_novice_box(acc, "eda_comprehension_expert_vs_novice")

    # ── Summary table (counts + within-major proportions) ─────────────────────
    rows = []
    for score in ["mean_acc_tq", "mean_acc_bq"]:
        for domain_num, domain in [(0, "biology"), (1, "physics")]:
            sub = acc[acc["text_domain_numeric"] == domain_num].copy()
            sub["decile"] = np.round(sub[score], 1)
            for major_num, major in MAJORS.items():
                g = sub[sub["reader_discipline_numeric"] == major_num]
                vc = g["decile"].value_counts().reindex(DECILES, fill_value=0)
                tot = vc.sum()
                for dec in DECILES:
                    rows.append({
                        "score": score.replace("mean_acc_", ""),
                        "text_domain": domain, "major": major,
                        "decile": dec, "n": int(vc[dec]),
                        "prop_within_major": round(vc[dec] / tot, 4) if tot else 0.0,
                    })
    summary = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results_eda_comprehension.csv"
    summary.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    print(f"\nDone. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
