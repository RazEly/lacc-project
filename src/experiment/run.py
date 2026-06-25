"""Driver for the encoder attention experiment (separate from src/main.py).

E1 fine-tune  — domain MLM DAPT of the encoder on data/domain_{phy,bio}, runs
                FIRST; checkpoints saved under artifacts/ (config.CHECKPOINTS_DIR).
E2 raw        — raw attention (paper §3.2.1) vs gaze across the checkpoints:
                x = tokens seen, y = Spearman r, per reader group × domain.
E3 flow       — attention flow (§3.2.2) vs gaze, same tokens-vs-correlation view.
                Expect alignment to rise with tokens seen as the encoder adapts.

Run from the project root:

    python -m src.experiment.run

Flow is per-token max-flow per layer per sentence — slow. Use ``max_sents`` /
small ``epochs`` / ``max_docs`` for a quick check; drop them for a real run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import CHECKPOINTS_DIR, ENCODER_MODEL
from src.experiment import analysis as ea
from src.experiment import encoder as enc
from src.experiment import viz as ev
from src.features import data
from src.modeling import finetune as ft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "figures"
MANIFEST_CSV = PROJECT_ROOT / "results_finetune_manifest.csv"


def _checkpoints_ready(manifest: pd.DataFrame) -> bool:
    """True when every checkpoint the manifest indexes is on disk.

    save_model writes config.json into each checkpoint dir, so we treat that as
    the "weights saved" sentinel — a manifest pointing at a missing/partial dir
    forces a re-run rather than a load that would fail downstream.
    """
    return len(manifest) > 0 and all(
        (Path(p) / "config.json").exists() for p in manifest["checkpoint"]
    )


def _save_fig(ax, name: str) -> None:
    fig = ax.get_figure()
    fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def _subset_sents(words, max_sents):
    """Keep the first ``max_sents`` sentences (smoke testing)."""
    if not max_sents:
        return words
    keys = (
        words[["text_id", "sent_index_in_text"]]
        .drop_duplicates()
        .head(max_sents)
    )
    return words.merge(keys, on=["text_id", "sent_index_in_text"])


def main(max_sents=None, max_tokens: int | None = None, n_checkpoints: int = 4, max_docs=None):
    FIG_DIR.mkdir(exist_ok=True)
    words = _subset_sents(data.load_word_features(), max_sents)
    rm_raw = data.load_reading_measures()

    # ── E1 — domain MLM fine-tuning of the encoder (runs FIRST) ───────────────
    # Checkpoints are saved under artifacts/ (config.CHECKPOINTS_DIR) by
    # finetune_dapt; the manifest indexes them. Raw/flow alignment is then read
    # off these checkpoints as a function of tokens seen, so DAPT must precede it.
    # Skip the (expensive) DAPT when a prior run already saved the checkpoints:
    # reuse them iff the manifest exists and every checkpoint dir is on disk.
    if MANIFEST_CSV.exists() and _checkpoints_ready(manifest := pd.read_csv(MANIFEST_CSV)):
        print(f"E1 — reusing {len(manifest)} existing checkpoints under "
              f"{CHECKPOINTS_DIR}/ (skip DAPT)")
    else:
        print("E1 — domain MLM DAPT (saving checkpoints to artifacts/)")
        # Budget by tokens, not epochs: equal token exposure across domains. Default
        # to the largest budget that fits one pass of each corpus.
        budget = max_tokens or min(
            ft.count_domain_tokens("physics", ENCODER_MODEL, objective="mlm", max_docs=max_docs),
            ft.count_domain_tokens("biology", ENCODER_MODEL, objective="mlm", max_docs=max_docs),
        )
        manifest = pd.concat(
            [
                ft.finetune_dapt("physics", ENCODER_MODEL, objective="mlm",
                                 max_tokens=budget, n_checkpoints=n_checkpoints,
                                 batch_size=8, max_docs=max_docs),
                ft.finetune_dapt("biology", ENCODER_MODEL, objective="mlm",
                                 max_tokens=budget, n_checkpoints=n_checkpoints,
                                 batch_size=8, max_docs=max_docs),
            ],
            ignore_index=True,
        )
        manifest.to_csv(MANIFEST_CSV, index=False)
        print(f"  checkpoints under {CHECKPOINTS_DIR}/ — {len(manifest)} saved")

    # ── E2/E3 — raw then flow alignment across fine-tuning (x = tokens seen) ───
    # Same analysis for both methods (paper order: raw §3.2.1, then flow §3.2.2).
    # Each correlates its checkpoint against its own domain's texts, per reader
    # group; we expect Spearman r to climb with tokens seen.
    for tag, method, fig_name in [
        ("E2", "raw", "exp_raw_finetune_tokens"),
        ("E3", "flow", "exp_flow_finetune_tokens"),
    ]:
        print(f"{tag} — {method} vs gaze across fine-tuning (tokens seen)")
        versions = enc.attention_over_checkpoints(words, manifest, method=method)
        # Method 1 — correlation (paper §3.3): peak-layer Spearman vs gaze PCA.
        curve = ea.correlation_over_checkpoints(versions, rm_raw, method=method)
        print(curve.to_string(index=False))
        curve.to_csv(PROJECT_ROOT / f"results_attention_{method}.csv", index=False)
        fig, ax = plt.subplots()
        ev.expert_novice_finetune_curve(curve, ax=ax, method=method)
        _save_fig(ax, fig_name)

        # Method 2 — predictive modeling (paper §3.3): held-out adjusted R²,
        # does attention predict gaze beyond word frequency/length?
        reg = ea.regression_over_checkpoints(versions, rm_raw, method=method)
        print(reg.to_string(index=False))
        reg.to_csv(PROJECT_ROOT / f"results_regression_{method}.csv", index=False)
        fig, ax = plt.subplots()
        ev.expert_novice_regression_curve(reg, ax=ax, method=method)
        _save_fig(ax, f"exp_{method}_regression_tokens")

    print(f"Done. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
