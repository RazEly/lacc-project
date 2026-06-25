"""Driver for the encoder attention experiment (separate from src/main.py).

E1 reproduce  — German BERT attention flow vs gaze, per reader group (the paper's
                strongest result, expertise-split).
E2 fine-tune  — domain MLM DAPT of the encoder on data/domain_{phy,bio}.
E3 compare    — flow↔gaze alignment across fine-tuning, experts vs novices.

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

from src.config import ENCODER_MODEL
from src.experiment import analysis as ea
from src.experiment import encoder as enc
from src.experiment import viz as ev
from src.features import data
from src.modeling import finetune as ft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "figures"


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

    # ── E1 — reproduce: baseline encoder flow vs gaze, per reader group ───────
    print("E1 — baseline encoder attention flow vs gaze")
    model, tok = enc.load_encoder(ENCODER_MODEL)
    flow = enc.extract_flow(words, model, tok)
    corr = ea.correlate_flow_by_group(flow, rm_raw, domain="all")
    peak = corr.loc[corr.groupby("group")["spearman"].idxmax()]
    print(peak[["group", "layer", "spearman"]].to_string(index=False))
    fig, ax = plt.subplots()
    ev.expert_novice_layer_curve(corr, ax=ax)
    _save_fig(ax, "exp_flow_layers_expert_novice")

    # ── E2 — domain MLM fine-tuning of the encoder ────────────────────────────
    print("E2 — domain MLM DAPT")
    # Budget by tokens, not epochs: equal token exposure across domains. Default to
    # the largest budget that fits one pass of each corpus.
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

    # ── E3 — flow alignment across fine-tuning, experts vs novices ────────────
    print("E3 — flow vs gaze across fine-tuning")
    flow_versions = enc.flow_over_checkpoints(words, manifest)
    curve = ea.flow_correlation_over_checkpoints(flow_versions, rm_raw)
    print(curve.to_string(index=False))
    curve.to_csv(PROJECT_ROOT / "results_attention.csv", index=False)
    fig, ax = plt.subplots()
    ev.expert_novice_finetune_curve(curve, ax=ax)
    _save_fig(ax, "exp_flow_finetune_expert_novice")

    print(f"Done. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
