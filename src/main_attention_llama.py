"""Second-layer attention of LLäMmlein-1B (no fine-tuning): correlation + linear fit.

Paper 02 (Mouratidi & Poesio) finds the decoder's (Llama's) raw attention aligns
with gaze most at its **second layer** (μ≈0.4; the first layer is ~0/negative). This
script tests LLäMmlein-1B's second layer on PoTeC, expert vs novice readers:

  1. layer-wise Spearman correlation of raw attention with the 6 gaze features + PCA
     (layers 0 and 1);
  2. a **linear model fit** predicting each gaze target from text features plus the
     second layer's raw attention — adjusted R² on a 20% holdout, Wilcoxon test on
     squared errors vs the text-only baseline, |t| importance.

Compute saver: only the first two layers are needed, so the stack is **truncated to
`layers[:2]`** after load (forward never runs layers 2-23; the rest are freed). No
fine-tuning (index-0 baseline checkpoint, base weights from the HF cache). Surprisal
is omitted from the predictors — it would require a full-stack forward — so the
text-only set is word frequency, word length, and function/content category.

Run from the project root:

    uv run python -m src.main_attention_llama
"""
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config
from src.analysis import attention_prediction as ap
from src.analysis import correlation as co
from src.analysis import viz
from src.features import attention as at
from src.features import data
from src.features import surprisal as su

GROUPS = ("experts", "novices")
WORD_KEY = ["text_id", "word_index_in_text"]
SECOND_LAYER = 1  # 0-based: the paper's peak "second layer"
N_LAYERS = SECOND_LAYER + 1  # forward this many layers
# Text-only predictors WITHOUT surprisal (which would need a full-stack forward).
TEXT_FEATURES = ["log_word_freq", "log_word_length", "is_function_word"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"


def load_first_layers(name: str, n_layers: int):
    """Load a causal LM (eager attention) and keep only its first ``n_layers``.

    Truncating ``model.model.layers`` means the forward stops after those layers
    and returns ``n_layers`` attention matrices; the deeper layers are freed.
    """
    model, tok = su.load_causal_lm(name, attn=True)
    layers = model.model.layers
    model.model.layers = layers[:n_layers]
    model.config.num_hidden_layers = n_layers
    del layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tok


def text_features(rm: pd.DataFrame) -> pd.DataFrame:
    """Per-word text-only predictors (no surprisal): log freq, log length, function."""
    t = rm.drop_duplicates(WORD_KEY)[
        WORD_KEY + ["word_length", "lemma_frequency_normalized", "STTS_PoS_tag"]
    ].copy()
    t["log_word_freq"] = np.log1p(t["lemma_frequency_normalized"])
    t["log_word_length"] = np.log(t["word_length"].clip(lower=1))
    t["is_function_word"] = at.function_word_flag(t["STTS_PoS_tag"])
    return t[WORD_KEY + TEXT_FEATURES]


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ckpt = config.CHECKPOINTS_DIR / "LLaMmlein_1B_physics_lora" / "checkpoint_00"

    print("Loading PoTeC features")
    words = data.load_word_features()
    rm = data.load_reading_measures()
    et_by_group = {g: co.build_et_table(rm, participants=g) for g in GROUPS}
    txt = text_features(rm)

    print(f"Loading base LLäMmlein-1B from {ckpt} (first {N_LAYERS} layers)")
    model, tok = load_first_layers(str(ckpt), N_LAYERS)
    print("  n layers kept:", len(model.model.layers))

    print("Raw attention (layers 0-1)")
    attn_df = at.extract_attention(words, model, tok, method="raw")

    corr_rows, pred_rows, imp_rows = [], [], []
    for g in GROUPS:
        et = et_by_group[g]
        corr = co.correlate_attention(attn_df, et)
        corr.insert(0, "group", g)
        corr_rows.append(corr)

        # Linear-model fit using the SECOND layer's attention.
        attn_l2 = attn_df[attn_df["layer"] == SECOND_LAYER][WORD_KEY + ["attention"]]
        design = et.merge(txt, on=WORD_KEY).merge(attn_l2, on=WORD_KEY)
        for target in ap.TARGETS:
            if target not in design.columns:
                continue
            row, tvals = ap.attention_gain(design, target, features=TEXT_FEATURES)
            row.update({"group": g})
            pred_rows.append(row)
            for predictor, t in tvals.items():
                if predictor != "const":
                    imp_rows.append({"group": g, "target": target,
                                     "predictor": predictor, "abs_t": abs(t)})

    corr_df = pd.concat(corr_rows, ignore_index=True)
    corr_df.insert(0, "model", "llammlein-1b")
    pred_df = pd.DataFrame(pred_rows)
    imp_df = pd.DataFrame(imp_rows)
    corr_df.to_csv(PROJECT_ROOT / "results_attention_llama_correlation.csv", index=False)
    pred_df.to_csv(PROJECT_ROOT / "results_attention_llama_layer2_prediction.csv", index=False)

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n=== second-layer raw attention vs gaze (Spearman) ===")
    l2 = corr_df[corr_df["layer"] == SECOND_LAYER]
    print(l2[["group", "feature", "spearman", "p", "n"]].to_string(index=False))

    print("\n=== linear-model fit (gaze ~ text features ± second-layer attention) ===")
    print(pred_df[["group", "target", "adj_r2_text", "adj_r2_attn",
                   "delta_adj_r2", "wilcoxon_p", "n"]].to_string(index=False))

    print("\n=== mean |t| per predictor (second-layer model) ===")
    print(imp_df.groupby(["group", "predictor"])["abs_t"].mean().round(2).to_string())

    # Prediction bars per group (text-only vs +second-layer attention).
    for g in GROUPS:
        ps = pred_df[pred_df["group"] == g]
        fig, ax = plt.subplots()
        viz.attention_prediction_bars(ps, ax=ax)
        ax.set_title(f"LLäMmlein-1B second-layer attention — {g}")
        fig.tight_layout()
        out = FIG_DIR / f"attn_llama_layer2_prediction_{g}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
