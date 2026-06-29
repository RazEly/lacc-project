"""Diagnostic: does including self-attention crush the attention-gaze correlation?

Paper 02 scores each word by the attention it receives "from all other words"
(excludes self). Our raw_attention averages queries q>=k, INCLUDING the diagonal
A[k,k] (self). This compares both per-layer against gaze PCA on the cheap cached
german-gpt2 baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.analysis import correlation as co
from src.features import attention as at
from src.features import data
from src.features import surprisal as su

WORD_KEY = ["text_id", "word_index_in_text"]


def recv_variants(A, seq):
    """Per-token received attention: with-self (q>=k) vs others-only (q>k)."""
    with_self = np.array([A[k:, k].mean() for k in range(seq)])
    others = np.array([A[k + 1 :, k].mean() if k + 1 < seq else np.nan
                       for k in range(seq)])
    return with_self, others


def per_word(scores, positions):
    return np.array([scores[p].max() if p else np.nan for p in positions])


def main():
    words = data.load_word_features()
    rm = data.load_reading_measures()
    et = co.build_et_table(rm, participants="all")[WORD_KEY + ["pca"]]

    ckpt = config.CHECKPOINTS_DIR / "german-gpt2_physics_lora" / "checkpoint_00"
    model, tok = su.load_causal_lm(str(ckpt), attn=True)

    sents = [s for _, s in words.sort_values(
        ["text_id", "sent_index_in_text", "word_index_in_sent"]
    ).groupby(["text_id", "sent_index_in_text"], sort=False) if len(s) >= 3]

    rows = []
    for s in sents:
        wl = s["word"].fillna("").astype(str).tolist()
        atts, word_ids = at._forward_attentions(wl, model, tok)
        n_layers, seq, _ = atts.shape
        pos = at._word_positions(word_ids, len(wl))
        idx = s["word_index_in_text"].tolist()
        tid = s["text_id"].iloc[0]
        beg, end = s["is_sent_beginning"].tolist(), s["is_sent_end"].tolist()
        for l in range(n_layers):
            ws, ot = recv_variants(atts[l], seq)
            ws_w = at._normalize(np.nan_to_num(per_word(ws, pos)))
            ot_w = at._normalize(np.nan_to_num(per_word(ot, pos)))
            for k in range(len(wl)):
                if beg[k] == 1 or end[k] == 1:
                    continue
                rows.append((tid, idx[k], l, ws_w[k], ot_w[k]))
    df = pd.DataFrame(rows, columns=WORD_KEY + ["layer", "with_self", "others_only"])
    m = df.merge(et, on=WORD_KEY).dropna(subset=["pca"])

    print("layer  spearman(with_self,pca)  spearman(others_only,pca)")
    for l, g in m.groupby("layer"):
        r_ws = spearmanr(g["with_self"], g["pca"])[0]
        r_ot = spearmanr(g["others_only"], g["pca"])[0]
        print(f"{l:5d}      {r_ws:+.3f}                  {r_ot:+.3f}")


if __name__ == "__main__":
    main()
