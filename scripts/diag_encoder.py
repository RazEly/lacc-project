"""Decisive test: encoder (BERT) raw attention vs gaze, like paper 02's headline.

Our pipeline uses causal DECODERS, where received attention (A[k:,k]) only comes
from later tokens — a position artifact. The paper's strong numbers are the
bidirectional ENCODER (BERT), where received attention = the FULL column A[:,k]
(every other word). This runs german BERT with full-column received attention and
correlates with gaze PCA per layer — if it jumps to strong positive, encoder vs
decoder is the methodological gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from src import config
from src.analysis import correlation as co
from src.features import attention as at
from src.features import data

WORD_KEY = ["text_id", "word_index_in_text"]


@torch.no_grad()
def main():
    words = data.load_word_features()
    rm = data.load_reading_measures()
    et = co.build_et_table(rm, participants="all")[WORD_KEY + ["pca"]]

    name = config.ENCODER_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation="eager").eval()

    sents = [s for _, s in words.sort_values(
        ["text_id", "sent_index_in_text", "word_index_in_sent"]
    ).groupby(["text_id", "sent_index_in_text"], sort=False) if len(s) >= 3]

    rows = []
    for s in sents:
        wl = s["word"].fillna("").astype(str).tolist()
        enc = tok(wl, is_split_into_words=True, return_tensors="pt")
        out = model(**enc, output_attentions=True)
        atts = torch.stack([a[0].mean(0) for a in out.attentions]).numpy()  # [L,seq,seq]
        word_ids = enc.word_ids(0)
        pos = at._word_positions(word_ids, len(wl))
        n_layers, seq, _ = atts.shape
        idx = s["word_index_in_text"].tolist()
        tid = s["text_id"].iloc[0]
        beg, end = s["is_sent_beginning"].tolist(), s["is_sent_end"].tolist()
        for l in range(n_layers):
            A = atts[l]
            recv = A.mean(axis=0)  # FULL column: received from all queries (bidirectional)
            word = at._normalize(np.array([recv[p].max() if p else 0.0 for p in pos]))
            for k in range(len(wl)):
                if beg[k] == 1 or end[k] == 1:
                    continue
                rows.append((tid, idx[k], l, word[k]))
    df = pd.DataFrame(rows, columns=WORD_KEY + ["layer", "attention"])
    m = df.merge(et, on=WORD_KEY).dropna(subset=["pca"])
    print(f"{name}  (bidirectional, full-column received attention)")
    print("layer  spearman(attention, gaze PCA)")
    for l, g in m.groupby("layer"):
        print(f"{l:5d}      {spearmanr(g['attention'], g['pca'])[0]:+.3f}")


if __name__ == "__main__":
    main()
