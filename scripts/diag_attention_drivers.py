"""What does our per-word attention actually track? Length? Position? Gaze?

Pipeline-style received attention (sum-normalized within sentence, max over
subtokens) on the cached german-gpt2 baseline, correlated per layer against:
  - word_length and within-sentence position (surface drivers)
  - raw FFD / TFT (un-normalized gaze, as the pipeline correlates)
  - sentence-sum-normalized FFD (gaze normalized to MATCH the attention norm,
    i.e. the paper's "relative value in each sentence")
  - gaze PCA (pipeline default)
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


def main():
    words = data.load_word_features()
    rm = data.load_reading_measures()
    et = co.build_et_table(rm, participants="all")  # raw 6 features + pca

    # raw per-word gaze means + length/position
    feats = rm.drop_duplicates(WORD_KEY)[WORD_KEY + ["word_length", "word_index_in_sent"]]
    et = et.merge(feats, on=WORD_KEY, how="left")
    # sentence-sum-normalized FFD (paper's "relative value in sentence");
    # build_et_table already carries sent_index_in_text.
    et["FFD_rel"] = et.groupby(["text_id", "sent_index_in_text"])["FFD"].transform(
        lambda x: x / x.sum()
    )

    ckpt = config.CHECKPOINTS_DIR / "german-gpt2_physics_lora" / "checkpoint_00"
    model, tok = su.load_causal_lm(str(ckpt), attn=True)
    attn = at.extract_attention(words, model, tok, method="raw")

    m = attn.merge(et, on=WORD_KEY)
    targets = ["word_length", "word_index_in_sent", "FFD", "TFT", "FFD_rel", "pca"]
    print("layer | " + " ".join(f"{t:>10s}" for t in targets))
    for l, g in m.groupby("layer"):
        cells = []
        for t in targets:
            d = g[["attention", t]].dropna()
            r = spearmanr(d["attention"], d[t])[0] if len(d) > 3 else np.nan
            cells.append(f"{r:+.3f}".rjust(10))
        print(f"{l:5d} | " + " ".join(cells))


if __name__ == "__main__":
    main()
