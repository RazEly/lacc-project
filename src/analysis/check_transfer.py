"""Minimal transfer check: does a DAPT run lower surprisal on the PoTeC stimuli?

The core validation for the corpus/labelling rework. Loads the baseline checkpoint
(index 0 = un-adapted) and the final checkpoint of a DAPT run, scores the PoTeC
stimulus texts with each, and reports mean per-word surprisal (bits) split by
whether the stimulus domain matches the fine-tune domain.

A corpus that actually transfers should DROP in-domain surprisal (negative delta);
the pre-rework archaic-OCR corpus left it flat / slightly positive. Out-of-domain
is the control — it should move less (or up).

    pixi run python -m src.analysis.check_transfer                                  # german-gpt2 physics
    pixi run python -m src.analysis.check_transfer --run artifacts/german-gpt2_biology_lora

CPU is fine (german-gpt2 is small, 12 texts). No GPU needed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import WORD_KEY
from src.features import potec
from src.features.surprisal import compute_surprisal
from src.modeling.lm import load_causal_lm


def _score(words, ckpt):
    """Per-word PoTeC surprisal (bits) for one checkpoint."""
    model, tok = load_causal_lm(ckpt)
    sup = compute_surprisal(words, model, tok)
    del model
    return sup


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="artifacts/german-gpt2_physics_lora",
                    help="DAPT run dir holding manifest.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(Path(args.run) / "manifest.csv")
    ft_domain = str(manifest["domain"].iloc[0])
    base_ck = manifest.loc[manifest["index"] == 0, "checkpoint"].iloc[0]
    trained = manifest.loc[manifest["index"].idxmax()]
    trained_ck = trained["checkpoint"]
    print(f"run={args.run}  fine-tune domain={ft_domain}")
    print(f"  baseline (idx 0) : {base_ck}")
    print(f"  trained  (idx {int(trained['index'])}, step {int(trained['step'])}) : {trained_ck}")

    words = potec.load_word_features().dropna(subset=["text_domain"])
    dom = words[WORD_KEY + ["text_domain"]].drop_duplicates()

    means = {}
    for label, ck in [("baseline", base_ck), ("trained", trained_ck)]:
        sup = _score(words, ck).merge(dom, on=WORD_KEY)
        sup["match"] = np.where(
            sup["text_domain"] == ft_domain, "in-domain", "out-of-domain"
        )
        means[label] = sup.groupby("match")["surprisal"].mean()

    out = pd.DataFrame(means)
    out["delta_bits"] = out["trained"] - out["baseline"]
    out["ppl_baseline"] = 2 ** out["baseline"]   # per-word pseudo-perplexity
    out["ppl_trained"] = 2 ** out["trained"]

    print("\nMean PoTeC surprisal (bits/word) + per-word perplexity:")
    print(out.round(3).to_string())
    in_delta = out.loc["in-domain", "delta_bits"]
    verdict = "TRANSFER ✓ (in-domain surprisal dropped)" if in_delta < -0.05 else \
              "NO transfer (in-domain flat / up)"
    print(f"\nin-domain delta = {in_delta:+.3f} bits  →  {verdict}")


if __name__ == "__main__":
    main()
