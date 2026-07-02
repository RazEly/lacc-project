"""Prove the residual-split negative ΔLL is an OPTIMIZER artifact, not a worse fit.

GPT-2, baseline vs index 5 (~1024 steps), TFT. Three checks:
  A raw residual      : ref=S_baseline block, exp=+D_aligned block  -> reproduces dLL<0
  B scaled residual   : same, but surprisal columns z-scored (affine, same true
                        optimum) -> optimizer now reaches it, dLL >= 0
  C reparam identity  : exp as (S_baseline + D_aligned) vs (S_baseline + S_aligned)
                        -> identical LL (same column span; residual == both-blocks)

    PYTHONPATH=. pixi run python -m src.diag_optimizer
"""

from __future__ import annotations

import warnings

import numpy as np

from src.analysis import model_comparison as mc
from src.features import data, reading_time as rt, surprisal as su
import src.main as M

INDEX, MEASURE = 5, "TFT"
RE = "(1|reader_id) + (1 + e01|word_id)"
IXT = "e01*t01"


def _fit(rhs: str, d) -> float:
    import polars as pl
    import rpy2.robjects as ro
    from pymer4.models import lmer

    dd = d.copy()
    dd["resp_log"] = np.log(dd[MEASURE].to_numpy())
    m = lmer(f"resp_log ~ {rhs} + {RE}", data=pl.from_pandas(dd))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(summary=False, REML=False, conf_method="wald")
    import rpy2.robjects as ro2
    return float(np.asarray(ro2.r("logLik")(m.r_model)).ravel()[0])


def main() -> None:
    rm_raw = data.load_reading_measures()
    b = M.bundle_from_cache(su.load_cache(), "german-gpt2")
    rmc = rt.clean_reading_times(rm_raw, MEASURE)
    d = mc.build_index_df(b["surp_versions"], rmc, b["prompt_surp"], INDEX, MEASURE).copy()
    d["e01"] = ((d["is_expert"] + 1) / 2).astype(int)
    d["t01"] = ((d["is_technical"] + 1) / 2).astype(int)
    # correlation of the two near-collinear surprisal columns
    r = np.corrcoef(d["S_baseline"], d["S_aligned"])[0, 1]
    print(f"corr(S_baseline, S_aligned) = {r:.4f}   n={len(d)}\n")

    z = lambda s: (s - s.mean()) / s.std()
    d["Sb_z"] = z(d["S_baseline"])
    d["Sa_z"] = z(d["S_aligned"])
    d["D_z"] = z(d["D_aligned"])

    base = f"word_length + word_position + {IXT} + log_word_freq"

    # A — raw residual (current pipeline): expect dLL < 0 (optimizer stops short).
    ref_a = f"{base} + S_baseline*{IXT}"
    exp_a = f"{ref_a} + D_aligned*{IXT}"
    lla_ref, lla_exp = _fit(ref_a, d), _fit(exp_a, d)
    print(f"A raw residual     ref={lla_ref:12.4f} exp={lla_exp:12.4f} dLL={lla_exp-lla_ref:+.4f}")

    # B — scaled residual (affine reparam, SAME true optimum): expect dLL >= 0.
    ref_b = f"{base} + Sb_z*{IXT}"
    exp_b = f"{ref_b} + D_z*{IXT}"
    llb_ref, llb_exp = _fit(ref_b, d), _fit(exp_b, d)
    print(f"B scaled residual  ref={llb_ref:12.4f} exp={llb_exp:12.4f} dLL={llb_exp-llb_ref:+.4f}")

    # C — reparam identity: residual (Sb + D) vs both-blocks (Sb + Sa), scaled so
    # both converge. Same column span -> LL must match.
    exp_resid = f"{base} + Sb_z*{IXT} + D_z*{IXT}"
    exp_both = f"{base} + Sb_z*{IXT} + Sa_z*{IXT}"
    ll_resid, ll_both = _fit(exp_resid, d), _fit(exp_both, d)
    print(f"C reparam identity resid={ll_resid:12.4f} both={ll_both:12.4f} diff={ll_resid-ll_both:+.6f}")

    print("\nRef LL (scaled, S_baseline only) =", f"{llb_ref:.4f}")
    print("If B dLL>=0 and A dLL<0 with same data -> A's negative is optimizer, not fit.")


if __name__ == "__main__":
    main()
