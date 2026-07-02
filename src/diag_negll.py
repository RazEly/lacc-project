"""Isolation: WHERE does the residual-split ΔLL go negative?

GPT-2 only, baseline (index 0) vs checkpoint index 5 (~1024 steps), TFT.
Starts from the ORIGINAL working spec (direct surprisal, positive ΔLL) and
introduces each change from the current pipeline one at a time. A nested ML ΔLL
can only be negative if the bigger model failed to converge — so we also print
lme4's convergence messages (normally suppressed).

    PYTHONPATH=. pixi run python -m src.diag_negll
"""

from __future__ import annotations

import warnings

import numpy as np

from src.analysis import model_comparison as mc
from src.features import data, reading_time as rt, surprisal as su
import src.main as M

INDEX = 5  # DAPT_CHECKPOINT_STEPS[4] = 1024 steps
MEASURE = "TFT"


def _fit(rhs: str, re: str, d):
    """ML lmer fit. Returns (loglik, n_obs, convergence_message)."""
    import polars as pl
    import rpy2.robjects as ro
    from pymer4.models import lmer

    dd = d.copy()
    dd["resp_log"] = np.log(dd[MEASURE].to_numpy())
    m = lmer(f"resp_log ~ {rhs} + {re}", data=pl.from_pandas(dd))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(summary=False, REML=False, conf_method="wald")
    ll = float(np.asarray(ro.r("logLik")(m.r_model)).ravel()[0])
    nobs = int(np.asarray(ro.r("nobs")(m.r_model)).ravel()[0])
    msg = ro.r(
        'function(m){x<-m@optinfo$conv$lme4$messages;'
        'if(is.null(x)) "converged" else paste(x,collapse=" | ")}'
    )(m.r_model)
    return ll, nobs, str(list(msg)[0])


def _compare(tag: str, ref_rhs: str, exp_rhs: str, re: str, d, nested: bool):
    ll_ref, n_ref, _ = _fit(ref_rhs, re, d)
    ll_exp, n_exp, conv = _fit(exp_rhs, re, d)
    dll = ll_exp - ll_ref
    flag = ""
    if nested and dll < -1e-6:
        flag = "  <-- NEGATIVE on NESTED (impossible unless non-converged)"
    print(
        f"[{tag}]\n"
        f"    ref  LL={ll_ref:14.4f} (n={n_ref})\n"
        f"    exp  LL={ll_exp:14.4f} (n={n_exp})  conv: {conv}\n"
        f"    dLL={dll:+.4f}  nested={nested}{flag}\n"
    )
    return dll


def main() -> None:
    rm_raw = data.load_reading_measures()
    cache = su.load_cache()
    b = M.bundle_from_cache(cache, "german-gpt2")
    rmc = rt.clean_reading_times(rm_raw, MEASURE)
    d = mc.build_index_df(b["surp_versions"], rmc, b["prompt_surp"], INDEX, MEASURE)
    print(f"rows at index {INDEX}: {len(d)}\n")

    # d columns from current _prep: S_baseline, S_aligned, D_aligned,
    # is_expert/is_technical are DEVIATION coded (+/-1), word_length / log_word_freq
    # / word_position are RAW. Derive the ORIGINAL knobs off them.
    d = d.copy()
    d["e01"] = ((d["is_expert"] + 1) / 2).astype(int)      # 0/1 treatment
    d["t01"] = ((d["is_technical"] + 1) / 2).astype(int)
    z = lambda s: (s - s.mean()) / s.std()
    d["pos_z"] = z(d["word_position"])
    d["wl_z"] = z(d["word_length"])

    RE_orig = "(1|reader_id) + (1 + e01|word_id)"          # 0/1 slope
    RE_dev = "(1|reader_id) + (1 + is_expert|word_id)"     # +/-1 slope
    RE_simple = "(1|reader_id) + (1|word_id)"              # no random slope

    IXT01 = "e01*t01"
    IXTdev = "is_expert*is_technical"

    # C0 — ORIGINAL (paper_full, direct): no freq, raw word_length, z position,
    # 0/1 factors, surprisal source DIRECT, reference = NO surprisal. Expect dLL>0.
    base0 = f"word_length + pos_z + {IXT01}"
    _compare("C0 original direct (vs no-surprisal)", base0,
             base0 + f" + S_aligned*{IXT01}", RE_orig, d, nested=True)

    # C1 — + log_word_freq main effect (the word_log_frequency change).
    base1 = base0 + " + log_word_freq"
    _compare("C1 +log_word_freq", base1,
             base1 + f" + S_aligned*{IXT01}", RE_orig, d, nested=True)

    # C2 — RESIDUAL split: ref = base LM surprisal, exp = base + D_aligned block.
    # This is the current formulation. Nested -> dLL MUST be >= 0.
    ref2 = base1 + f" + S_baseline*{IXT01}"
    _compare("C2 residual split (ref=S_baseline, +D block)", ref2,
             ref2 + f" + D_aligned*{IXT01}", RE_orig, d, nested=True)

    # C3 — residual + deviation coding of the factors (current coding).
    base3 = f"word_length + pos_z + {IXTdev} + log_word_freq"
    ref3 = base3 + f" + S_baseline*{IXTdev}"
    _compare("C3 residual + deviation coding", ref3,
             ref3 + f" + D_aligned*{IXTdev}", RE_dev, d, nested=True)

    # C4 — residual + deviation, but SIMPLE random effects (drop the by-word
    # is_expert random slope). Tests whether the random slope is the culprit.
    _compare("C4 residual + deviation + SIMPLE RE (no random slope)", ref3,
             ref3 + f" + D_aligned*{IXTdev}", RE_simple, d, nested=True)

    # C5 — the real (NON-nested) science question the residual split replaced:
    # does S_aligned beat S_baseline directly? Negative dLL here is legitimate.
    ref5 = base1 + f" + S_baseline*{IXT01}"
    exp5 = base1 + f" + S_aligned*{IXT01}"
    _compare("C5 direct S_aligned vs S_baseline (NON-nested)", ref5, exp5,
             RE_orig, d, nested=False)


if __name__ == "__main__":
    main()
