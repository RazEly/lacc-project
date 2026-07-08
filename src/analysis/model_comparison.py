"""
Baseline model (Eq. 2), NO surprisal:
    log(RT) ~ Length + LogFreq + Position + Expertise*Terminology
              + (1|SubjectID) + (1 + Expertise|WordID)
Experimental model (Eq. 4): baseline + Surprisal, a SINGLE PLAIN MAIN EFFECT
(no interaction, no residualization).

Surprisal sources:
  baseline          : un-adapted step-0 model (same for every reader).
  physics / biology : the domain fine-tuned model (every reader).
  aligned           : by READER discipline — physics surprisal for physicists,
                      biology for biologists (Study 2).
  prompted          : baseline weights + discipline-matched prior passage.
  prompt_neutral    : fixed-domain PSEUDO-TEST — baseline weights + an off-domain
                      (neutral) prior passage for EVERY reader, holding context
                      length fixed while the prior carries no domain signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from pymer4.models import lmer  # lazy: importing loads R via rpy2
from scipy.stats import chi2
from tqdm import tqdm

from src.config import DAPT_CHECKPOINT_STEPS, WORD_KEY
from src.features.dataset import build_index_df

STEP_OF_INDEX = {i: s for i, s in enumerate([0, *DAPT_CHECKPOINT_STEPS])}

SURPRISAL_MODELS = (
    "baseline",
    "physics",
    "biology",
    "aligned",
    "prompted",
    "prompt_neutral",
)
_CKPT_INDEP = {"baseline", "prompted", "prompt_neutral"}

# linear mixed effects model
_BASE_TERMS = "word_length + log_word_freq + word_position + is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"


def _fit(d, measure, surprisal_cols=None):
    """ML lmer fit of ``log(measure) ~ <base> [+ surprisal…] + <random>`` (Eq. 2/4).

    ``surprisal_cols``: None (no-surprisal baseline, Eq. 2), one column name,
    or a list — each enters as a plain main effect. ML (REML=False) so the LLs
    feed the nested LRT / ΔLL. Coefficient p-values are Wald z.
    """

    if isinstance(surprisal_cols, str):
        surprisal_cols = [surprisal_cols]
    rhs = " + ".join([_BASE_TERMS, *(surprisal_cols or [])])
    dd = d.assign(resp_log=np.log(d[measure].to_numpy()))  # valid R identifier
    m = lmer(f"resp_log ~ {rhs} + {_RANDOM_EFFECTS}", data=pl.from_pandas(dd))
    m.fit(summary=False, REML=False, conf_method="wald")
    return m


def _stat(m, name) -> float:
    """Teturns fit statistic (log-likelihood / AIC / std)"""
    return float(m.result_fit_stats[name].item())


def _coef(m, name, field):
    """Fixed-effect ``field`` ('estimate'/'std_error'/'p_value') for term ``name``,
    from pymer4 0.9's ``result_fit`` (a polars frame keyed by ``term``)."""
    rf = m.result_fit
    row = rf.filter(rf["term"] == name)
    return float(row[field][0]) if row.height else np.nan


def _score_source(d, measure, name, index, training_steps, ll_null):
    """Fit Eq. 2 + one surprisal main effect; LRT stats vs the shared
    no-surprisal null. Returns the results row."""
    col = f"s_{name}"
    fit = _fit(d, measure, col)
    ll = _stat(fit, "logLik")
    dll = ll - ll_null
    row = {
        "index": index,
        "training_steps": training_steps,
        "model": name,
        "n": len(d),
        "ll": ll,
        "delta_ll": dll,
        # nested LRT: 2·ΔLL ~ chi2(1), the single added surprisal term.
        # Negative ΔLL (no better than the null) -> p = 1.
        "chisq": 2.0 * dll,
        "p_lrt": float(chi2.sf(max(2.0 * dll, 0.0), 1)),
        "aic": _stat(fit, "AIC"),
        "b_surprisal": _coef(fit, col, "estimate"),
        "se_surprisal": _coef(fit, col, "std_error"),
    }
    return row


def model_comparison_over_steps(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    models=SURPRISAL_MODELS,
    indices=None,
) -> pd.DataFrame:
    """Per-checkpoint surprisal-source comparison on the whole corpus (Eq. 2/4/5).

    ``surp_versions`` must hold both domains; ``indices`` restricts the sweep.
    Checkpoint-independent sources (``_CKPT_INDEP``) are fit once (``index`` /
    ``training_steps`` = NA). The no-surprisal reference LL is computed once, on the
    first index's frame, and reused for every checkpoint (valid: the row set is
    checkpoint-independent, asserted per index).

    Returns ``results`` — one row per source × checkpoint, every source vs the
    shared no-surprisal baseline: ``index``, ``training_steps``, ``model``,
    ``n``, ``ll``, ``delta_ll``, ``chisq`` (= 2·ΔLL), ``p_lrt``, ``aic``, plus
    ``b_surprisal`` / ``se_surprisal`` (slope estimate).
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        all_indices = [i for i in all_indices if i in set(indices)]
    steps_of = {i: STEP_OF_INDEX[int(i)] for i in all_indices}

    d0 = build_index_df(surp_versions, rt_df, prompt_surp, all_indices[0], measure)
    dep = [m for m in models if m not in _CKPT_INDEP]
    # column check: skip an indep source whose surprisal column is absent.
    indep = [m for m in models if m in _CKPT_INDEP and f"s_{m}" in d0.columns]

    rows: list[dict] = []

    # once: no-surprisal baseline + each indep source; per index: dep sources.
    total_fits = 1 + len(indep) + len(all_indices) * len(dep)
    with tqdm(total=total_fits, desc="lme4 fits", unit="fit") as pbar:
        pbar.set_postfix(model="no_surprisal_baseline")
        null_fit = _fit(d0, measure)
        ll_null = _stat(null_fit, "logLik")
        pbar.update(1)

        def _score(name, d, index, training_steps):
            rows.append(_score_source(d, measure, name, index, training_steps, ll_null))
            pbar.update(1)

        for name in indep:
            pbar.set_postfix(model=name)
            _score(name, d0, pd.NA, pd.NA)

        for i, index in enumerate(all_indices):
            if i == 0:
                d = d0
            else:
                d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
                # reusing d0's reference LL at this index requires the same rows.
                assert d[WORD_KEY].equals(d0[WORD_KEY]), (
                    "checkpoint row sets differ; baseline reference not reusable"
                )
            for name in dep:
                pbar.set_postfix(index=index, model=name)
                _score(name, d, index, steps_of[index])

    return pd.DataFrame(rows).sort_values(["model", "index"])
