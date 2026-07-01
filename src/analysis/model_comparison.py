"""Surprisal comparison: reader-aligned vs single model (step 5).

On the whole corpus, fit each surprisal source S in the same lme4 mixed model
(via pymer4) and compare ΔLL over the no-surprisal baseline. Following Škrjanec &
Demberg the response is ``log(RT)`` and the random effects are crossed —
``(1|reader_id) + (1 + is_expert|word_id)``. S is one of:
  baseline          : un-adapted step-0 model (same for every reader)
  physics / biology : the domain fine-tuned model (every reader)
  aligned           : by READER discipline — physics surprisal for physicists,
                      biology for biologists (even on the other domain's texts)
  prompted          : baseline weights + discipline-matched system prompt

The question: does ``aligned`` beat the single-model sources? Significance is the
nested likelihood-ratio p (``p_lrt``).
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from tqdm import tqdm

from src.config import WORD_KEY

SURPRISAL_MODELS = ("baseline", "physics", "biology", "aligned", "prompted")


def _prep_models(df, measure):
    """One row per reader×word with every surprisal column + covariates.

    ``df`` must carry ``s_base`` / ``s_phys`` / ``s_bio`` + reading measures. The
    prompted columns are optional; absent, the prompted models are not built.
    """
    basic_prompt_cols = ["s_prompt_phys", "s_prompt_bio"]
    has_basic_prompt = all(c in df.columns for c in basic_prompt_cols)
    d = df[df[measure] > 0].copy()
    d = d.dropna(
        subset=[
            measure,
            "word_length",
            "lemma_frequency_normalized",
            "s_base",
            "s_phys",
            "s_bio",
            *(basic_prompt_cols if has_basic_prompt else []),
        ]
    )
    d = d[d["word_length"] > 0]
    # dlexDB lemma freq is right-skewed; log1p keeps zero-freq words finite.
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])
    d["log_word_length"] = np.log(d["word_length"])
    # reader discipline (1 = physics, 0 = biology)
    physicist = (d["reader_discipline_numeric"] == 1).to_numpy(dtype=float)
    d["S_baseline"] = d["s_base"]
    d["S_physics"] = d["s_phys"]
    d["S_biology"] = d["s_bio"]
    # aligned: discipline-matched fine-tuned LM, mixed by reader discipline.
    d["S_aligned"] = physicist * d["s_phys"] + (1.0 - physicist) * d["s_bio"]
    if has_basic_prompt:
        d["S_prompted"] = (
            physicist * d["s_prompt_phys"] + (1.0 - physicist) * d["s_prompt_bio"]
        )
    # extra predictors for the fixed-effects spec (see _BASE_TERMS). Position is
    # standardized WITHIN sentence (Škrjanec et al.'s WordIndexInSentence).
    d["word_position"] = (d["word_index_in_sent"] - d["word_index_in_sent"].mean()) / d[
        "word_index_in_sent"
    ].std()
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    # residual (split-signal) columns: what each adapted model ADDS over the base
    # LM. D_<name> = S_<name> - S_baseline. Fit as S_baseline + D_<name> when
    # model_comparison_over_epochs(residual=True) so p(D) tests the fine-tune's
    # own contribution. D_baseline == 0, so baseline stays the reference.
    for _name in ("physics", "biology", "aligned", "prompted"):
        _col = f"S_{_name}"
        if _col in d.columns:
            d[f"D_{_name}"] = d[_col] - d["S_baseline"]
    # deviation (sum) coding for the two factors that interact with surprisal: -1
    # novice / +1 expert, -1 common / +1 technical. Makes the surprisal (and D)
    # main effect the GRAND-MEAN slope over the 2x2 cells, not the novice-common
    # corner that 0/1 treatment coding would report. Fit / ΔLL unchanged.
    d["is_expert"] = 2 * d["is_expert"] - 1
    d["is_technical"] = 2 * d["is_technical"] - 1
    # single grouping key for the by-word random effect (lme4 word_id).
    d["word_id"] = d["text_id"].astype(str) + "_" + d["word_index_in_text"].astype(str)
    return d


# The only fixed-effects spec we fit: Škrjanec, Broy & Demberg (2023) richest
# model (footnote 5) — the three-way surprisal × expertise × terminology
# interaction. ``_BASE_TERMS`` are the terms WITHOUT surprisal, ``_SURPRISAL_TERMS``
# the terms ADDED with it (``{col}`` = the surprisal column), ``_RANDOM_EFFECTS``
# the crossed random effects with a by-word expertise slope.
_BASE_TERMS = "word_length + word_position + is_expert * is_technical"
_SURPRISAL_TERMS = "{col} * is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"


def _fit_model(d, measure, surprisal_col=None, extra_terms=None):
    """lme4 mixed model ``log(measure) ~ <base> [+ surprisal terms] + <re>``.

    Fit with ML (``REML=False``) via pymer4 (lme4) so the log-likelihoods drive
    a nested likelihood-ratio test (the split-signal exp model adds one term, the
    residual, to the base-surprisal reference). Coefficient p-values are Wald z;
    the Satterthwaite Hessian is skipped since significance is the LRT. Without
    ``surprisal_col`` the no-surprisal reference is fit. Returns the fitted
    ``pymer4.models.lmer``.
    """
    import polars as pl
    from pymer4.models import lmer  # lazy: importing loads R via rpy2

    rhs = _BASE_TERMS
    if surprisal_col is not None:
        rhs += " + " + _SURPRISAL_TERMS.format(col=surprisal_col)
    # raw extra terms (e.g. the split-signal residual D_<name> as a plain slope,
    # added on top of the base surprisal's full interaction block).
    if extra_terms:
        rhs += " + " + extra_terms
    # paper always models log RT; measure is filtered > 0 in _prep_models. Precompute
    # the response so residuals/log-likelihood are on the log scale.
    dd = d.copy()
    dd["resp_log"] = np.log(dd[measure].to_numpy())  # valid R identifier (no leading _)
    formula = f"resp_log ~ {rhs} + {_RANDOM_EFFECTS}"
    m = lmer(formula, data=pl.from_pandas(dd))
    # pymer4 streams R convergence chatter to stdout; silence it for the sweep.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # conf_method="wald" skips the Satterthwaite Hessian (unused: significance
        # is the nested LRT); a large win on the random-slope specs.
        m.fit(summary=False, REML=False, conf_method="wald")
    return m


def _loglik(m) -> float:
    """ML log-likelihood of a fitted pymer4 0.9 ``lmer`` (via the R model)."""
    import rpy2.robjects as ro

    return float(np.asarray(ro.r("logLik")(m.r_model)).ravel()[0])


def build_index_df(
    surp_versions, rt_df, prompt_surp, index, measure
):  # prompt_surp may be None
    """Reader×word frame with all surprisal columns + covariates at one checkpoint.

    Physics/biology checkpoints are paired by ``index`` (0 = baseline), not
    ``epoch`` (step counts differ). Index 0 supplies ``s_base``. ``prompt_surp``
    (optional) carries the checkpoint-independent prompted columns.
    """
    base_index = sorted(surp_versions["index"].unique())[0]

    def _surp(idx, domain, name):
        sel = surp_versions[
            (surp_versions["index"] == idx) & (surp_versions["domain"] == domain)
        ]
        return sel[WORD_KEY + ["surprisal"]].rename(columns={"surprisal": name})

    surp = (
        _surp(base_index, "physics", "s_base")
        .merge(_surp(index, "physics", "s_phys"), on=WORD_KEY)
        .merge(_surp(index, "biology", "s_bio"), on=WORD_KEY)
    )
    if prompt_surp is not None:
        surp = surp.merge(prompt_surp, on=WORD_KEY)
    merged = surp.merge(rt_df, on=WORD_KEY, how="inner")
    d = _prep_models(merged, measure).reset_index(drop=True)
    return d


def _coef(m, name, field):
    """Fixed-effect ``field`` ('estimate'/'p_value') for term ``name``.

    Reads pymer4 0.9's ``result_fit`` (a polars frame keyed by ``term``).
    """
    try:
        rf = m.result_fit
        row = rf.filter(rf["term"] == name)
        return float(row[field][0]) if row.height else np.nan
    except (KeyError, AttributeError, TypeError, IndexError):
        return np.nan


def _n_fixed(m) -> int | None:
    """Number of fixed-effect terms in a fitted pymer4 ``lmer`` (LRT df helper)."""
    try:
        return int(m.result_fit.height)
    except (AttributeError, TypeError):
        return None


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    models=SURPRISAL_MODELS,
    indices=None,
    residual=True,
) -> pd.DataFrame:
    """Per-checkpoint surprisal-model comparison on the whole corpus.

    ``surp_versions`` must hold both domains. ``prompt_surp=None`` skips the
    prompted models. ``models`` defaults to ``SURPRISAL_MODELS``. ``indices``
    restricts the checkpoint sweep.

    Two modes:
      residual=True (default, split-signal) — the base LM surprisal
        (``S_baseline``, full interaction) is in BOTH reference and experimental
        model; each adapted model adds only its residual
        ``D_<name> = S_<name> - S_baseline`` as a plain slope. ``b_surprisal`` /
        ``p_surprisal`` test that residual (Wald z); ``delta_ll`` = gain of the
        residual OVER base surprisal; ``p_lrt`` = nested likelihood-ratio p (the
        reference is nested in the experimental model). ``baseline`` is skipped
        (D == 0, it IS the reference).
      residual=False — each model's surprisal is fit against the NO-surprisal
        reference. ``b_surprisal`` / ``p_surprisal`` = the source's own slope;
        ``delta_ll`` = gain over no surprisal; ``p_lrt`` = LRT p for the whole
        surprisal block.

    Long-form columns: ``index``, ``epoch``, ``ref``, ``model``, ``n``,
    ``ll``, ``delta_ll``, ``b_surprisal``, ``p_surprisal``, ``p_lrt``.
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        keep = set(indices)
        all_indices = [i for i in all_indices if i in keep]
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()

    fit_models = [m for m in models if not (residual and m == "baseline")]
    ref_label = "base_surprisal" if residual else "no_surprisal"
    total_fits = len(all_indices) * (1 + len(fit_models))
    rows = []
    with tqdm(total=total_fits, desc="lme4 fits", unit="fit") as pbar:
        for index in all_indices:
            d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
            pbar.set_postfix(index=index, model=ref_label)
            # reference: no surprisal, or base surprisal only (split-signal mode).
            ref = _fit_model(d, measure, "S_baseline" if residual else None)
            ll0 = _loglik(ref)
            n_ref = _n_fixed(ref)
            pbar.update(1)
            for name in fit_models:
                pbar.set_postfix(index=index, model=name)
                if residual:
                    report_col = f"D_{name}"
                    res = _fit_model(d, measure, "S_baseline", extra_terms=report_col)
                else:
                    report_col = f"S_{name}"
                    res = _fit_model(d, measure, report_col)
                ll = _loglik(res)
                pbar.update(1)
                dll = ll - ll0
                # nested LRT: 2*ΔLL ~ chi2 with df = added fixed terms (residual
                # mode: 1, the D slope). Negative ΔLL (exp no better) -> p = 1.
                df_lrt = (_n_fixed(res) or 0) - (n_ref or 0)
                p_lrt = (
                    float(chi2.sf(2.0 * max(dll, 0.0), df_lrt))
                    if df_lrt > 0
                    else np.nan
                )
                rows.append(
                    {
                        "index": index,
                        "epoch": epoch_of[index],
                        "ref": ref_label,
                        "model": name,
                        "n": len(d),
                        "ll": ll,
                        "delta_ll": dll,
                        # slope (grand-mean, sum-coded) + Wald p of the tested term;
                        # interaction terms carry the expert/terminology modulation.
                        "b_surprisal": _coef(res, report_col, "estimate"),
                        "p_surprisal": _coef(res, report_col, "p_value"),
                        # nested likelihood-ratio p (replaces the old Vuong test).
                        "p_lrt": p_lrt,
                    }
                )
    return pd.DataFrame(rows).sort_values(["model", "index"])
