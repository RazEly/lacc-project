"""Surprisal comparison: reader-aligned vs single model (step 5).

On the whole corpus, fit each surprisal source S in the same lme4 mixed model
(via pymer4) and compare ΔLL over the base-surprisal reference. Following Škrjanec &
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

    ``df`` must carry ``s_base`` / ``s_phys`` / ``s_bio`` / ``s_prompt_*`` plus the
    reading measures.
    """
    d = df[df[measure] > 0].copy()
    d = d.dropna(
        subset=[
            measure,
            "word_length",
            "lemma_frequency_normalized",
            "s_base",
            "s_phys",
            "s_bio",
            "s_prompt_phys",
            "s_prompt_bio",
        ]
    )
    d = d[d["word_length"] > 0]
    # dlexDB lemma freq: +1 smoothing then log (paper), z-scored below.
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])

    # per-reader surprisal sources. aligned / prompted pick the discipline-matched
    # column (1 = physicist, 0 = biologist).
    is_physicist = d["reader_discipline_numeric"] == 1
    d["S_baseline"] = d["s_base"]
    d["S_physics"] = d["s_phys"]
    d["S_biology"] = d["s_bio"]
    d["S_aligned"] = np.where(is_physicist, d["s_phys"], d["s_bio"])
    d["S_prompted"] = np.where(is_physicist, d["s_prompt_phys"], d["s_prompt_bio"])

    # position standardized WITHIN sentence (Škrjanec et al.'s WordIndexInSentence).
    pos = d["word_index_in_sent"]
    d["word_position"] = (pos - pos.mean()) / pos.std()
    # scale + center the remaining continuous covariates (paper scales all).
    # Surprisal columns stay in raw bits: the residual D = S_model - S_base needs
    # every S on one common scale, and slopes stay comparable across sources.
    for col in ("word_length", "log_word_freq"):
        d[col] = (d[col] - d[col].mean()) / d[col].std()
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    # residual (split-signal) columns: what each adapted model ADDS over the base LM.
    # Fit as S_baseline + D_<name> so p(D) tests the fine-tune's own contribution.
    # D_baseline == 0, so baseline stays the reference.
    for name in ("physics", "biology", "aligned", "prompted"):
        d[f"D_{name}"] = d[f"S_{name}"] - d["S_baseline"]
    # deviation (sum) coding for the two factors that interact with surprisal: -1
    # novice / +1 expert, -1 common / +1 technical. Makes the surprisal (and D) main
    # effect the GRAND-MEAN slope over the 2x2 cells, not the novice-common corner
    # that 0/1 treatment coding would report. Fit / ΔLL unchanged.
    d["is_expert"] = 2 * d["is_expert"] - 1
    d["is_technical"] = 2 * d["is_technical"] - 1
    # single grouping key for the by-word random effect (lme4 word_id).
    d["word_id"] = d["text_id"].astype(str) + "_" + d["word_index_in_text"].astype(str)
    return d


# The only fixed-effects spec we fit: Škrjanec, Broy & Demberg (2023) richest
# model (footnote 5) — the three-way surprisal × expertise × terminology
# interaction — plus the paper's Eq. (2) covariates (length, log frequency,
# position). ``_BASE_TERMS`` are the terms WITHOUT surprisal, ``_SURPRISAL_TERMS``
# the terms ADDED with it (``{col}`` = the surprisal column), ``_RANDOM_EFFECTS``
# the crossed random effects with a by-word expertise slope.
_BASE_TERMS = "word_length + log_word_freq + word_position + is_expert * is_technical"
_SURPRISAL_TERMS = "{col} * is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"
# surprisal × expert × technical expands to 4 fixed terms (main + 3 interactions);
# that is the df of the standard baseline-vs-null LRT. The residual D gets the
# SAME interaction structure (adaptation gains may live in the expertise
# interaction), so the residual LRT has the same 4 df.
_SURPRISAL_DF = 4
_RESID_DF = 4
# models whose surprisal doesn't depend on the DAPT checkpoint: fit once, not per
# checkpoint. baseline is handled by name; these are the residual-split models.
_CKPT_INDEP = {"prompted"}


def _fit_model(d, measure, surprisal_col, extra_terms=None):
    """lme4 mixed model ``log(measure) ~ <base> + <surprisal> [+ extra] + <re>``.

    ML fit (``REML=False``) via pymer4 (lme4) so the log-likelihoods drive the nested
    likelihood-ratio test — the split-signal experimental model adds the residual
    ``D_<name>`` block (main effect + expertise/terminology interactions) to the
    base-surprisal reference. Coefficient p-values are Wald z; the Satterthwaite
    Hessian is skipped since significance is the LRT.
    Returns the fitted ``pymer4.models.lmer``.
    """
    import polars as pl
    from pymer4.models import lmer  # lazy: importing loads R via rpy2

    rhs = _BASE_TERMS + " + " + _SURPRISAL_TERMS.format(col=surprisal_col)
    # raw extra terms (e.g. the split-signal residual D_<name> interaction block).
    if extra_terms:
        rhs += " + " + extra_terms
    # paper always models log RT; measure is filtered > 0 in _prep_models.
    dd = d.copy()
    dd["resp_log"] = np.log(dd[measure].to_numpy())  # valid R identifier (no leading _)
    formula = f"resp_log ~ {rhs} + {_RANDOM_EFFECTS}"
    m = lmer(formula, data=pl.from_pandas(dd))
    # pymer4 streams R convergence chatter to stdout; silence it for the sweep.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # conf_method="wald" skips the Satterthwaite Hessian (unused: significance is
        # the nested LRT); a large win on the random-slope specs.
        m.fit(summary=False, REML=False, conf_method="wald")
    return m


def _fit_null(d, measure):
    """Null mixed model: base terms only, NO surprisal. Reference for the standard
    baseline LRT — does the base-LM surprisal (with its expert/terminology
    interactions) help at all. ML fit to match ``_fit_model``.
    """
    import polars as pl
    from pymer4.models import lmer  # lazy: importing loads R via rpy2

    dd = d.copy()
    dd["resp_log"] = np.log(dd[measure].to_numpy())
    formula = f"resp_log ~ {_BASE_TERMS} + {_RANDOM_EFFECTS}"
    m = lmer(formula, data=pl.from_pandas(dd))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(summary=False, REML=False, conf_method="wald")
    return m


def _loglik(m) -> float:
    """ML log-likelihood of a fitted pymer4 0.9 ``lmer`` (via the R model)."""
    import rpy2.robjects as ro

    return float(np.asarray(ro.r("logLik")(m.r_model)).ravel()[0])


def _reader_loglik(m, d, measure) -> pd.Series:
    """Per-reader sums of pointwise CONDITIONAL log-densities of a fitted lmer.

    Normal log-density of each observation around ``fitted()`` (which includes
    the BLUPs) with the ML residual sigma — an approximation of each reader's
    contribution to the fit, NOT a decomposition of the marginal ML
    log-likelihood. Meant only for PAIRED reader-level comparisons between fits
    on identical rows (claim_tests), where the shared random-effect penalty
    terms cancel in the pairing. Indexed by ``reader_id``.
    """
    import rpy2.robjects as ro

    fitted = np.asarray(ro.r("fitted")(m.r_model), dtype=float).ravel()
    sigma = float(np.asarray(ro.r("sigma")(m.r_model)).ravel()[0])
    if len(fitted) != len(d):
        raise ValueError(f"fitted() length {len(fitted)} != data rows {len(d)}")
    resp = np.log(d[measure].to_numpy(dtype=float))
    dens = -0.5 * np.log(2 * np.pi * sigma**2) - (resp - fitted) ** 2 / (2 * sigma**2)
    return pd.Series(dens).groupby(d["reader_id"].to_numpy()).sum()


def build_index_df(surp_versions, rt_df, prompt_surp, index, measure):
    """Reader×word frame with all surprisal columns + covariates at one checkpoint.

    Physics/biology checkpoints are paired by ``index`` (0 = baseline), not ``epoch``
    (step counts differ). Index 0 supplies ``s_base``. ``prompt_surp`` carries the
    checkpoint-independent prompted columns.
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
        .merge(prompt_surp, on=WORD_KEY)
    )
    merged = surp.merge(rt_df, on=WORD_KEY, how="inner")
    return _prep_models(merged, measure).reset_index(drop=True)


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


def _row(index, epoch, ref, model, d, ll, dll, fit, resid_col, p_lrt) -> dict:
    """One result row. EVERY model reports both slopes from its own fit:

      ``b_surprisal`` / ``se_surprisal`` / ``p_surprisal``: the base LM surprisal
        (``S_baseline``) main effect — present in every fit.
      ``b_resid`` / ``se_resid`` / ``p_resid``: the residual
        ``D_<name> = S_<name> - S_baseline`` main-effect slope (``resid_col``).
        NA for baseline, which has no residual.

    ``index`` / ``epoch`` may be NA for a checkpoint-independent model (fit once).
    """
    return {
        "index": index,
        "epoch": epoch,
        "ref": ref,
        "model": model,
        "n": len(d),
        "ll": ll,
        "delta_ll": dll,
        "b_surprisal": _coef(fit, "S_baseline", "estimate"),
        "se_surprisal": _coef(fit, "S_baseline", "std_error"),
        "p_surprisal": _coef(fit, "S_baseline", "p_value"),
        "b_resid": _coef(fit, resid_col, "estimate") if resid_col else pd.NA,
        "se_resid": _coef(fit, resid_col, "std_error") if resid_col else pd.NA,
        "p_resid": _coef(fit, resid_col, "p_value") if resid_col else pd.NA,
        "p_lrt": p_lrt,
    }


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    models=SURPRISAL_MODELS,
    indices=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-checkpoint split-signal surprisal-model comparison on the whole corpus.

    ``surp_versions`` must hold both domains. ``models`` defaults to
    ``SURPRISAL_MODELS``; ``indices`` restricts the checkpoint sweep.

    Two kinds of row, distinguished by ``ref``:
      ``baseline`` (ref = ``null``): the STANDARD LRT — base-LM surprisal (full
        interaction) vs the null model with no surprisal at all. ``delta_ll`` is the
        gain over null; ``p_lrt`` is the nested LRT p (``_SURPRISAL_DF`` df).
      every other model (ref = ``base_surprisal``): split-signal — the base LM
        surprisal (``S_baseline``, full interaction) is in BOTH the reference and the
        experimental model; each adapted model adds its residual
        ``D_<name> = S_<name> - S_baseline`` with the SAME expertise × terminology
        interaction block. ``delta_ll`` = gain of the residual block OVER base
        surprisal; ``p_lrt`` = nested LRT p (``_RESID_DF`` df).
    Every row reports both slopes from its own fit (Wald z): ``b_surprisal`` /
    ``se_surprisal`` / ``p_surprisal`` for the base LM surprisal (``S_baseline``),
    and ``b_resid`` / ``se_resid`` / ``p_resid`` for the residual ``D_<name>`` main
    effect (NA for baseline, which has none).

    baseline and prompted (``_CKPT_INDEP``) don't depend on the checkpoint, so each
    is fit ONCE and emitted as a single row with ``index`` / ``epoch`` = NA (not one
    identical copy per checkpoint). The base-surprisal reference LL is likewise
    computed once and reused — valid because the row set is checkpoint-independent
    (asserted per index).

    Returns ``(results, reader_ll)``:
      results — long-form columns ``index``, ``epoch``, ``ref``, ``model``, ``n``,
        ``ll``, ``delta_ll``, ``b_surprisal``, ``se_surprisal``, ``p_surprisal``,
        ``b_resid``, ``se_resid``, ``p_resid``, ``p_lrt``.
      reader_ll — per-reader conditional log-lik sums (``_reader_loglik``) for the
        base-surprisal reference (``model`` = ``base_ref``) and every experimental
        fit; columns ``model``, ``index``, ``reader_id``, ``ll_reader``. Feeds the
        paired cross-LM tests in ``analysis.claim_tests``.
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        all_indices = [i for i in all_indices if i in set(indices)]
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()

    want_baseline = "baseline" in models
    fit_models = [m for m in models if m != "baseline"]
    # baseline and prompted don't use the checkpoint: S_baseline / S_prompted (and so
    # their fits, the base-surprisal reference, and the null) are identical across the
    # whole sweep. Fit them ONCE; only these vary per checkpoint.
    dep_models = [m for m in fit_models if m not in _CKPT_INDEP]
    indep_models = [m for m in fit_models if m in _CKPT_INDEP]

    # once: base reference + (null if baseline) + prompted; per index: dependent models.
    total_fits = 1 + want_baseline + len(indep_models) + len(all_indices) * len(dep_models)
    rows = []
    rll_rows: list[pd.DataFrame] = []

    def _collect_rll(fit, d, model, index):
        rll = _reader_loglik(fit, d, measure).rename("ll_reader").reset_index()
        rll.columns = ["reader_id", "ll_reader"]
        rll.insert(0, "model", model)
        rll.insert(1, "index", index)
        rll_rows.append(rll)

    d0 = None  # first index' frame; base reference / indep fits are reused off it.
    ll0 = None
    with tqdm(total=total_fits, desc="lme4 fits", unit="fit") as pbar:
        for index in all_indices:
            d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
            if d0 is None:
                d0 = d
                pbar.set_postfix(index=index, model="base_surprisal")
                # reference: base surprisal only, no residual. Reused for every
                # checkpoint's nested LRT (below) — valid because the row set is
                # checkpoint-independent (asserted per index).
                base_fit = _fit_model(d, measure, "S_baseline")
                ll0 = _loglik(base_fit)
                _collect_rll(base_fit, d, "base_ref", pd.NA)
                pbar.update(1)
                if want_baseline:
                    pbar.set_postfix(model="baseline")
                    # standard LRT: base surprisal (+interactions) vs no-surprisal null.
                    dll_base = ll0 - _loglik(_fit_null(d, measure))
                    pbar.update(1)
                    p_lrt_base = float(chi2.sf(2.0 * max(dll_base, 0.0), _SURPRISAL_DF))
                    # baseline: no residual (resid_col=None) — reports S_baseline only.
                    rows.append(_row(
                        pd.NA, pd.NA, "null", "baseline", d, ll0, dll_base, base_fit,
                        None, p_lrt_base,
                    ))
                for name in indep_models:
                    pbar.set_postfix(model=name)
                    report_col = f"D_{name}"
                    res = _fit_model(
                        d, measure, "S_baseline",
                        extra_terms=_SURPRISAL_TERMS.format(col=report_col),
                    )
                    dll = _loglik(res) - ll0
                    _collect_rll(res, d, name, pd.NA)
                    pbar.update(1)
                    p_lrt = float(chi2.sf(2.0 * max(dll, 0.0), _RESID_DF))
                    rows.append(_row(
                        pd.NA, pd.NA, "base_surprisal", name, d, ll0 + dll, dll, res,
                        report_col, p_lrt,
                    ))
            else:
                # reusing d0's ll0 for this index' LRTs requires the same rows.
                assert d[WORD_KEY].equals(d0[WORD_KEY]), (
                    "checkpoint row sets differ; base reference not reusable"
                )
            for name in dep_models:
                pbar.set_postfix(index=index, model=name)
                report_col = f"D_{name}"
                res = _fit_model(
                    d, measure, "S_baseline",
                    extra_terms=_SURPRISAL_TERMS.format(col=report_col),
                )
                dll = _loglik(res) - ll0
                _collect_rll(res, d, name, index)
                pbar.update(1)
                # nested LRT: 2*ΔLL ~ chi2(_RESID_DF) (the added D interaction
                # block). Negative ΔLL (exp no better) -> p = 1.
                p_lrt = float(chi2.sf(2.0 * max(dll, 0.0), _RESID_DF))
                rows.append(_row(
                    index, epoch_of[index], "base_surprisal", name, d, ll0 + dll, dll,
                    res, report_col, p_lrt,
                ))
    results = pd.DataFrame(rows).sort_values(["model", "index"])
    reader_ll = pd.concat(rll_rows, ignore_index=True)
    return results, reader_ll
