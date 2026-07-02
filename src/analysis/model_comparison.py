"""Surprisal comparison — EXACTLY Škrjanec & Demberg (2026), J. Mem. Lang. 146.

For each surprisal source S we fit the paper's two nested lme4 models (via pymer4)
and report ΔLL = LL_experimental − LL_baseline (Eq. 5), the paper's measure of a
source's predictive power.

Baseline model (Eq. 2), NO surprisal:
    log(RT) ~ Length + LogFreq + Position + Expertise*Terminology
              + (1|SubjectID) + (1 + Expertise|WordID)
Experimental model (Eq. 4): baseline + Surprisal, a SINGLE PLAIN MAIN EFFECT
(no interaction, no residualization).

Exactly as the paper:
  * all continuous predictors (Length, LogFreq, Position, Surprisal) are scaled
    and centered;
  * Expertise and Terminology are sum-coded (−1 / +1); the two technical levels
    (general + expert) are merged into one "technical" level;
  * Position is the word's position IN TEXT;
  * ML fit (REML=False) so the log-likelihoods are comparable;
  * the shared no-surprisal baseline is the reference for EVERY source.

Surprisal source S is one of:
  baseline          : un-adapted step-0 model (same for every reader)
  physics / biology : the domain fine-tuned model (every reader)
  aligned           : by READER discipline — physics surprisal for physicists,
                      biology for biologists (reader-aligned surprisal, Study 2)
  prompted          : baseline weights + discipline-matched prior passage
  prompt_neutral    : length-matched non-domain prior control

Significance of adding a source's surprisal is the nested likelihood-ratio test
(1 df, the single surprisal term); ``p_lrt`` is its p-value.
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

SURPRISAL_MODELS = (
    "baseline", "physics", "biology", "aligned", "prompted", "prompt_neutral"
)

# Surprisal source -> its scaled column built in ``_prep_models``.
_SOURCE_COL = {
    "baseline": "z_S_baseline",
    "physics": "z_S_physics",
    "biology": "z_S_biology",
    "aligned": "z_S_aligned",
    "prompted": "z_S_prompted",
    "prompt_neutral": "z_S_prompt_neutral",
}
# sources whose surprisal doesn't depend on the DAPT checkpoint: fit once.
_CKPT_INDEP = {"baseline", "prompted", "prompt_neutral"}


def _prep_models(df, measure):
    """One row per reader×word with every scaled surprisal column + covariates.

    ``df`` must carry ``s_base`` / ``s_phys`` / ``s_bio`` / ``s_prompt_*`` plus the
    reading measures. The neutral-prior column is optional (older caches predate it).
    """
    has_neutral = "s_prompt_neutral" in df.columns
    d = df[df[measure] > 0].copy()
    subset = [
        measure,
        "word_length",
        "lemma_frequency_normalized",
        "s_base",
        "s_phys",
        "s_bio",
        "s_prompt_phys",
        "s_prompt_bio",
    ]
    if has_neutral:
        subset.append("s_prompt_neutral")
    d = d.dropna(subset=subset)
    d = d[d["word_length"] > 0]
    # dlexDB lemma freq: +1 smoothing then log (paper).
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])

    # per-reader surprisal sources. aligned / prompted pick the discipline-matched
    # column (1 = physicist, 0 = biologist).
    is_physicist = d["reader_discipline_numeric"] == 1
    d["S_baseline"] = d["s_base"]
    d["S_physics"] = d["s_phys"]
    d["S_biology"] = d["s_bio"]
    d["S_aligned"] = np.where(is_physicist, d["s_phys"], d["s_bio"])
    d["S_prompted"] = np.where(is_physicist, d["s_prompt_phys"], d["s_prompt_bio"])
    if has_neutral:
        d["S_prompt_neutral"] = d["s_prompt_neutral"]

    # Position IN TEXT (paper: "Word position in text").
    d["word_position"] = d["word_index_in_text"].astype(float)

    # terminology: merge general + expert technical into one "technical" level.
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    # SUM (deviation) coding for the two factors (paper): −1 novice/common,
    # +1 expert/technical.
    d["is_expert"] = 2 * d["is_expert"] - 1
    d["is_technical"] = 2 * d["is_technical"] - 1

    # scale + center every continuous predictor (paper: "All continuous predictors
    # were scaled and centered"): length, log frequency, position, and each
    # surprisal source. Scaling is affine — it leaves ΔLL / the LRT unchanged.
    def _z(s):
        return (s - s.mean()) / s.std()

    d["z_length"] = _z(d["word_length"])
    d["z_logfreq"] = _z(d["log_word_freq"])
    d["z_position"] = _z(d["word_position"])
    sources = ["baseline", "physics", "biology", "aligned", "prompted"]
    if has_neutral:
        sources.append("prompt_neutral")
    for name in sources:
        d[f"z_S_{name}"] = _z(d[f"S_{name}"])

    # single grouping key for the by-word random effect (lme4 word_id).
    d["word_id"] = d["text_id"].astype(str) + "_" + d["word_index_in_text"].astype(str)
    return d


# Paper Eq. (2) baseline fixed effects (WITHOUT surprisal) + Eq. (4) adds a single
# ``Surprisal`` main effect. Random effects: by-subject intercept, by-word intercept
# and by-word expertise slope (richer structures did not converge in the paper).
_BASE_TERMS = "z_length + z_logfreq + z_position + is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"
# the experimental model adds exactly ONE fixed term (the surprisal main effect),
# so the nested LRT has 1 degree of freedom.
_LRT_DF = 1


def _fit(d, measure, surprisal_col=None):
    """lme4 mixed model ``log(measure) ~ <base> [+ surprisal] + <re>`` (paper Eq. 2/4).

    ML fit (``REML=False``) via pymer4 (lme4) so log-likelihoods are comparable for
    the nested LRT / ΔLL. Without ``surprisal_col`` this is the no-surprisal baseline
    (Eq. 2). Coefficient p-values are Wald z; the Satterthwaite Hessian is skipped.
    Returns the fitted ``pymer4.models.lmer``.
    """
    import polars as pl
    from pymer4.models import lmer  # lazy: importing loads R via rpy2

    rhs = _BASE_TERMS
    if surprisal_col is not None:
        rhs += " + " + surprisal_col  # single plain main effect (no interaction)
    dd = d.copy()
    dd["resp_log"] = np.log(dd[measure].to_numpy())  # valid R identifier
    formula = f"resp_log ~ {rhs} + {_RANDOM_EFFECTS}"
    m = lmer(formula, data=pl.from_pandas(dd))
    # pymer4 streams R convergence chatter to stdout; silence it for the sweep.
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
    """Fixed-effect ``field`` ('estimate'/'std_error'/'p_value') for term ``name``.

    Reads pymer4 0.9's ``result_fit`` (a polars frame keyed by ``term``).
    """
    try:
        rf = m.result_fit
        row = rf.filter(rf["term"] == name)
        return float(row[field][0]) if row.height else np.nan
    except (KeyError, AttributeError, TypeError, IndexError):
        return np.nan


def _row(index, epoch, model, d, ll, dll, fit, scol, p_lrt) -> dict:
    """One result row for a surprisal source.

    ``b_surprisal`` / ``se_surprisal`` / ``p_surprisal`` are the source's surprisal
    main-effect slope (Wald z); ``delta_ll`` = LL_experimental − LL_baseline (Eq. 5);
    ``p_lrt`` = nested LRT p (1 df). ``index`` / ``epoch`` are NA for a
    checkpoint-independent source (fit once).
    """
    return {
        "index": index,
        "epoch": epoch,
        "ref": "no_surprisal",
        "model": model,
        "n": len(d),
        "ll": ll,
        "delta_ll": dll,
        "b_surprisal": _coef(fit, scol, "estimate"),
        "se_surprisal": _coef(fit, scol, "std_error"),
        "p_surprisal": _coef(fit, scol, "p_value"),
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
    """Per-checkpoint surprisal-source comparison on the whole corpus (paper Eq. 2/4/5).

    Every source is compared to the SAME shared no-surprisal baseline (Eq. 2). For
    each source S: ``delta_ll`` = LL(baseline + S) − LL(baseline), and ``p_lrt`` is
    the nested LRT (1 df). ``surp_versions`` must hold both domains; ``indices``
    restricts the checkpoint sweep.

    baseline and the prompted arms (``_CKPT_INDEP``) don't depend on the checkpoint,
    so each is fit ONCE (``index`` / ``epoch`` = NA). The no-surprisal baseline LL is
    also computed once and reused (valid: the row set is checkpoint-independent,
    asserted per index).

    Returns ``(results, reader_ll)``:
      results — columns ``index``, ``epoch``, ``ref``, ``model``, ``n``, ``ll``,
        ``delta_ll``, ``b_surprisal``, ``se_surprisal``, ``p_surprisal``, ``p_lrt``.
      reader_ll — per-reader conditional log-lik sums (``_reader_loglik``) for the
        no-surprisal baseline (``model`` = ``base_ref``) and every source; columns
        ``model``, ``index``, ``reader_id``, ``ll_reader``. Feeds ``claim_tests``.
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        all_indices = [i for i in all_indices if i in set(indices)]
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()

    dep_models = [m for m in models if m not in _CKPT_INDEP]
    indep_models = [m for m in models if m in _CKPT_INDEP]

    # once: no-surprisal baseline + each indep source; per index: dependent sources.
    total_fits = 1 + len(indep_models) + len(all_indices) * len(dep_models)
    rows = []
    rll_rows: list[pd.DataFrame] = []

    def _collect_rll(fit, d, model, index):
        rll = _reader_loglik(fit, d, measure).rename("ll_reader").reset_index()
        rll.columns = ["reader_id", "ll_reader"]
        rll.insert(0, "model", model)
        rll.insert(1, "index", index)
        rll_rows.append(rll)

    d0 = None  # first index' frame; the baseline / indep fits are reused off it.
    ll_null = None
    with tqdm(total=total_fits, desc="lme4 fits", unit="fit") as pbar:
        for index in all_indices:
            d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
            if d0 is None:
                d0 = d
                pbar.set_postfix(model="no_surprisal_baseline")
                null_fit = _fit(d, measure, None)
                ll_null = _loglik(null_fit)
                _collect_rll(null_fit, d, "base_ref", pd.NA)
                pbar.update(1)
                for name in indep_models:
                    col = _SOURCE_COL[name]
                    if col not in d.columns:  # e.g. neutral absent from an old cache
                        pbar.update(1)
                        continue
                    pbar.set_postfix(model=name)
                    fit = _fit(d, measure, col)
                    dll = _loglik(fit) - ll_null
                    _collect_rll(fit, d, name, pd.NA)
                    pbar.update(1)
                    p_lrt = float(chi2.sf(2.0 * max(dll, 0.0), _LRT_DF))
                    rows.append(_row(pd.NA, pd.NA, name, d, ll_null + dll, dll, fit,
                                     col, p_lrt))
            else:
                # reusing d0's ll_null for this index requires the same rows.
                assert d[WORD_KEY].equals(d0[WORD_KEY]), (
                    "checkpoint row sets differ; baseline reference not reusable"
                )
            for name in dep_models:
                col = _SOURCE_COL[name]
                pbar.set_postfix(index=index, model=name)
                fit = _fit(d, measure, col)
                dll = _loglik(fit) - ll_null
                _collect_rll(fit, d, name, index)
                pbar.update(1)
                # nested LRT: 2*ΔLL ~ chi2(1) (the added surprisal term). Negative
                # ΔLL (surprisal no better) -> p = 1.
                p_lrt = float(chi2.sf(2.0 * max(dll, 0.0), _LRT_DF))
                rows.append(_row(index, epoch_of[index], name, d, ll_null + dll, dll,
                                 fit, col, p_lrt))
    results = pd.DataFrame(rows).sort_values(["model", "index"])
    reader_ll = pd.concat(rll_rows, ignore_index=True)
    return results, reader_ll
