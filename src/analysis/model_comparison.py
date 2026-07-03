"""Surprisal comparison — EXACTLY Škrjanec & Demberg (2026), J. Mem. Lang. 146.

For each surprisal source S we fit the paper's two nested lme4 models (via
pymer4) and report ΔLL = LL_experimental − LL_reference (Eq. 5).

Baseline model (Eq. 2), NO surprisal:
    log(RT) ~ Length + LogFreq + Position + Expertise*Terminology
              + (1|SubjectID) + (1 + Expertise|WordID)
Experimental model (Eq. 4): baseline + Surprisal, a SINGLE PLAIN MAIN EFFECT
(no interaction, no residualization).

Exactly as the paper: continuous predictors scaled and centered; Expertise and
Terminology sum-coded (−1/+1), with the two technical levels merged into one
"technical" level; Position is the word's position IN TEXT; ML fits
(REML=False) so log-likelihoods are comparable.

Surprisal sources:
  baseline          : un-adapted step-0 model (same for every reader).
  physics / biology : the domain fine-tuned model (every reader).
  aligned           : by READER discipline — physics surprisal for physicists,
                      biology for biologists (Study 2).
  prompted          : baseline weights + discipline-matched prior passage.
  prompt_neutral    : length-matched non-domain prior control.

Each source is tested with a nested 1-df LRT (``p_lrt``) against its reference
model (``ref``). Most sources are referenced to the no-surprisal baseline
(Eq. 2); the reader-conditioned arms (``aligned`` / ``prompted``) instead enter
ON TOP of plain baseline surprisal and are referenced to the SURPRISAL baseline
(Eq. 2 + baseline surprisal) — i.e. they test whether reader-matched surprisal
helps BEYOND plain surprisal.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import polars as pl
import rpy2.robjects as ro
from pymer4.models import lmer  # lazy: importing loads R via rpy2
from scipy.stats import chi2
from tqdm import tqdm

from src.config import WORD_KEY

SURPRISAL_MODELS = (
    "baseline",
    "physics",
    "biology",
    "aligned",
    "prompted",
    "prompt_neutral",
)
# sources whose surprisal doesn't depend on the DAPT checkpoint: fit once.
_CKPT_INDEP = {"baseline", "prompted", "prompt_neutral"}
# reader-conditioned arms, referenced to the surprisal baseline (module doc).
_VS_SURP_BASELINE = {"aligned", "prompted"}

# Paper Eq. (2) fixed effects; Eq. (4) adds the surprisal main effect(s).
# Random effects: by-subject intercept, by-word intercept + expertise slope
# (richer structures did not converge in the paper).
_BASE_TERMS = "z_length + z_logfreq + z_position + is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"


def _prep_models(df, measure):
    """One row per reader×word with every scaled surprisal column + covariates.

    ``df`` must carry ``s_base`` / ``s_phys`` / ``s_bio`` / ``s_prompt_*`` plus
    the reading measures; ``s_prompt_neutral`` is optional (older caches
    predate it).
    """
    sources = ["baseline", "physics", "biology", "aligned", "prompted"]
    raw_cols = ["s_base", "s_phys", "s_bio", "s_prompt_phys", "s_prompt_bio"]
    if "s_prompt_neutral" in df.columns:
        sources.append("prompt_neutral")
        raw_cols.append("s_prompt_neutral")

    d = df[df[measure] > 0].copy()
    d = d.dropna(
        subset=[measure, "word_length", "lemma_frequency_normalized", *raw_cols]
    )
    d = d[d["word_length"] > 0]

    # dlexDB lemma freq: +1 smoothing then log (paper).
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])
    # Position IN TEXT (paper: "Word position in text").
    d["word_position"] = d["word_index_in_text"].astype(float)

    # per-reader sources: aligned / prompted pick the discipline-matched column
    # (reader_discipline_numeric: 1 = physicist, 0 = biologist).
    is_physicist = d["reader_discipline_numeric"] == 1
    d["S_baseline"] = d["s_base"]
    d["S_physics"] = d["s_phys"]
    d["S_biology"] = d["s_bio"]
    d["S_aligned"] = np.where(is_physicist, d["s_phys"], d["s_bio"])
    d["S_prompted"] = np.where(is_physicist, d["s_prompt_phys"], d["s_prompt_bio"])
    if "prompt_neutral" in sources:
        d["S_prompt_neutral"] = d["s_prompt_neutral"]

    # terminology merges general + expert technical; both factors sum
    # (deviation) coded (paper): −1 novice/common, +1 expert/technical.
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    d["is_expert"] = 2 * d["is_expert"] - 1
    d["is_technical"] = 2 * d["is_technical"] - 1

    # scale + center every continuous predictor (paper). Scaling is affine —
    # it leaves ΔLL / the LRT unchanged.
    def _z(s):
        return (s - s.mean()) / s.std()

    d["z_length"] = _z(d["word_length"])
    d["z_logfreq"] = _z(d["log_word_freq"])
    d["z_position"] = _z(d["word_position"])
    for name in sources:
        d[f"z_S_{name}"] = _z(d[f"S_{name}"])

    # single grouping key for the by-word random effect (lme4 word_id).
    d["word_id"] = d["text_id"].astype(str) + "_" + d["word_index_in_text"].astype(str)
    return d


def _fit(d, measure, surprisal_cols=None):
    """ML lmer fit of ``log(measure) ~ <base> [+ surprisal…] + <random>`` (Eq. 2/4).

    ``surprisal_cols``: None (no-surprisal baseline, Eq. 2), one column name,
    or a list — each enters as a plain main effect. ML (REML=False) so the LLs
    feed the nested LRT / ΔLL. Coefficient p-values are Wald z.
    """

    rhs = _BASE_TERMS
    if surprisal_cols:
        cols = (
            [surprisal_cols]
            if isinstance(surprisal_cols, str)
            else list(surprisal_cols)
        )
        rhs += " + " + " + ".join(cols)
    dd = d.copy()
    dd["resp_log"] = np.log(dd[measure].to_numpy())  # valid R identifier
    m = lmer(f"resp_log ~ {rhs} + {_RANDOM_EFFECTS}", data=pl.from_pandas(dd))
    # pymer4 streams R convergence chatter to stdout; silence it for the sweep.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(summary=False, REML=False, conf_method="wald")
    return m


def _loglik(m) -> float:
    """ML log-likelihood of a fitted pymer4 0.9 ``lmer`` (via the R model)."""

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

    fitted = np.asarray(ro.r("fitted")(m.r_model), dtype=float).ravel()
    sigma = float(np.asarray(ro.r("sigma")(m.r_model)).ravel()[0])
    if len(fitted) != len(d):
        raise ValueError(f"fitted() length {len(fitted)} != data rows {len(d)}")
    resp = np.log(d[measure].to_numpy(dtype=float))
    dens = -0.5 * np.log(2 * np.pi * sigma**2) - (resp - fitted) ** 2 / (2 * sigma**2)
    return pd.Series(dens).groupby(d["reader_id"].to_numpy()).sum()


def build_index_df(surp_versions, rt_df, prompt_surp, index, measure):
    """Reader×word frame with all surprisal columns + covariates at one checkpoint.

    Physics/biology checkpoints are paired by ``index`` (0 = baseline), not
    ``epoch`` (step counts differ). Index 0 supplies ``s_base``. ``prompt_surp``
    carries the checkpoint-independent prompted columns.
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
    """Fixed-effect ``field`` ('estimate'/'std_error'/'p_value') for term ``name``,
    from pymer4 0.9's ``result_fit`` (a polars frame keyed by ``term``)."""
    rf = m.result_fit
    row = rf.filter(rf["term"] == name)
    return float(row[field][0]) if row.height else np.nan


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    models=SURPRISAL_MODELS,
    indices=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-checkpoint surprisal-source comparison on the whole corpus (Eq. 2/4/5).

    ``surp_versions`` must hold both domains; ``indices`` restricts the sweep.
    Checkpoint-independent sources (``_CKPT_INDEP``) are fit once (``index`` /
    ``epoch`` = NA). The two reference LLs are computed once, on the first
    index's frame, and reused for every checkpoint (valid: the row set is
    checkpoint-independent, asserted per index).

    Returns ``(results, reader_ll)``:
      results — one row per source × checkpoint: ``index``, ``epoch``, ``ref``,
        ``model``, ``n``, ``ll``, ``delta_ll``, ``b_surprisal``,
        ``se_surprisal``, ``p_surprisal`` (Wald z on the source's slope),
        ``p_lrt`` (nested 1-df LRT vs ``ref``).
      reader_ll — per-reader conditional log-lik sums (``_reader_loglik``) for
        the no-surprisal baseline (``model`` = ``base_ref``) and every source;
        columns ``model``, ``index``, ``reader_id``, ``ll_reader``. Feeds
        ``claim_tests``.
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        all_indices = [i for i in all_indices if i in set(indices)]
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()
    dep = [m for m in models if m not in _CKPT_INDEP]
    indep = [m for m in models if m in _CKPT_INDEP]

    rows: list[dict] = []
    rll_frames: list[pd.DataFrame] = []

    def _collect_rll(fit, d, model, index):
        ll = _reader_loglik(fit, d, measure)
        rll_frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "index": index,
                    "reader_id": ll.index,
                    "ll_reader": ll.values,
                }
            )
        )

    d0 = build_index_df(surp_versions, rt_df, prompt_surp, all_indices[0], measure)
    # once: no-surprisal baseline + each indep source; per index: dep sources.
    total_fits = 1 + len(indep) + len(all_indices) * len(dep)
    with tqdm(total=total_fits, desc="lme4 fits", unit="fit") as pbar:
        pbar.set_postfix(model="no_surprisal_baseline")
        null_fit = _fit(d0, measure)
        ll_null = _loglik(null_fit)
        _collect_rll(null_fit, d0, "base_ref", pd.NA)
        pbar.update(1)

        # surprisal baseline (Eq. 2 + plain baseline surprisal): the ``baseline``
        # source's own fit AND the reference for aligned / prompted. Fit once.
        base_fit = _fit(d0, measure, "z_S_baseline")
        ll_base = _loglik(base_fit)

        def _score(name, d, index, epoch):
            """Fit source ``name`` on ``d``; record its result row + reader LLs."""
            if name == "baseline":
                fit, ref_ll, ref = base_fit, ll_null, "no_surprisal"
            elif name in _VS_SURP_BASELINE:
                fit = _fit(d, measure, ["z_S_baseline", f"z_S_{name}"])
                ref_ll, ref = ll_base, "surprisal_baseline"
            else:
                fit = _fit(d, measure, f"z_S_{name}")
                ref_ll, ref = ll_null, "no_surprisal"
            dll = _loglik(fit) - ref_ll
            _collect_rll(fit, d, name, index)
            pbar.update(1)
            rows.append(
                {
                    "index": index,
                    "epoch": epoch,
                    "ref": ref,
                    "model": name,
                    "n": len(d),
                    "ll": ref_ll + dll,
                    "delta_ll": dll,
                    "b_surprisal": _coef(fit, f"z_S_{name}", "estimate"),
                    "se_surprisal": _coef(fit, f"z_S_{name}", "std_error"),
                    "p_surprisal": _coef(fit, f"z_S_{name}", "p_value"),
                    # nested LRT: 2·ΔLL ~ chi2(1), the single added surprisal term.
                    # Negative ΔLL (no better than the reference) -> p = 1.
                    "p_lrt": float(chi2.sf(2.0 * max(dll, 0.0), 1)),
                }
            )

        for name in indep:
            if f"z_S_{name}" not in d0.columns:  # neutral absent from an old cache
                pbar.update(1)
                continue
            pbar.set_postfix(model=name)
            _score(name, d0, pd.NA, pd.NA)

        for i, index in enumerate(all_indices):
            if i == 0:
                d = d0
            else:
                d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
                # reusing d0's reference LLs at this index requires the same rows.
                assert d[WORD_KEY].equals(d0[WORD_KEY]), (
                    "checkpoint row sets differ; baseline reference not reusable"
                )
            for name in dep:
                pbar.set_postfix(index=index, model=name)
                _score(name, d, index, epoch_of[index])

    results = pd.DataFrame(rows).sort_values(["model", "index"])
    reader_ll = pd.concat(rll_frames, ignore_index=True)
    return results, reader_ll
