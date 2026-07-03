"""Surprisal comparison — EXACTLY Škrjanec & Demberg (2026), J. Mem. Lang. 146.

For each surprisal source S we fit the paper's two nested lme4 models (via
pymer4) and report ΔLL = LL_experimental − LL_reference (Eq. 5).

Baseline model (Eq. 2), NO surprisal:
    log(RT) ~ Length + LogFreq + Position + Expertise*Terminology
              + (1|SubjectID) + (1 + Expertise|WordID)
Experimental model (Eq. 4): baseline + Surprisal, a SINGLE PLAIN MAIN EFFECT
(no interaction, no residualization).

As the paper: Expertise and Terminology sum-coded (−1/+1), with the two
technical levels merged into one "technical" level; Position is the word's
position IN TEXT; ML fits (REML=False) so log-likelihoods are comparable.
Predictors enter on their natural scale — the paper z-scales them, but scaling
is affine and leaves ΔLL / the LRT unchanged.

Surprisal sources:
  baseline          : un-adapted step-0 model (same for every reader).
  physics / biology : the domain fine-tuned model (every reader).
  aligned           : by READER discipline — physics surprisal for physicists,
                      biology for biologists (Study 2).
  prompted          : baseline weights + discipline-matched prior passage.
  prompt_neutral    : length-matched non-domain prior control.

Each source is tested with a nested 1-df LRT (``p_lrt``) against its reference
model (``ref``). Most sources are referenced to the no-surprisal baseline
(Eq. 2); the ``aligned`` / ``prompted`` / ``prompt_neutral`` arms instead enter
ON TOP of plain baseline surprisal and are referenced to the SURPRISAL baseline
(Eq. 2 + baseline surprisal) — i.e. they test whether the added surprisal helps
BEYOND plain surprisal. ``prompt_neutral`` shares the prompted reference so the
domain-prompt lift and its non-domain control are on one scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import rpy2.robjects as ro
from pymer4.models import lmer  # lazy: importing loads R via rpy2
from scipy.stats import chi2
from tqdm import tqdm

from src.config import WORD_KEY

# surprisal column per source; aligned / prompted are built in _prep_models.
_SURP_COL = {
    "baseline": "s_base",
    "physics": "s_phys",
    "biology": "s_bio",
    "aligned": "s_aligned",
    "prompted": "s_prompted",
    "prompt_neutral": "s_prompt_neutral",
}
SURPRISAL_MODELS = tuple(_SURP_COL)
# sources whose surprisal doesn't depend on the DAPT checkpoint: fit once.
_CKPT_INDEP = {"baseline", "prompted", "prompt_neutral"}
# arms referenced to the surprisal baseline — tested for lift BEYOND plain
# baseline surprisal (module doc). prompt_neutral shares prompted's reference so
# the two prompted arms are read on the SAME scale (neutral = matched control).
_VS_SURP_BASELINE = {"aligned", "prompted", "prompt_neutral"}

# Paper Eq. (2) fixed effects; Eq. (4) adds the surprisal main effect(s).
# Random effects: by-subject intercept, by-word intercept + expertise slope
# (richer structures did not converge in the paper).
_BASE_TERMS = "word_length + log_word_freq + word_position + is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"


def _sum_code(flag):
    """0/1 flag -> −1/+1."""
    return 2 * flag.astype(int) - 1


def _prep_models(df, measure):
    """One row per reader×word with every surprisal column + covariates.

    ``df`` must carry ``s_base`` / ``s_phys`` / ``s_bio`` / ``s_prompt_*`` plus
    the reading measures; ``s_prompt_neutral`` is optional (older caches
    predate it).
    """
    raw_cols = ["s_base", "s_phys", "s_bio", "s_prompt_phys", "s_prompt_bio"]
    if "s_prompt_neutral" in df.columns:
        raw_cols.append("s_prompt_neutral")

    # reader_discipline_numeric: 1 = physicist, 0 = biologist.
    def is_physicist(x):
        return x["reader_discipline_numeric"] == 1

    return (
        df.loc[lambda x: x[measure] > 0]
        .dropna(
            subset=[measure, "word_length", "lemma_frequency_normalized", *raw_cols]
        )
        .loc[lambda x: x["word_length"] > 0]
        .assign(
            # dlexDB lemma freq: +1 smoothing then log (paper).
            log_word_freq=lambda x: np.log1p(x["lemma_frequency_normalized"]),
            # Position IN TEXT (paper: "Word position in text").
            word_position=lambda x: x["word_index_in_text"].astype(float),
            # reader-conditioned sources: pick the discipline-matched column.
            s_aligned=lambda x: x["s_phys"].where(is_physicist(x), x["s_bio"]),
            s_prompted=lambda x: x["s_prompt_phys"].where(
                is_physicist(x), x["s_prompt_bio"]
            ),
            # terminology merges general + expert technical (paper).
            is_technical=lambda x: _sum_code(
                (x["is_expert_technical_term"] == 1)
                | (x["is_general_technical_term"] == 1)
            ),
            is_expert=lambda x: _sum_code(x["is_expert"]),
            # single grouping key for the by-word random effect (lme4 word_id).
            word_id=lambda x: (
                x["text_id"].astype(str) + "_" + x["word_index_in_text"].astype(str)
            ),
        )
    )


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


def _r(m, fn) -> np.ndarray:
    """R ``fn(<fitted lmer>)`` on a pymer4 0.9 model, as a flat float array."""
    return np.asarray(ro.r(fn)(m.r_model), dtype=float).ravel()


def _loglik(m) -> float:
    """ML log-likelihood of a fitted lmer."""
    return float(_r(m, "logLik")[0])


def _reader_loglik(m, d, measure) -> pd.Series:
    """Per-reader sums of pointwise CONDITIONAL log-densities of a fitted lmer.

    Normal log-density of each observation around ``fitted()`` (which includes
    the BLUPs) with the ML residual sigma — an approximation of each reader's
    contribution to the fit, NOT a decomposition of the marginal ML
    log-likelihood. Meant only for PAIRED reader-level comparisons between fits
    on identical rows (claim_tests), where the shared random-effect penalty
    terms cancel in the pairing. Indexed by ``reader_id``.
    """

    fitted = _r(m, "fitted")
    sigma = float(_r(m, "sigma")[0])
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
        rll_frames.append(
            _reader_loglik(fit, d, measure)
            .rename("ll_reader")
            .rename_axis("reader_id")
            .reset_index()
            .assign(model=model, index=index)
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
        base_fit = _fit(d0, measure, "s_base")
        ll_base = _loglik(base_fit)

        def _score(name, d, index, epoch):
            """Fit source ``name`` on ``d``; record its result row + reader LLs."""
            col = _SURP_COL[name]
            if name == "baseline":
                fit, ref_ll, ref = base_fit, ll_null, "no_surprisal"
            elif name in _VS_SURP_BASELINE:
                fit = _fit(d, measure, ["s_base", col])
                ref_ll, ref = ll_base, "surprisal_baseline"
            else:
                fit = _fit(d, measure, col)
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
                    "b_surprisal": _coef(fit, col, "estimate"),
                    "se_surprisal": _coef(fit, col, "std_error"),
                    "p_surprisal": _coef(fit, col, "p_value"),
                    # nested LRT: 2·ΔLL ~ chi2(1), the single added surprisal term.
                    # Negative ΔLL (no better than the reference) -> p = 1.
                    "p_lrt": float(chi2.sf(2.0 * max(dll, 0.0), 1)),
                }
            )

        for name in indep:
            if _SURP_COL[name] not in d0.columns:  # neutral absent from old cache
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
