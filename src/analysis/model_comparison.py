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
  prompt_neutral    : off-domain scientific prior control (same corpus and
                      register as the domain priors, no domain content).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from pymer4.models import lmer  # lazy: importing loads R via rpy2
from scipy.stats import chi2, norm
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
# reader-conditioned arms compared to the baseline-surprisal model by Vuong test.
_VUONG_ARMS = {"aligned", "prompted", "prompt_neutral"}

# Paper Eq. (2) fixed effects; Eq. (4) adds the surprisal main effect(s).
# Random effects: by-subject intercept, by-word intercept + expertise slope
# (richer structures did not converge in the paper).
_BASE_TERMS = "word_length + log_word_freq + word_position + is_expert * is_technical"
_RANDOM_EFFECTS = "(1|reader_id) + (1 + is_expert|word_id)"


def _sum_code(flag: int):
    """0/1 flag -> −1/+1."""
    return 2 * flag - 1


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
        # skips (measure == 0) are owned by reading_time.clean_reading_times;
        # re-applied here only as a guard for callers passing uncleaned frames.
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


def _stat(m, name) -> float:
    """Fit statistic (``logLik`` / ``AIC`` / ``sigma`` …) from pymer4 0.9's
    ``result_fit_stats``."""
    return float(m.result_fit_stats[name].item())


def _reader_loglik(m, d) -> pd.Series:
    """Per-reader sums of pointwise CONDITIONAL log-densities of a fitted lmer.

    Normal log-density of each observation's conditional residual (around the
    fitted values, which include the BLUPs) with the ML residual sigma — an
    approximation of each reader's contribution to the fit, NOT a decomposition
    of the marginal ML log-likelihood. Meant only for PAIRED reader-level
    comparisons between fits on identical rows, where the shared random-effect
    penalty terms cancel in the pairing. Indexed by ``reader_id``.
    """

    # pymer4 augments m.data with broom-style `resid` (response - conditional
    # fitted) in the original row order.
    resid = m.data["resid"].to_numpy()
    sigma = _stat(m, "sigma")
    if len(resid) != len(d):
        raise ValueError(f"resid length {len(resid)} != data rows {len(d)}")
    dens = -0.5 * np.log(2 * np.pi * sigma**2) - resid**2 / (2 * sigma**2)
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


def _score_source(d, measure, name, index, epoch, ll_null):
    """Fit Eq. 2 + one surprisal main effect; LRT stats vs the shared
    no-surprisal null. Returns ``(results row, per-reader LL sums)``."""
    col = _SURP_COL[name]
    fit = _fit(d, measure, col)
    ll = _stat(fit, "logLik")
    dll = ll - ll_null
    row = {
        "index": index,
        "epoch": epoch,
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
    return row, _reader_loglik(fit, d)


def _vuong(ll_base: pd.Series, ll_arm: pd.Series) -> tuple[int, float, float]:
    """Paired Vuong test on per-reader LL sums: ``(n_readers, z, p)``."""
    d1, d2 = ll_base.align(ll_arm, join="inner")
    diff = d1 - d2
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        # indistinguishable fits (e.g. aligned at checkpoint 0 duplicates
        # the baseline model) — the Vuong statistic is undefined.
        return len(diff), np.nan, np.nan
    z = float(diff.mean() / (sd / np.sqrt(len(diff))))
    return len(diff), z, float(2.0 * norm.sf(abs(z)))


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    models=SURPRISAL_MODELS,
    indices=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-checkpoint surprisal-source comparison on the whole corpus (Eq. 2/4/5).

    ``surp_versions`` must hold both domains; ``indices`` restricts the sweep.
    Checkpoint-independent sources (``_CKPT_INDEP``) are fit once (``index`` /
    ``epoch`` = NA). The no-surprisal reference LL is computed once, on the
    first index's frame, and reused for every checkpoint (valid: the row set is
    checkpoint-independent, asserted per index).

    Returns ``(results, vuong, reader_ll)``:
      results — one row per source × checkpoint, every source vs the shared
        no-surprisal baseline: ``index``, ``epoch``, ``model``, ``n``, ``ll``,
        ``delta_ll``, ``chisq`` (= 2·ΔLL), ``p_lrt``, ``aic``, plus
        ``b_surprisal`` / ``se_surprisal`` (slope estimate).
      vuong — one row per reader-conditioned arm (``_VUONG_ARMS``; aligned per
        checkpoint): baseline-surprisal model vs the arm; ``model``, ``index``,
        ``epoch``, ``n_readers``, ``vuong_z``, ``p_vuong``.
      reader_ll — per-reader conditional log-lik sums (``_reader_loglik``) for
        the no-surprisal baseline (``model`` = ``base_ref``) and every source;
        columns ``model``, ``index``, ``reader_id``, ``ll_reader``.
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        all_indices = [i for i in all_indices if i in set(indices)]
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()

    d0 = build_index_df(surp_versions, rt_df, prompt_surp, all_indices[0], measure)
    dep = [m for m in models if m not in _CKPT_INDEP]
    # column check: neutral prompt absent from old caches.
    indep = [m for m in models if m in _CKPT_INDEP and _SURP_COL[m] in d0.columns]

    rows: list[dict] = []
    # one (model, index, epoch, per-reader LL sums) entry per fit; feeds both
    # the Vuong table and the reader_ll frame.
    reader_lls: list[tuple] = []

    # once: no-surprisal baseline + each indep source; per index: dep sources.
    total_fits = 1 + len(indep) + len(all_indices) * len(dep)
    with tqdm(total=total_fits, desc="lme4 fits", unit="fit") as pbar:
        pbar.set_postfix(model="no_surprisal_baseline")
        null_fit = _fit(d0, measure)
        ll_null = _stat(null_fit, "logLik")
        rll_null = _reader_loglik(null_fit, d0)
        reader_lls.append(("base_ref", pd.NA, pd.NA, rll_null))
        pbar.update(1)

        def _score(name, d, index, epoch):
            row, rll = _score_source(d, measure, name, index, epoch, ll_null)
            rows.append(row)
            reader_lls.append((name, index, epoch, rll))
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
                _score(name, d, index, epoch_of[index])

    # Vuong table: baseline-surprisal model vs each reader-conditioned arm
    # (module doc §2). Per-reader LL sums cluster the repeated measures; the
    # arms were fit on the same rows as the baseline (asserted above).
    base_rll = next((r for name, _, _, r in reader_lls if name == "baseline"), None)
    vuong_rows = []
    if base_rll is not None:
        for name, index, epoch, rll in reader_lls:
            if name not in _VUONG_ARMS:
                continue
            n_readers, z, p = _vuong(base_rll, rll)
            vuong_rows.append(
                {
                    "index": index,
                    "epoch": epoch,
                    "model": name,
                    "n_readers": n_readers,
                    "vuong_z": z,
                    "p_vuong": p,
                }
            )

    results = pd.DataFrame(rows).sort_values(["model", "index"])
    vuong = pd.DataFrame(
        vuong_rows,
        columns=["index", "epoch", "model", "n_readers", "vuong_z", "p_vuong"],
    ).sort_values(["model", "index"])
    reader_ll = pd.concat(
        [
            rll.rename("ll_reader")
            .rename_axis("reader_id")
            .reset_index()
            .assign(model=name, index=index)
            for name, index, _, rll in reader_lls
        ],
        ignore_index=True,
    )
    return results, vuong, reader_ll
