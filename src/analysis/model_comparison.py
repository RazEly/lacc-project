"""Four-model surprisal comparison: reader-aligned vs single model (step 5).

On the WHOLE corpus (both text domains, all readers) compare how well four
surprisal sources predict reading time, each in the same mixed model:

    RT ~ log_word_freq + log_word_length + S + (1 | reader)

with S one of:
  baseline : the un-adapted step-0 model's surprisal (same for every reader)
  physics  : the physics fine-tuned model's surprisal (every reader)
  biology  : the biology fine-tuned model's surprisal (every reader)
  aligned  : reader-aligned — physics surprisal for physicists, biology
             surprisal for biologists, i.e.
             physicist * S_physics + (1 - physicist) * S_biology.

In ``baseline`` / ``physics`` / ``biology`` the reader's discipline is ignored;
only ``aligned`` matches the language model to the reader's field. Alignment is
by READER discipline, not text domain, so a physicist gets physics surprisal
even on a biology text. Every model adds a single surprisal slope, so their
log-likelihoods are directly comparable. ΔLL is measured against the
no-surprisal baseline ``RT ~ freq + length + (1|reader)``. The question: does
``aligned`` improve the fit over the three single-model sources? RT is the raw
measure (swap to ``np.log`` here for log-RT). Only ``(1|reader)`` is random, so
each fit is sub-second.

Significance of the ΔLL gaps is tested in ``stats.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis.correlation import WORD_KEY

SURPRISAL_MODELS = ("baseline", "physics", "biology", "aligned", "prompted")


def _prep_models(df, measure):
    """One row per reader×word with the five surprisal columns + covariates.

    ``df`` must already carry per-word ``s_base`` / ``s_phys`` / ``s_bio``
    surprisal (base, physics-adapted, biology-adapted), the prompted-baseline
    columns ``s_prompt_phys`` / ``s_prompt_bio``, and the reading measures.
    """
    d = df[df[measure] > 0].copy()
    d = d.dropna(
        subset=[measure, "word_length", "lemma_frequency_normalized",
                "s_base", "s_phys", "s_bio", "s_prompt_phys", "s_prompt_bio"]
    )
    d = d[d["word_length"] > 0]
    # dlexDB lemma freq (per million) is heavily right-skewed; log1p keeps the
    # zero-frequency words (missing dlexDB entries) finite.
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])
    d["log_word_length"] = np.log(d["word_length"])
    # reader discipline (1 = physics, 0 = biology); see data.add_expertise.
    physicist = (d["reader_discipline_numeric"] == 1).to_numpy(dtype=float)
    d["S_baseline"] = d["s_base"]
    d["S_physics"] = d["s_phys"]
    d["S_biology"] = d["s_bio"]
    # reader-aligned: discipline-matched LM. ``aligned`` swaps the model WEIGHTS
    # (fine-tuned), ``prompted`` keeps baseline weights but prepends a
    # discipline-matched system prompt — both mixed by reader discipline.
    d["S_aligned"] = physicist * d["s_phys"] + (1.0 - physicist) * d["s_bio"]
    d["S_prompted"] = (
        physicist * d["s_prompt_phys"] + (1.0 - physicist) * d["s_prompt_bio"]
    )
    # extra predictors for the richer model specs (see MODEL_SPECS).
    # ``is_expert`` (reader major == text domain) is added by data.add_expertise.
    d["word_position"] = (
        (d["word_index_in_text"] - d["word_index_in_text"].mean())
        / d["word_index_in_text"].std()
    )
    # PoTeC terminology levels: 0 common, 1 general-technical, 2 expert/domain.
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    d["is_domain_term"] = (d["is_expert_technical_term"] == 1).astype(int)
    return d


# Fixed-effects specifications swept alongside the five surprisal sources. Each
# spec is (terms with NO surprisal, terms ADDED with surprisal); ΔLL is the
# log-likelihood gain of the second over the first, so the surprisal term may
# itself be an interaction. ``{col}`` is the surprisal column. ΔLL is comparable
# across surprisal models WITHIN a spec, not across specs (different baselines).
MODEL_SPECS = {
    # word-frequency / length covariates + plain surprisal slope (the original).
    "covariates": ("log_word_freq + log_word_length", "{col}"),
    # minimal: surprisal × reader expertise only.
    "expertise": ("is_expert", "{col} * is_expert"),
    # full: covariates + position + expertise×terminology, surprisal interacting
    # with expertise and with technical-term status.
    "full": (
        "log_word_freq + log_word_length + word_position"
        " + is_expert * is_technical + is_domain_term",
        "{col} * is_expert + {col}:is_technical",
    ),
}


def _fit_model(d, measure, surprisal_col=None, spec="covariates"):
    """Mixed model ``measure ~ <spec> [+ surprisal terms] + (1|reader)``.

    ``spec`` selects a fixed-effects formula from ``MODEL_SPECS``. With
    ``surprisal_col`` the spec's surprisal terms (possibly interactions) are
    added; without it the no-surprisal reference is fit.
    """
    base, surp_tmpl = MODEL_SPECS[spec]
    rhs = base
    if surprisal_col is not None:
        rhs += " + " + surp_tmpl.format(col=surprisal_col)
    md = smf.mixedlm(f"{measure} ~ {rhs}", d, groups=d["reader_id"])
    return md.fit(reml=False)  # ML so LLs are comparable across the surprisal term


def build_index_df(surp_versions, rt_df, prompt_surp, index, measure):
    """Reader×word frame with all surprisal columns + covariates at one checkpoint.

    Physics and biology checkpoints are paired by ``index`` (0 = baseline), not
    ``epoch``: the two runs have different step counts, so their ``epoch`` floats
    drift apart and would mis-pair. The index-0 checkpoint is the un-adapted
    model and supplies the domain-agnostic ``s_base``. ``prompt_surp`` carries the
    checkpoint-independent prompted-baseline columns ``s_prompt_phys`` /
    ``s_prompt_bio`` (per-word, both prompts).
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
    return _prep_models(merged, measure)


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    spec="covariates",
) -> pd.DataFrame:
    """Per-checkpoint five-model surprisal comparison on the whole corpus.

    ``surp_versions`` must contain BOTH a ``physics`` and a ``biology`` domain
    (fine-tune both). ``prompt_surp`` holds the per-word prompted-baseline columns
    ``s_prompt_phys`` / ``s_prompt_bio``. ``spec`` selects the fixed-effects
    formula from ``MODEL_SPECS``. For every checkpoint the physics, biology,
    reader-aligned and prompted surprisal columns are built and each of the models
    in ``SURPRISAL_MODELS`` is fit on all readers × words of the whole corpus,
    plus the no-surprisal baseline, recording the log-likelihood, ΔLL over that
    baseline and the surprisal slope. Returns long-form columns ``index``,
    ``epoch``, ``spec``, ``model``, ``n``, ``ll``, ``delta_ll``, ``b_surprisal``,
    ``p_surprisal``.
    """
    indices = sorted(surp_versions["index"].unique())
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()

    rows = []
    for index in indices:
        d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
        ll0 = _fit_model(d, measure, None, spec=spec).llf  # no-surprisal reference
        for name in SURPRISAL_MODELS:
            col = f"S_{name}"
            res = _fit_model(d, measure, col, spec=spec)
            rows.append(
                {
                    "index": index,
                    "epoch": epoch_of[index],
                    "spec": spec,
                    "model": name,
                    "n": len(d),
                    "ll": res.llf,
                    "delta_ll": res.llf - ll0,
                    # main-effect surprisal slope; interaction terms (if any in the
                    # spec) carry the rest and are named e.g. "S_aligned:is_expert".
                    "b_surprisal": res.fe_params.get(col, np.nan),
                    "p_surprisal": res.pvalues.get(col, np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "index"])
