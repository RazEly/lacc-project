"""Surprisal comparison: reader-aligned vs single model (step 5).

On the whole corpus, fit each surprisal source S in the same mixed model
``RT ~ log_word_freq + log_word_length + S + (1|reader)`` and compare ΔLL over
the no-surprisal baseline. S is one of:
  baseline          : un-adapted step-0 model (same for every reader)
  physics / biology : the domain fine-tuned model (every reader)
  aligned           : by READER discipline — physics surprisal for physicists,
                      biology for biologists (even on the other domain's texts)
  expertise_aligned : routed by demonstrated expertise (mean_acc_bq > 0.9)
  prompted          : baseline weights + discipline-matched system prompt
  prompted_level    : prompt matched to discipline AND study level

The question: does ``aligned`` beat the single-model sources? Significance in
``stats.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis.correlation import WORD_KEY

SURPRISAL_MODELS = (
    "baseline", "physics", "biology", "aligned", "prompted", "prompted_level"
)

# (level_of_studies_numeric, reader_discipline_numeric) → field×level prompt column.
_FIELD_LEVEL_COLS = {
    (0, 1): "s_prompt_phys_ug",
    (1, 1): "s_prompt_phys_grad",
    (0, 0): "s_prompt_bio_ug",
    (1, 0): "s_prompt_bio_grad",
}


def _prep_models(df, measure):
    """One row per reader×word with every surprisal column + covariates.

    ``df`` must carry ``s_base`` / ``s_phys`` / ``s_bio`` + reading measures. The
    prompted columns are optional; absent, the prompted models are not built.
    """
    prompt_cols = ["s_prompt_phys", "s_prompt_bio", *_FIELD_LEVEL_COLS.values()]
    has_prompt = all(c in df.columns for c in prompt_cols)
    d = df[df[measure] > 0].copy()
    d = d.dropna(
        subset=[measure, "word_length", "lemma_frequency_normalized",
                "s_base", "s_phys", "s_bio",
                *(prompt_cols if has_prompt else [])]
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
    # expertise_aligned: route per trial by demonstrated expertise (mean_acc_bq
    # > 0.9) on the TEXT's domain; non-experts keep baseline. Masks are disjoint.
    de = (d["mean_acc_bq"] > 0.9).to_numpy(dtype=float)
    phys_text = (d["text_domain_numeric"] == 1).to_numpy(dtype=float)
    de_phys = de * phys_text
    de_bio = de * (1.0 - phys_text)
    d["S_expertise_aligned"] = (
        de_phys * d["s_phys"] + de_bio * d["s_bio"]
        + (1.0 - de_phys - de_bio) * d["s_base"]
    )
    if has_prompt:
        d["S_prompted"] = (
            physicist * d["s_prompt_phys"] + (1.0 - physicist) * d["s_prompt_bio"]
        )
        # prompted_level: pick each reader's (level, discipline) prompt column.
        graduate = (d["level_of_studies_numeric"] == 1).to_numpy(dtype=float)
        pl_phys = (
            graduate * d["s_prompt_phys_grad"]
            + (1.0 - graduate) * d["s_prompt_phys_ug"]
        )
        pl_bio = (
            graduate * d["s_prompt_bio_grad"]
            + (1.0 - graduate) * d["s_prompt_bio_ug"]
        )
        d["S_prompted_level"] = physicist * pl_phys + (1.0 - physicist) * pl_bio
    # extra predictors for the richer specs (see MODEL_SPECS). Position is
    # standardized WITHIN sentence (Škrjanec et al.'s WordIndexInSentence).
    d["word_position"] = (
        (d["word_index_in_sent"] - d["word_index_in_sent"].mean())
        / d["word_index_in_sent"].std()
    )
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    d["is_domain_term"] = (d["is_expert_technical_term"] == 1).astype(int)
    return d


# Fixed-effects specs. Each = (terms WITHOUT surprisal, terms ADDED with it);
# ``{col}`` is the surprisal column. ΔLL is comparable across models within a
# spec, not across specs.
MODEL_SPECS = {
    # freq/length covariates + plain surprisal slope (the original).
    "covariates": ("log_word_freq + log_word_length", "{col}"),
    # + position + expertise×terminology; surprisal interacts with both.
    "full": (
        "log_word_freq + log_word_length + word_position"
        " + is_expert * is_technical + is_domain_term",
        "{col} * is_expert + {col}:is_technical",
    ),
    # Škrjanec, Broy & Demberg (2023) richest model (footnote 5): the three-way
    # surprisal × expertise × terminology interaction. For checkpoints 0-1.
    "paper_full": (
        "word_length + word_position + is_expert * is_technical",
        "{col} * is_expert * is_technical",
    ),
}


def _fit_model(d, measure, surprisal_col=None, spec="covariates", log_rt=False):
    """Mixed model ``[log] measure ~ <spec> [+ surprisal terms] + (1|reader)``.

    Without ``surprisal_col`` the no-surprisal reference is fit. ``log_rt`` logs
    the RT response; its LLs are only comparable within one ``log_rt`` setting.
    """
    base, surp_tmpl = MODEL_SPECS[spec]
    rhs = base
    if surprisal_col is not None:
        rhs += " + " + surp_tmpl.format(col=surprisal_col)
    lhs = f"np.log({measure})" if log_rt else measure  # np in scope for patsy
    md = smf.mixedlm(f"{lhs} ~ {rhs}", d, groups=d["reader_id"])
    return md.fit(reml=False)  # ML so LLs are comparable across the surprisal term


def build_index_df(surp_versions, rt_df, prompt_surp, index, measure):  # prompt_surp may be None
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
    return _prep_models(merged, measure)


def model_comparison_over_epochs(
    surp_versions: pd.DataFrame,
    rt_df: pd.DataFrame,
    prompt_surp: pd.DataFrame,
    measure="TFT",
    spec="covariates",
    models=SURPRISAL_MODELS,
    indices=None,
    log_rt=False,
) -> pd.DataFrame:
    """Per-checkpoint surprisal-model comparison on the whole corpus.

    ``surp_versions`` must hold both domains. ``prompt_surp=None`` skips the
    prompted models. ``models`` defaults to ``SURPRISAL_MODELS`` (``expertise_aligned``
    also available). ``indices`` restricts the checkpoint sweep. Each model is fit
    against the no-surprisal baseline. Long-form columns: ``index``, ``epoch``,
    ``spec``, ``model``, ``n``, ``ll``, ``delta_ll``, ``b_surprisal``, ``p_surprisal``.
    """
    all_indices = sorted(surp_versions["index"].unique())
    if indices is not None:
        keep = set(indices)
        all_indices = [i for i in all_indices if i in keep]
    epoch_of = surp_versions.groupby("index")["epoch"].first().to_dict()

    rows = []
    for index in all_indices:
        d = build_index_df(surp_versions, rt_df, prompt_surp, index, measure)
        ll0 = _fit_model(d, measure, None, spec=spec, log_rt=log_rt).llf  # reference
        for name in models:
            col = f"S_{name}"
            res = _fit_model(d, measure, col, spec=spec, log_rt=log_rt)
            rows.append(
                {
                    "index": index,
                    "epoch": epoch_of[index],
                    "spec": spec,
                    "model": name,
                    "n": len(d),
                    "ll": res.llf,
                    "delta_ll": res.llf - ll0,
                    # main-effect slope; spec interaction terms carry the rest.
                    "b_surprisal": res.fe_params.get(col, np.nan),
                    "p_surprisal": res.pvalues.get(col, np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "index"])
