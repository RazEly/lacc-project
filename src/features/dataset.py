"""Assemble the reader×word modeling frame from raw PoTeC + surprisal tables.

One home for "raw PoTeC -> clean reader×word design frame": reading-time
cleaning (step 1), then the per-checkpoint merge of every surprisal source with
the reading measures and paper covariates (``build_index_df``). The statistical
fits that consume the frame live in ``analysis.model_comparison``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import WORD_KEY


def clean_reading_times(
    rm: pd.DataFrame,
    measure: str = "TFT",
    sd_k: float = 3.0,
) -> pd.DataFrame:
    """Return ``rm`` with text-edge words, skips, and per-reader outliers removed.

    ``sd_k`` = number of standard deviations around each reader's mean RT beyond
    which a data point is dropped (paper uses 3).
    """
    # drop text-edge words (no preceding / following context) and skips (RT 0)
    edge = rm.groupby("text_id")["word_index_in_text"].transform
    df = rm[
        rm["word_index_in_text"].ne(edge("min"))
        & rm["word_index_in_text"].ne(edge("max"))
        & rm[measure].gt(0)
    ]

    # per-reader fence: keep RTs within sd_k·SD of that reader's mean
    grp = df.groupby("reader_id")[measure]
    dev = (df[measure] - grp.transform("mean")).abs()
    sd = grp.transform("std")
    return df[sd.isna() | dev.le(sd_k * sd)].copy()


def _sum_code(flag: int):
    """0/1 flag -> −1/+1."""
    return 2 * flag - 1


def _prep_models(df, measure):
    """One row per reader×word with every surprisal column + covariates.

    ``df`` must carry ``s_baseline`` / ``s_physics`` / ``s_biology`` /
    ``s_prompt_physics`` / ``s_prompt_biology`` plus the reading measures. The two
    ``s_prompt_*`` columns are both the reader-aligned ``prompted`` arm and the two
    fixed-domain pseudo-tests (used directly as ``s_prompt_physics`` /
    ``s_prompt_biology`` in the model comparison).
    """
    raw_cols = [
        "s_baseline", "s_physics", "s_biology", "s_prompt_physics", "s_prompt_biology"
    ]

    # reader_discipline_numeric: 1 = physicist, 0 = biologist.
    def is_physicist(x):
        return x["reader_discipline_numeric"] == 1

    return (
        # skips (measure == 0) are owned by clean_reading_times; re-applied here
        # only as a guard for callers passing uncleaned frames.
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
            s_aligned=lambda x: x["s_physics"].where(is_physicist(x), x["s_biology"]),
            s_prompted=lambda x: x["s_prompt_physics"].where(
                is_physicist(x), x["s_prompt_biology"]
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


def build_index_df(surp_versions, rt_df, prompt_surp, index, measure):
    """Reader×word frame with all surprisal columns + covariates at one checkpoint.

    Physics/biology checkpoints are paired by ``index`` (0 = baseline), not
    ``training_steps`` (which the shared schedule keeps equal, but epoch would
    differ). Index 0 supplies ``s_baseline``.
    ``prompt_surp`` carries the checkpoint-independent prompted columns.
    """
    base_index = surp_versions["index"].min()

    def _surp(idx, domain, name):
        sel = surp_versions[
            (surp_versions["index"] == idx) & (surp_versions["domain"] == domain)
        ]
        return sel[WORD_KEY + ["surprisal"]].rename(columns={"surprisal": name})

    surp = (
        _surp(base_index, "physics", "s_baseline")
        .merge(_surp(index, "physics", "s_physics"), on=WORD_KEY)
        .merge(_surp(index, "biology", "s_biology"), on=WORD_KEY)
        .merge(prompt_surp, on=WORD_KEY)
    )
    merged = surp.merge(rt_df, on=WORD_KEY, how="inner")
    return _prep_models(merged, measure).reset_index(drop=True)
