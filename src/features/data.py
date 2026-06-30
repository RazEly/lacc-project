"""Load PoTeC word features and reading measures into tidy DataFrames.

``reading_measures_merged`` files already carry word features + reader metadata,
so reading-time work needs one read per reader×text — no join to ``word_features``.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from src.config import (
    POTEC_READING_MEASURES_DIR,
    POTEC_WORD_FEATURES_DIR,
)

# pandas' default NA set minus "null" — PoTeC's German word "null" (p3) must stay
# a string, so keep_default_na=False + this explicit list.
_NA_VALUES = [
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NaN",
    "None",
    "n/a",
    "nan",
    "",
]

# Word-level identity + features kept from word_features / reading_measures.
WORD_COLS = [
    "word",
    "word_length",
    "word_index_in_text",
    "word_index_in_sent",
    "sent_index_in_text",
    "text_id",
    "text_domain",
    "is_sent_beginning",
    "is_sent_end",
    "is_expert_technical_term",
    "is_general_technical_term",
    "STTS_PoS_tag",  # Stuttgart-Tübingen PoS tag -> function/content category
]

# Eye-tracking + reader-metadata columns added on top of WORD_COLS for the
# reading-measures table.
READING_COLS = WORD_COLS + [
    "reader_id",
    "FFD",
    "SFD",
    "FPRT",
    "TFT",
    "TFC",
    "RPD_inc",
    "Fix",
    "lemma_frequency_normalized",
    "reader_discipline_numeric",
    "text_domain_numeric",
    "level_of_studies_numeric",
    "discipline_level_of_studies_numeric",
    "expert_reading_label_numeric",
    "mean_acc_tq",
    "mean_acc_bq",
]


def read_potec(path, **kw) -> pd.DataFrame:
    """Read a PoTeC TSV with the correct NA handling (keeps the word "null")."""
    return pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False,
        na_values=_NA_VALUES,
        dtype={"word": str},
        **kw,
    )


def _load_concat(directory, pattern, cols) -> pd.DataFrame:
    """Read + concat every ``pattern`` file under ``directory`` (selected ``cols``)."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    frames = [read_potec(f, usecols=cols) for f in files]
    return pd.concat(frames, ignore_index=True)[cols]


def load_word_features(cols=WORD_COLS) -> pd.DataFrame:
    """Concatenate every ``word_features_*.tsv`` into one words DataFrame.

    One row per word per text, keyed by (``text_id``, ``word_index_in_text``).
    """
    return _load_concat(POTEC_WORD_FEATURES_DIR, "word_features_*.tsv", cols)


def add_expertise(rm: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_expert`` = reader_discipline == text_domain (any study level).

    Škrjanec & Demberg 2026's definition. The dataset's
    ``expert_reading_label_numeric`` also requires graduate status (conflates
    expertise with seniority), so we recompute.
    """
    rm = rm.copy()
    rm["is_expert"] = (
        rm["reader_discipline_numeric"] == rm["text_domain_numeric"]
    ).astype(int)
    return rm


def load_reading_measures(cols=READING_COLS) -> pd.DataFrame:
    """Concatenate the ~900 merged reading-measure files into one DataFrame.

    One row per word per reader (+ word features, reader metadata, ``is_expert``).
    Keyed by (``reader_id``, ``text_id``, ``word_index_in_text``).
    """
    rm = _load_concat(POTEC_READING_MEASURES_DIR, "reader*_merged.tsv", cols)
    return add_expertise(rm)
