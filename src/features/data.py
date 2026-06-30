"""Load PoTeC word features and reading measures into tidy DataFrames.

The ``reading_measures_merged`` files already carry the full word-feature block
*and* the reader metadata, so for reading-time work a single read per
reader×text file is enough — no separate join to ``word_features`` is needed.

Run-only helpers; import from notebooks or other ``src`` modules.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from src.config import (
    POTEC_READING_MEASURES_DIR,
    POTEC_WORD_FEATURES_DIR,
)

# pandas turns these into NaN by default. PoTeC contains the German word "null"
# (text p3) which must stay a string, so we disable the defaults and supply the
# list explicitly (minus "null").
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
    """Add ``is_expert`` = reader's major matches the text domain.

    PoTeC reader-experience labelling (Škrjanec & Demberg 2026): a reader is an
    expert on a word iff ``reader_discipline == text_domain`` (physics student on
    a physics text, biology student on a biology text), independent of study
    level. The dataset's ``expert_reading_label_numeric`` additionally requires
    graduate status, which conflates expertise with seniority — so we recompute.
    """
    rm = rm.copy()
    rm["is_expert"] = (
        rm["reader_discipline_numeric"] == rm["text_domain_numeric"]
    ).astype(int)
    return rm


def load_reading_measures(cols=READING_COLS) -> pd.DataFrame:
    """Concatenate every merged reading-measure file (~900) into one DataFrame.

    Each row is one word for one reader; carries word features and reader
    metadata (incl. the recomputed ``is_expert`` flag). Keyed by
    (``reader_id``, ``text_id``, ``word_index_in_text``).
    """
    rm = _load_concat(POTEC_READING_MEASURES_DIR, "reader*_merged.tsv", cols)
    return add_expertise(rm)
