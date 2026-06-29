"""Predictive regression of eye-gaze from text features ± raw attention.

The predictive half of Mouratidi & Poesio's methodology (their RQ2/RQ3), applied
here to GPT-2 and PoTeC. For one reader group, a word-level OLS predicts each gaze
target from the text-only baseline features and, separately, from the same
features **plus** the model's raw attention (at its peak-correlating layer). Models
are compared by adjusted R² on a 20 % holdout and by a Wilcoxon signed-rank test on
the paired holdout squared errors (the paper's MSE test); predictor importance is
the absolute t-value.

Gaze targets are the six PoTeC-mapped eye-tracking measures plus the per-domain PCA
component (``correlation.build_et_table``). PoTeC has no word-level pupil size, so
six features stand in for the paper's seven.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.analysis import correlation as co
from src.config import ET_MEASURE_MAP
from src.features.attention import function_word_flag

WORD_KEY = ["text_id", "word_index_in_text"]
TEXT_FEATURES = ["log_word_freq", "log_word_length", "is_function_word", "surprisal"]
# Six PoTeC gaze measures (deduped) + the PCA component — the regression targets.
TARGETS = list(dict.fromkeys(ET_MEASURE_MAP.values())) + ["pca"]


def word_text_features(rm: pd.DataFrame, surprisal_df: pd.DataFrame) -> pd.DataFrame:
    """One row per word: the four text-only predictors.

    ``log_word_freq`` = log1p(lemma freq), ``log_word_length`` = log(word length),
    ``is_function_word`` from the STTS PoS tag, and per-word ``surprisal`` (bits)
    from the model under analysis (``surprisal_df`` = ``compute_surprisal`` output).
    """
    base = rm.drop_duplicates(WORD_KEY)[
        WORD_KEY + ["word_length", "lemma_frequency_normalized", "STTS_PoS_tag"]
    ].copy()
    base["log_word_freq"] = np.log1p(base["lemma_frequency_normalized"])
    base["log_word_length"] = np.log(base["word_length"].clip(lower=1))
    base["is_function_word"] = function_word_flag(base["STTS_PoS_tag"])
    out = base.merge(surprisal_df, on=WORD_KEY, how="inner")
    return out[WORD_KEY + TEXT_FEATURES]


def _adj_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² for ``n`` holdout points and ``p`` predictors."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan


def fit_holdout(X: pd.DataFrame, y: pd.Series, seed: int = 0, test_size: float = 0.2):
    """OLS on a train split, scored on the holdout. Same ``seed`` ⇒ same split.

    Returns adjusted R², R², MSE, the per-sample holdout squared errors (aligned
    across calls with the same seed/length), and the fitted t-values.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    res = sm.OLS(y_tr.values, sm.add_constant(X_tr.values)).fit()
    pred = res.predict(sm.add_constant(X_te.values, has_constant="add"))
    r2 = r2_score(y_te, pred)
    sq = (y_te.values - pred) ** 2
    tvals = dict(zip(["const"] + list(X.columns), res.tvalues))
    return {
        "adj_r2": _adj_r2(r2, len(y_te), X.shape[1]),
        "r2": r2,
        "mse": float(np.mean(sq)),
        "sq_errors": sq,
        "tvalues": tvals,
    }


def attention_gain(
    design: pd.DataFrame, target: str, features=TEXT_FEATURES, attn_col="attention", seed=0
):
    """Text-only vs text+attention holdout fit for one gaze ``target``.

    ``features`` selects the text-only predictor set — pass a subset (e.g. without
    ``surprisal``) to test how much of attention's contribution surprisal absorbs.
    Both models share the holdout split (same seed/length), so their squared errors
    are paired for the Wilcoxon signed-rank test. Returns a summary row plus the
    text+attention model's t-values for the feature-importance table.
    """
    features = list(features)
    d = design.dropna(subset=[target] + features + [attn_col])
    y = d[target]
    base = fit_holdout(d[features], y, seed)
    att = fit_holdout(d[features + [attn_col]], y, seed)
    try:
        _, w_p = wilcoxon(base["sq_errors"], att["sq_errors"])
    except ValueError:  # all paired differences zero, or empty
        w_p = np.nan
    row = {
        "target": target,
        "n": len(d),
        "adj_r2_text": base["adj_r2"],
        "adj_r2_attn": att["adj_r2"],
        "mse_text": base["mse"],
        "mse_attn": att["mse"],
        "delta_adj_r2": att["adj_r2"] - base["adj_r2"],
        "wilcoxon_p": w_p,
    }
    return row, att["tvalues"]


def run_model_prediction(
    model_name: str,
    surprisal_df: pd.DataFrame,
    attn_by_group: dict[str, pd.DataFrame],
    rm: pd.DataFrame,
    groups=("experts", "novices"),
    feature_sets=None,
    seed: int = 0,
):
    """Predictive comparison for one model across reader groups and gaze targets.

    ``attn_by_group`` maps a reader group to its per-word attention table
    (``text_id``, ``word_index_in_text``, ``attention``) at that group's
    peak-correlating layer. ``feature_sets`` maps a label to the text-only predictor
    list (default: ``with_surprisal`` = all four, ``no_surprisal`` = drop surprisal),
    so attention's contribution can be compared with and without surprisal. Returns
    ``(summary_df, importance_df)``, both tagged with ``feature_set``.
    """
    if feature_sets is None:
        feature_sets = {
            "with_surprisal": TEXT_FEATURES,
            "no_surprisal": [f for f in TEXT_FEATURES if f != "surprisal"],
        }
    txt = word_text_features(rm, surprisal_df)
    rows, imp = [], []
    for group in groups:
        et = co.build_et_table(rm, participants=group)
        attn_pw = attn_by_group[group]
        design = et.merge(txt, on=WORD_KEY).merge(attn_pw, on=WORD_KEY)
        for fs_name, feats in feature_sets.items():
            for target in TARGETS:
                if target not in design.columns:
                    continue
                row, tvals = attention_gain(design, target, features=feats, seed=seed)
                row.update({"model": model_name, "group": group, "feature_set": fs_name})
                rows.append(row)
                for predictor, t in tvals.items():
                    if predictor == "const":
                        continue
                    imp.append({
                        "model": model_name, "group": group, "feature_set": fs_name,
                        "target": target, "predictor": predictor, "abs_t": abs(t),
                    })
    return pd.DataFrame(rows), pd.DataFrame(imp)
