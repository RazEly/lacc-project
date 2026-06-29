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
import statsmodels.formula.api as smf
from scipy.stats import chi2, spearmanr, wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.analysis import correlation as co
from src.analysis import model_comparison as mc
from src.analysis import stats as st
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


# ── robust domain-attention vs expert reading-time analysis ───────────────────
# The aligned-attention ΔLL comparison (main_attention) was a null: raw attention
# is surface-dominated (length/position), shared across the domain-FT models, so
# reader-matching can't win. These functions implement the more robust measures:
#   (1) residualize FT attention on baseline -> "specialized" attention (the part
#       fine-tuning actually added), mirroring Škrjanec et al.'s residualized
#       specialized surprisal;
#   (2) the paper-09 three-way test general + aligned-specialized attention ×
#       expertise × terminology, with an LRT for the specialized block;
#   (3) a model-free ΔA-vs-Δ(expert speed-up) difference correlation on technical
#       terms, where surface cancels on both sides.

ATTN_KEY = WORD_KEY


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of OLS ``y ~ 1 + x`` — the part of ``y`` not explained by ``x``."""
    X = np.column_stack([np.ones_like(x, dtype=float), np.asarray(x, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    return np.asarray(y, dtype=float) - X @ beta


def build_specialized(baseline_pw, physics_pw, biology_pw) -> pd.DataFrame:
    """Per-word general + residualized-specialized attention at the chosen layer.

    Inputs are ``text_id, word_index_in_text, attention`` (layer L) for the
    baseline, physics-FT and biology-FT models. Returns one row per word with
    ``a_base`` (general attention) and ``a_spec_phys`` / ``a_spec_bio`` (each FT
    attention residualized on the baseline — the domain-specific component, with the
    shared surface signal removed).
    """
    pw = (
        baseline_pw.rename(columns={"attention": "a_base"})
        .merge(physics_pw.rename(columns={"attention": "a_phys"}), on=ATTN_KEY)
        .merge(biology_pw.rename(columns={"attention": "a_bio"}), on=ATTN_KEY)
    )
    pw["a_spec_phys"] = residualize(pw["a_phys"].to_numpy(), pw["a_base"].to_numpy())
    pw["a_spec_bio"] = residualize(pw["a_bio"].to_numpy(), pw["a_base"].to_numpy())
    return pw[ATTN_KEY + ["a_base", "a_spec_phys", "a_spec_bio"]]


def specialized_versions(spec_pw: pd.DataFrame) -> pd.DataFrame:
    """Shape specialized attention into the ``surp_versions`` schema for reuse.

    index 0 (both domains) = general attention ``a_base`` (the ``baseline`` source);
    index 1 physics/biology = the residualized specialized attention (the
    ``physics``/``biology`` sources). ``model_comparison`` then forms ``aligned`` =
    reader-discipline mix of the two specialized columns. Attention lives in the
    ``surprisal`` column.
    """
    def tag(col, index, domain):
        d = spec_pw[ATTN_KEY + [col]].rename(columns={col: "surprisal"}).copy()
        d["index"] = index
        d["domain"] = domain
        d["epoch"] = float(index)
        return d

    return pd.concat(
        [
            tag("a_base", 0, "physics"),
            tag("a_base", 0, "biology"),
            tag("a_spec_phys", 1, "physics"),
            tag("a_spec_bio", 1, "biology"),
        ],
        ignore_index=True,
    )


def aligned_attention_vuong(av, rm, measure="TFT", index=1, spec="paper_full"):
    """Reader-clustered Vuong: aligned attention vs baseline/physics/biology.

    Proper significance for the source comparison (ΔLL has no SE). Restricted to the
    three non-prompted attention sources. BH-corrected. ``av`` is
    ``specialized_versions`` output; ``spec`` selects the fixed-effects formula.
    """
    d = mc.build_index_df(av, rm, None, index, measure)
    fits = {n: mc._fit_model(d, measure, f"S_{n}", spec=spec)
            for n in ("aligned", "baseline", "physics", "biology")}
    rows = []
    for other in ("baseline", "physics", "biology"):
        r = st.vuong_test(fits["aligned"], fits[other])
        rows.append({"comparison": f"aligned vs {other}", "spec": spec, **r,
                     "winner": "aligned" if r["z"] > 0 else other})
    out = pd.DataFrame(rows)
    _, out["p_adj"] = st.bh_correct(out["p"].to_numpy())
    return out


def _design_with_attention(spec_pw, rm, measure):
    """Reader×word frame: RT + covariates + general & aligned-specialized attention."""
    d = rm[rm[measure] > 0].merge(spec_pw, on=ATTN_KEY).copy()
    d = d.dropna(subset=[measure, "word_length", "lemma_frequency_normalized",
                         "a_base", "a_spec_phys", "a_spec_bio"])
    d = d[d["word_length"] > 0]
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])
    d["log_word_length"] = np.log(d["word_length"])
    d["word_position"] = (
        (d["word_index_in_sent"] - d["word_index_in_sent"].mean())
        / d["word_index_in_sent"].std()
    )
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    physicist = (d["reader_discipline_numeric"] == 1).to_numpy(dtype=float)
    d["A_general"] = d["a_base"]
    # reader-aligned specialized attention (discipline-matched FT residual)
    d["A_aligned"] = physicist * d["a_spec_phys"] + (1.0 - physicist) * d["a_spec_bio"]
    return d


def _three_way_term(params, var):
    """fe_params key for ``var:is_expert:is_technical`` (patsy term-order agnostic)."""
    for k in params.index:
        if var in k and "is_expert" in k and "is_technical" in k and k.count(":") == 2:
            return k
    return None


def specialized_interaction_test(spec_pw, rm, measure="TFT"):
    """Paper-09 three-way test: does aligned-specialized attention add over general?

    Fits ``RT ~ freq+length+position + is_expert*is_technical
    + A_general*is_expert*is_technical [+ A_aligned*is_expert*is_technical] + (1|reader)``
    and likelihood-ratio-tests the aligned-specialized block (4 terms). Reports the
    LRT p-value and the key ``A_aligned : is_expert : is_technical`` coefficient —
    the expert-on-technical-term effect of domain-matched attention. (Reader random
    intercept only; crossed by-word RE is the statsmodels-impractical ideal — the
    ΔA/ΔRT correlation below is the word-level robustness check.)
    """
    d = _design_with_attention(spec_pw, rm, measure)
    base_rhs = ("log_word_freq + log_word_length + word_position"
                " + is_expert * is_technical"
                " + A_general * is_expert * is_technical")
    full_rhs = base_rhs + " + A_aligned * is_expert * is_technical"
    reduced = smf.mixedlm(f"{measure} ~ {base_rhs}", d, groups=d["reader_id"]).fit(reml=False)
    full = smf.mixedlm(f"{measure} ~ {full_rhs}", d, groups=d["reader_id"]).fit(reml=False)
    lr = 2.0 * (full.llf - reduced.llf)
    df_diff = len(full.fe_params) - len(reduced.fe_params)
    key = _three_way_term(full.fe_params, "A_aligned")
    return {
        "n": len(d),
        "ll_reduced": reduced.llf,
        "ll_full": full.llf,
        "lr_stat": lr,
        "df": df_diff,
        "p_lrt": float(chi2.sf(lr, df_diff)) if df_diff > 0 else np.nan,
        "coef_3way": float(full.fe_params.get(key, np.nan)),
        "p_3way": float(full.pvalues.get(key, np.nan)),
        "term_3way": key,
    }


def attention_expertise_diff_correlation(baseline_pw, physics_pw, biology_pw, rm,
                                         measure="TFT"):
    """Model-free ΔA vs Δ(expert speed-up) correlation; surface cancels both sides.

    For each word: ΔA = (domain-FT attention − baseline attention) using the word's
    OWN text domain; Δ(expert speed-up) = mean RT of out-of-domain (novice) readers
    − mean RT of in-domain (expert) readers. Positive Spearman = where domain
    fine-tuning shifts attention, experts read faster. Reported on all / technical /
    common words.
    """
    pw = (
        baseline_pw.rename(columns={"attention": "a_base"})
        .merge(physics_pw.rename(columns={"attention": "a_phys"}), on=ATTN_KEY)
        .merge(biology_pw.rename(columns={"attention": "a_bio"}), on=ATTN_KEY)
    )
    # per-word expert/novice mean RT
    g = (rm[rm[measure] > 0].groupby(ATTN_KEY + ["is_expert"])[measure].mean()
         .unstack("is_expert"))
    g["dRT"] = g.get(0) - g.get(1)  # novice − expert (positive = experts faster)
    g = g.reset_index()[ATTN_KEY + ["dRT"]]
    meta = rm.drop_duplicates(ATTN_KEY)[
        ATTN_KEY + ["text_domain_numeric", "is_expert_technical_term",
                    "is_general_technical_term"]
    ]
    df = pw.merge(meta, on=ATTN_KEY).merge(g, on=ATTN_KEY)
    df["is_technical"] = (
        (df["is_expert_technical_term"] == 1) | (df["is_general_technical_term"] == 1)
    ).astype(int)
    phys = df["text_domain_numeric"] == 1
    df["dA"] = np.where(phys, df["a_phys"] - df["a_base"], df["a_bio"] - df["a_base"])

    rows = []
    subsets = [("all", df), ("technical", df[df["is_technical"] == 1]),
               ("common", df[df["is_technical"] == 0])]
    for label, sub in subsets:
        s = sub[["dA", "dRT"]].dropna()
        rho, p = (spearmanr(s["dA"], s["dRT"]) if len(s) > 3 else (np.nan, np.nan))
        rows.append({"subset": label, "n": len(s), "spearman": rho, "p": p})
    return pd.DataFrame(rows)
