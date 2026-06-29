"""Knowledge-titrated personal surprisal (follow-up experiment).

Implements ``potec-knowledge-titrated-surprisal-plan.md``: make the *degree* of
LM domain adaptation a continuous function of each reader's measured
background-knowledge score, rather than a single group ("reader-aligned") model.

The pipeline reuses the existing DAPT checkpoint ladder (4ⁿ steps, see
``finetune.finetune_dapt`` / ``src.main``) and the surprisal recomputed at each
checkpoint (``finetune.recompute_surprisal_over_checkpoints``). On top of that it:

1. builds **domain-matched** surprisal at every checkpoint ``k`` — physics-LM
   surprisal on physics texts, biology-LM surprisal on biology texts — for every
   reader×word (``_surp_wide`` + ``per_reader_delta_ll``);
2. fits a **per-reader** simple LM (paper Appendix C/D: LM ≈ LMER here) and reads
   off ``ΔLL_p(M, k) = LL_surp − LL_base`` at each ``k`` (``per_reader_delta_ll``);
3. takes ``k*_p = argmax_k ΔLL_p`` (``find_kstar``) and tests the central
   hypothesis — within each discipline group, Spearman ``ρ(k*_p, bg_p)`` with a
   reader-bootstrap CI (``within_group_rho``);
4. builds a leakage-controlled, knowledge-**titrated** per-reader surprisal via a
   monotone ``bg → k`` map fit under participant-level nested CV
   (``titrated_oof_index``) and pits it against the binary reader-aligned model
   (paper's best) and the single population-optimal ``k`` with a reader-clustered
   Vuong test (``head_to_head``).

``bg_p`` is the PoTeC background-knowledge score (``mean_acc_bq``, per reader×text,
averaged to one value per reader). Response is log-RT, matching the paper's Eq.
Reading measures: FPRT (first-pass, the placebo channel — expect a null ρ), GP
(go-past, ``RPD_inc``) and TFT (total fixation time); the effect is predicted on
the late measures only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from src.analysis import model_comparison as mc
from src.analysis import stats as st
from src.analysis.correlation import WORD_KEY

# Reading measures swept: plan label -> PoTeC reading-measure column. GP (go-past
# / regression-path duration) is PoTeC's ``RPD_inc``. FPRT is the built-in placebo
# (first-pass should NOT track adaptation amount).
MEASURES = {"FPRT": "FPRT", "GP": "RPD_inc", "TFT": "TFT"}

# discipline numeric (reader_discipline_numeric / text_domain_numeric) -> name.
_DISCIPLINE = {1: "physics", 0: "biology"}

# Minimum per-reader words for a stable per-reader ΔLL fit.
_MIN_READER_WORDS = 30


# ── building blocks ──────────────────────────────────────────────────────────
def reader_table(rm: pd.DataFrame) -> pd.DataFrame:
    """One row per reader: discipline group + background-knowledge score ``bg``.

    ``bg`` aggregates the per-text background-question accuracy ``mean_acc_bq``
    (``bg_p = mean_t bg(p,t)``); ``group`` is the reader's discipline name. Using
    the per-text score (not the discipline split) is what keeps the calibrator
    independent of the discipline confound (plan §6.2).
    """
    g = rm.groupby("reader_id")
    out = pd.DataFrame(
        {
            "bg": g["mean_acc_bq"].mean(),
            "comp": g["mean_acc_tq"].mean(),
            "discipline_numeric": g["reader_discipline_numeric"].first(),
        }
    ).reset_index()
    out["group"] = out["discipline_numeric"].map(_DISCIPLINE)
    return out


def _surp_wide(surp_versions: pd.DataFrame, index: int) -> pd.DataFrame:
    """Per-word physics + biology surprisal at one checkpoint ``index``."""
    sel = surp_versions[surp_versions["index"] == index]
    phys = sel[sel["domain"] == "physics"][WORD_KEY + ["surprisal"]].rename(
        columns={"surprisal": "s_phys"}
    )
    bio = sel[sel["domain"] == "biology"][WORD_KEY + ["surprisal"]].rename(
        columns={"surprisal": "s_bio"}
    )
    return phys.merge(bio, on=WORD_KEY)


def _add_covariates(d: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Filter to scorable words and add the baseline covariates + log-RT response.

    Mirrors ``model_comparison._prep_models``: drop skipped/zero rows, log1p word
    frequency, log word length, standardized in-sentence position, technical-term
    flag (``is_expert`` is already on ``rm``). The response is ``log_rt`` — log of
    the reading measure, the Škrjanec et al. response.
    """
    d = d[d[measure] > 0].copy()
    d = d.dropna(subset=[measure, "word_length", "lemma_frequency_normalized"])
    d = d[d["word_length"] > 0]
    d["log_word_freq"] = np.log1p(d["lemma_frequency_normalized"])
    d["log_word_length"] = np.log(d["word_length"])
    d["word_position"] = (
        d["word_index_in_sent"] - d["word_index_in_sent"].mean()
    ) / d["word_index_in_sent"].std()
    d["is_technical"] = (
        (d["is_expert_technical_term"] == 1) | (d["is_general_technical_term"] == 1)
    ).astype(int)
    d["log_rt"] = np.log(d[measure])
    return d


def _domain_matched(d: pd.DataFrame, phys_col: str, bio_col: str) -> np.ndarray:
    """Pick the text-domain-matched surprisal: physics LM on physics texts, else bio."""
    return np.where(d["text_domain_numeric"] == 1, d[phys_col], d[bio_col])


def _reader_delta_ll(g: pd.DataFrame) -> float:
    """``LL_surp − LL_base`` for one reader via per-reader OLS on log-RT.

    Baseline = Length + LogFreq + Position (+ Expertise×Terminology when both vary
    for this reader); the surprisal model adds the domain-matched surprisal slope
    ``s_dom``. Per-reader simple LMs (plan §6.3, paper Appendix C/D: LM ≈ LMER).
    """
    if len(g) < _MIN_READER_WORDS:
        return np.nan
    base = "log_word_freq + log_word_length + word_position"
    if g["is_expert"].nunique() > 1 and g["is_technical"].nunique() > 1:
        base += " + is_expert * is_technical"
    elif g["is_technical"].nunique() > 1:
        base += " + is_technical"
    try:
        m0 = smf.ols(f"log_rt ~ {base}", g).fit()
        m1 = smf.ols(f"log_rt ~ {base} + s_dom", g).fit()
        return float(m1.llf - m0.llf)
    except Exception:  # noqa: BLE001 — singular design for a sparse reader -> drop
        return np.nan


def per_reader_delta_ll(
    surp_versions: pd.DataFrame,
    rm: pd.DataFrame,
    reader_tbl: pd.DataFrame,
    measure: str,
    idx_to_step: dict[int, int],
) -> pd.DataFrame:
    """Per-reader ΔLL at every checkpoint for one reading ``measure``.

    For each checkpoint ``index`` the domain-matched surprisal (physics LM on
    physics texts, biology LM on biology texts) is joined onto the reader×word
    table and each reader's ΔLL is read off (``_reader_delta_ll``). Returns long
    columns ``reader_id``, ``index``, ``step``, ``delta_ll`` + reader ``group`` /
    ``bg`` (merged from ``reader_tbl``).
    """
    rows = []
    for index in sorted(surp_versions["index"].unique()):
        wide = _surp_wide(surp_versions, index)
        d = wide.merge(rm, on=WORD_KEY, how="inner")
        d["s_dom"] = _domain_matched(d, "s_phys", "s_bio")
        d = _add_covariates(d, measure).dropna(subset=["s_dom"])
        for rid, g in d.groupby("reader_id"):
            rows.append(
                {
                    "reader_id": rid,
                    "index": index,
                    "step": idx_to_step.get(index, index),
                    "delta_ll": _reader_delta_ll(g),
                }
            )
    return pd.DataFrame(rows).merge(reader_tbl, on="reader_id", how="left")


def find_kstar(dll_df: pd.DataFrame) -> pd.DataFrame:
    """Each reader's ``k*`` — the checkpoint with the largest ΔLL (plan §6.3)."""
    g = dll_df.dropna(subset=["delta_ll"])
    return g.loc[g.groupby("reader_id")["delta_ll"].idxmax()].reset_index(drop=True)


# ── test 1: monotonicity ρ(k*, bg) within group ──────────────────────────────
def within_group_rho(
    kstar: pd.DataFrame, value_col: str = "step", n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Within-discipline Spearman ``ρ(k*_p, bg_p)`` with a reader-bootstrap 95% CI.

    Correlating **within** each discipline group controls the discipline confound
    (plan §8): a positive ρ means higher-knowledge readers peak at a more
    domain-adapted checkpoint. ``value_col`` is the adaptation amount (``step`` or
    ``index`` — Spearman is rank-based, so they agree). The CI resamples readers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for grp, g in kstar.groupby("group"):
        x = g[value_col].to_numpy(dtype=float)
        y = g["bg"].to_numpy(dtype=float)
        rho, p = spearmanr(x, y)
        boot = []
        if len(g) >= 3:
            for _ in range(n_boot):
                s = rng.integers(0, len(g), len(g))
                if np.unique(x[s]).size > 1 and np.unique(y[s]).size > 1:
                    boot.append(spearmanr(x[s], y[s])[0])
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        rows.append(
            {
                "group": grp,
                "n_readers": len(g),
                "rho": rho,
                "p": p,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


# ── test 2: titrated vs binary vs population-optimal ──────────────────────────
def population_optimal_index(dll_df: pd.DataFrame) -> int:
    """The single checkpoint maximizing summed per-reader ΔLL (the population ``k``)."""
    s = dll_df.dropna(subset=["delta_ll"]).groupby("index")["delta_ll"].sum()
    return int(s.idxmax())


def aligned_optimal_index(
    surp_versions: pd.DataFrame, rm: pd.DataFrame, measure: str
) -> int:
    """Checkpoint where the binary reader-aligned model (paper's best) peaks in ΔLL."""
    cmp = mc.model_comparison_over_epochs(
        surp_versions, rm, None, measure=measure, models=("aligned",), log_rt=True
    )
    return int(cmp.loc[cmp["delta_ll"].idxmax(), "index"])


def _fit_map(bg_train, k_train, available, kind: str):
    """Fit a monotone ``bg → checkpoint-index`` map on training readers.

    ``tercile``: median ``k*`` per training-bg tercile (the simple pre-registered
    map). ``isotonic``: monotone non-decreasing regression, snapped to the nearest
    available checkpoint index. Returns a function ``bg -> index``.
    """
    available = np.asarray(sorted(available))
    if kind == "tercile":
        edges = np.quantile(bg_train, [1 / 3, 2 / 3])
        meds = {}
        terc = np.digitize(bg_train, edges)
        for t in (0, 1, 2):
            sel = k_train[terc == t]
            meds[t] = int(np.median(sel)) if len(sel) else int(np.median(k_train))

        def f(bg):
            return meds[int(np.digitize([bg], edges)[0])]

        return f
    # isotonic
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(bg_train, k_train)

    def f(bg):
        pred = iso.predict([bg])[0]
        return int(available[np.argmin(np.abs(available - pred))])

    return f


def titrated_oof_index(
    kstar: pd.DataFrame, available_indices, kind: str = "tercile",
    n_folds: int = 5, seed: int = 0,
) -> dict[int, int]:
    """Out-of-fold knowledge-titrated checkpoint index per reader (nested CV).

    The ``bg → k`` map is fit on training readers and applied to held-out readers,
    so a reader's titrated checkpoint never depends on its own ``k*`` (plan §7,
    leakage control). Returns ``{reader_id: index}``. ``kind`` selects the map
    family (``tercile`` or ``isotonic``).
    """
    ks = kstar.dropna(subset=["index"]).reset_index(drop=True)
    bg = ks["bg"].to_numpy(dtype=float)
    k = ks["index"].to_numpy(dtype=float)
    rid = ks["reader_id"].to_numpy()
    out: dict[int, int] = {}
    n_splits = min(n_folds, len(ks))
    if n_splits < 2:
        return {int(r): int(kk) for r, kk in zip(rid, k)}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(bg):
        f = _fit_map(bg[tr], k[tr], available_indices, kind)
        for i in te:
            out[int(rid[i])] = int(f(bg[i]))
    return out


def build_headtohead(
    surp_versions: pd.DataFrame,
    rm: pd.DataFrame,
    oof_index: dict[int, int],
    pop_index: int,
    aligned_index: int,
    measure: str,
) -> pd.DataFrame:
    """Reader×word frame carrying the three competing surprisal columns.

    ``S_titrated`` = domain-matched surprisal at each reader's own OOF-titrated
    checkpoint; ``S_pop`` = domain-matched surprisal at the single population
    checkpoint; ``S_binary`` = the paper's reader-aligned model (discipline
    routing: physicists get physics-LM surprisal, biologists biology-LM) at its
    own optimal checkpoint. All on the same rows, with covariates + log-RT.
    """
    d = _add_covariates(rm.copy(), measure)

    pop = _surp_wide(surp_versions, pop_index).rename(
        columns={"s_phys": "pop_phys", "s_bio": "pop_bio"}
    )
    d = d.merge(pop, on=WORD_KEY, how="inner")
    d["S_pop"] = _domain_matched(d, "pop_phys", "pop_bio")

    ali = _surp_wide(surp_versions, aligned_index).rename(
        columns={"s_phys": "ali_phys", "s_bio": "ali_bio"}
    )
    d = d.merge(ali, on=WORD_KEY, how="inner")
    physicist = (d["reader_discipline_numeric"] == 1).to_numpy(dtype=float)
    d["S_binary"] = physicist * d["ali_phys"] + (1.0 - physicist) * d["ali_bio"]

    d["_oof"] = d["reader_id"].map(oof_index)
    parts = []
    for idx, gd in d.groupby("_oof"):
        w = _surp_wide(surp_versions, int(idx)).rename(
            columns={"s_phys": "tit_phys", "s_bio": "tit_bio"}
        )
        gg = gd.merge(w, on=WORD_KEY, how="left")
        gg["S_titrated"] = _domain_matched(gg, "tit_phys", "tit_bio")
        parts.append(gg)
    d = pd.concat(parts, ignore_index=True)
    return d.dropna(subset=["S_pop", "S_binary", "S_titrated"])


def head_to_head(d: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Reader-clustered Vuong tests among titrated / binary / population surprisal.

    Fits ``log(measure) ~ freq + length + S + (1|reader)`` for each surprisal
    source and runs the reader-clustered Vuong test (``stats.vuong_test``) on each
    pair. A positive ``z`` means the first model wins. Returns one row per pair.
    """
    fits = {
        name: mc._fit_model(d, measure, f"S_{name}", spec="covariates", log_rt=True)
        for name in ("titrated", "binary", "pop")
    }
    rows = []
    for a, b in (("titrated", "binary"), ("titrated", "pop"), ("binary", "pop")):
        r = st.vuong_test(fits[a], fits[b])
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "n_readers": r["n_readers"],
                "mean_dll": r["mean_dll"],
                "z": r["z"],
                "p": r["p"],
                "winner": a if r["z"] > 0 else b,
            }
        )
    out = pd.DataFrame(rows)
    _, out["p_adj"] = st.bh_correct(out["p"].to_numpy())
    return out
