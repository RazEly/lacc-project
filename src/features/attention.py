"""Attention representations per word per layer (step 3).

* ``raw``  — head-averaged self-attention; each word scored by the attention it
  *receives* (averaged over the query tokens that can attend to it under the
  causal mask), sub-token -> word by **max** (Sood et al. 2020), then
  sum-normalized within the sentence.

Also exposes the eye-tracking feature table and its per-domain PCA reduction
used for the attention-vs-gaze comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.config import ET_MEASURE_MAP, PCA_MEASURES

WORD_KEY = ["text_id", "word_index_in_text"]

# STTS (Stuttgart-Tübingen-Tagset) tags counted as FUNCTION words: determiners,
# adpositions, conjunctions, pronouns, particles, and auxiliary/modal verbs.
# Everything else (NN/NE nouns, ADJ*, ADV, full verbs VV*, FM, CARD, ITJ, XY) is
# treated as a CONTENT word. Used for the function/content predictor in the
# attention-vs-gaze regression (Mouratidi & Poesio's "functional category").
_STTS_FUNCTION_TAGS = frozenset({
    "ART", "APPR", "APPRART", "APPO", "APZR",
    "KON", "KOUS", "KOUI", "KOKOM",
    "PPER", "PPOSAT", "PPOSS", "PRELS", "PRELAT", "PDS", "PDAT",
    "PIS", "PIAT", "PWS", "PWAT", "PWAV", "PROAV", "PAV",
    "PTKZU", "PTKNEG", "PTKVZ", "PTKANT", "PTKA",
    "VAFIN", "VAIMP", "VAINF", "VAPP", "VMFIN", "VMINF", "VMPP",
})


def function_word_flag(pos: pd.Series) -> pd.Series:
    """1 if the STTS PoS tag is a function word, else 0 (content word).

    NaN/unknown tags fall through to 0 (content), the conservative default.
    """
    return pos.isin(_STTS_FUNCTION_TAGS).astype(int)


# ── helpers ──────────────────────────────────────────────────────────────────
def _word_positions(word_ids, n_words):
    """List of token positions belonging to each word (skips word_id None)."""
    pos = [[] for _ in range(n_words)]
    for tok_i, wid in enumerate(word_ids):
        if wid is not None:
            pos[wid].append(tok_i)
    return pos


def _normalize(scores):
    """Sum-normalize a per-word vector (relative attention within sentence)."""
    s = scores.sum()
    return scores / s if s > 0 else scores


@torch.no_grad()
def _forward_attentions(word_list, model, tokenizer, prefix=None):
    """Return (attentions [L, seq, seq] head-averaged, word_ids).

    ``prefix`` optionally prepends an instruction string before the sentence
    (Mouratidi & Poesio prepend task instructions). The prefix tokens carry
    ``word_id=None`` so they never receive a word score, but they remain in the
    sequence as query/key positions, so each sentence word's received attention
    is averaged over all other tokens *including* the instruction words.
    """
    enc = tokenizer(word_list, is_split_into_words=True, return_tensors="pt")
    input_ids = enc.input_ids
    word_ids = list(enc.word_ids(0))
    if prefix:
        pre = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = torch.cat([pre, input_ids], dim=1)
        word_ids = [None] * pre.shape[1] + word_ids
    device = next(model.parameters()).device
    out = model(input_ids.to(device), output_attentions=True)
    # each layer: [1, heads, seq, seq] -> head-average -> [seq, seq]
    atts = torch.stack([a[0].mean(0) for a in out.attentions]).cpu().numpy()
    return atts, word_ids


# ── raw attention ────────────────────────────────────────────────────────────
def raw_attention(word_list, model, tokenizer, prefix=None) -> np.ndarray:
    """Raw attention received per word per layer. -> [n_words, n_layers]."""
    atts, word_ids = _forward_attentions(word_list, model, tokenizer, prefix=prefix)
    n_layers, seq, _ = atts.shape
    n_words = len(word_list)
    positions = _word_positions(word_ids, n_words)

    out = np.zeros((n_words, n_layers))
    for l in range(n_layers):
        A = atts[l]
        # attention each key token receives, averaged over queries that attend
        # to it (causal: query q >= key k). A[k:, k].mean() avoids zero-dilution.
        recv = np.array([A[k:, k].mean() if seq > k else 0.0 for k in range(seq)])
        word_score = np.array([recv[p].max() if p else 0.0 for p in positions])
        out[:, l] = _normalize(word_score)
    return out


# ── per-sentence extraction over the corpus ──────────────────────────────────
def extract_attention(
    words_df: pd.DataFrame,
    model,
    tokenizer,
    method: str = "raw",
    prefix=None,
) -> pd.DataFrame:
    """Run raw attention over every sentence; long-form output.

    Returns columns ``text_id``, ``word_index_in_text``, ``layer``,
    ``attention``, ``attention_method``. First/last words of each sentence are
    dropped to match the surprisal / reading-time cleaning. ``prefix`` optionally
    prepends an instruction string to every sentence (see ``raw_attention``).
    """
    sents = [
        sent
        for _, sent in words_df.sort_values(
            ["text_id", "sent_index_in_text", "word_index_in_sent"]
        ).groupby(["text_id", "sent_index_in_text"], sort=False)
        if len(sent) >= 3
    ]

    scores_list = [
        raw_attention(
            s["word"].fillna("").astype(str).tolist(), model, tokenizer, prefix=prefix
        )
        for s in sents
    ]

    rows = []
    for sent, scores in zip(sents, scores_list):
        idx = sent["word_index_in_text"].tolist()
        tid = sent["text_id"].iloc[0]
        beg = sent["is_sent_beginning"].tolist()
        end = sent["is_sent_end"].tolist()
        for k in range(scores.shape[0]):
            if beg[k] == 1 or end[k] == 1:
                continue
            for l in range(scores.shape[1]):
                rows.append((tid, idx[k], l, scores[k, l], method))
    return pd.DataFrame(
        rows, columns=WORD_KEY + ["layer", "attention", "attention_method"]
    )


# ── eye-tracking features + PCA ──────────────────────────────────────────────
def eyetracking_features(rm: pd.DataFrame, relative: bool = True) -> pd.DataFrame:
    """Per-word eye-tracking measures, normalized to their within-sentence share.

    Each measure is averaged across participants (skips of measure == 0 excluded),
    then — when ``relative`` (the default and the ONLY paper-correct setting) —
    divided by its sentence total so every word carries its *relative value within
    the sentence*. This is Mouratidi & Poesio (2025) §3.1 ("normalized to their
    relative value in each sentence"), and it MUST match the sum-normalization of
    the attention scores (``_normalize``).

    DO NOT remove this normalization or replace it with a within-sentence z-score:
    the attention-vs-gaze Spearman correlation collapses from ~+0.7 to ~0/negative
    when the gaze side is left raw or is mean-centred per sentence, because the
    sentence-relative magnitude is exactly the signal attention tracks. See
    ``pca_eyetracking`` and the diagnostics in ``scripts/diag_attention_drivers.py``.

    Returns one row per word with the six PoTeC-mapped measures (as shares) plus
    ``text_domain`` and ``sent_index_in_text``.
    """
    measures = list(dict.fromkeys(ET_MEASURE_MAP.values()))
    meta = rm[WORD_KEY + ["text_domain", "sent_index_in_text"]].drop_duplicates(
        WORD_KEY
    )
    out = meta
    for m in measures:
        col = rm[rm[m] > 0].groupby(WORD_KEY)[m].mean().rename(m).reset_index()
        out = out.merge(col, on=WORD_KEY, how="left")
    if relative:
        sums = out.groupby(["text_id", "sent_index_in_text"])[measures].transform("sum")
        out[measures] = out[measures].div(sums)
    return out


def pca_eyetracking(et_df: pd.DataFrame, domain: str):
    """Fit a 1-component PCA on the 4 informative measures for one domain.

    ``et_df`` already carries the measures as within-sentence *relative shares*
    (``eyetracking_features``), so PCA runs on those shares — only globally
    standardized for numerical scale, NEVER re-centred within sentence (that would
    undo the relative-value normalization and collapse the gaze signal; see
    ``eyetracking_features``). The component is sign-oriented so higher = more
    reading (positive against the TFT share). Returns ``(scored_df,
    explained_variance_ratio)`` with ``text_id``, ``word_index_in_text``, ``pca``.
    """
    sub = et_df[et_df["text_domain"] == domain].copy()
    X = sub[PCA_MEASURES]
    mask = X.notna().all(axis=1)
    X = X[mask]
    Xs = (X - X.mean()) / X.std(ddof=0)  # global standardization only
    pca = PCA(n_components=1)
    comp = pca.fit_transform(Xs.values).ravel()
    tft = ET_MEASURE_MAP["TRT"]  # "TFT" share — orient the component positively
    if np.corrcoef(comp, X[tft].values)[0, 1] < 0:
        comp = -comp
    scored = sub.loc[mask, WORD_KEY].copy()
    scored["pca"] = comp
    return scored, float(pca.explained_variance_ratio_[0])
