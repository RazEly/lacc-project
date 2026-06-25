"""Attention representations per word per layer (step 3).

Two representations, both reduced to one score per word per layer:

* ``raw``  — head-averaged self-attention; each word scored by the attention it
  *receives* (averaged over the query tokens that can attend to it under the
  causal mask), sub-token -> word by **max** (Sood et al. 2020), then
  sum-normalized within the sentence.
* ``flow`` — attention treated as a flow network (Abnar & Zeng 2020). For each
  layer L, max-flow (Edmonds-Karp) is computed from every input token to the
  sentence-final token at layer L. A position-decay correction (Metzger et al.
  2022) offsets the early-token bias before sub-token -> word combination and
  sum-normalization.

Also exposes the eye-tracking feature table and its per-domain PCA reduction
used for the attention-vs-gaze comparison.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.config import ET_MEASURE_MAP, PCA_MEASURES

WORD_KEY = ["text_id", "word_index_in_text"]


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
def _forward_attentions(word_list, model, tokenizer):
    """Return (attentions [L, seq, seq] head-averaged, word_ids)."""
    enc = tokenizer(word_list, is_split_into_words=True, return_tensors="pt")
    out = model(enc.input_ids, output_attentions=True)
    # each layer: [1, heads, seq, seq] -> head-average -> [seq, seq]
    atts = torch.stack([a[0].mean(0) for a in out.attentions]).cpu().numpy()
    return atts, enc.word_ids(0)


# ── raw attention ────────────────────────────────────────────────────────────
def raw_attention(word_list, model, tokenizer) -> np.ndarray:
    """Raw attention received per word per layer. -> [n_words, n_layers]."""
    atts, word_ids = _forward_attentions(word_list, model, tokenizer)
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


# ── attention flow ───────────────────────────────────────────────────────────
def _flow_graph(atts_upto, residual=0.5):
    """Layered flow network over tokens; capacities = head-avg attention.

    Adds an identity residual (Abnar & Zeng 2020) and row-renormalizes so each
    token's incoming capacities form a distribution. Node = (layer, token).
    """
    n_layers, seq, _ = atts_upto.shape
    G = nx.DiGraph()
    for l in range(n_layers):
        A = atts_upto[l].copy()
        A = residual * np.eye(seq) + (1 - residual) * A
        A = A / A.sum(axis=1, keepdims=True).clip(min=1e-12)
        for i in range(seq):          # query (receiver at layer l+1)
            for j in range(seq):      # key   (source at layer l)
                c = A[i, j]
                if c > 0:
                    G.add_edge((l, j), (l + 1, i), capacity=float(c))
    return G, n_layers, seq


def attention_flow(word_list, model, tokenizer, decay=1.0) -> np.ndarray:
    """Attention-flow score per word per layer. -> [n_words, n_layers].

    ``decay`` controls the Metzger et al. (2022) position correction: each input
    token's flow is multiplied by ``(position+1) ** decay`` to offset the bias
    toward early tokens before normalization (decay=0 disables it).
    """
    atts, word_ids = _forward_attentions(word_list, model, tokenizer)
    n_layers, seq, _ = atts.shape
    n_words = len(word_list)
    positions = _word_positions(word_ids, n_words)
    target = seq - 1                      # sentence-final token = sink

    out = np.zeros((n_words, n_layers))
    for L in range(n_layers):
        G, _, _ = _flow_graph(atts[: L + 1])
        sink = (L + 1, target)
        tok_flow = np.zeros(seq)
        for i in range(seq):
            src = (0, i)
            if src == sink or not G.has_node(src):
                continue
            val, _ = nx.maximum_flow(G, src, sink, flow_func=nx.algorithms.flow.edmonds_karp)
            tok_flow[i] = val * ((i + 1) ** decay)   # position-decay correction
        word_score = np.array([tok_flow[p].max() if p else 0.0 for p in positions])
        out[:, L] = _normalize(word_score)
    return out


# ── per-sentence extraction over the corpus ──────────────────────────────────
def extract_attention(words_df: pd.DataFrame, model, tokenizer,
                      method: str = "raw") -> pd.DataFrame:
    """Run an attention method over every sentence; long-form output.

    Returns columns ``text_id``, ``word_index_in_text``, ``layer``,
    ``attention``, ``attention_method``. First/last words of each sentence are
    dropped to match the surprisal / reading-time cleaning.
    """
    fn = raw_attention if method == "raw" else attention_flow
    rows = []
    for _, sent in words_df.sort_values(
            ["text_id", "sent_index_in_text", "word_index_in_sent"]).groupby(
            ["text_id", "sent_index_in_text"], sort=False):
        words = sent["word"].fillna("").astype(str).tolist()
        if len(words) < 3:
            continue
        scores = fn(words, model, tokenizer)          # [n_words, n_layers]
        idx = sent["word_index_in_text"].tolist()
        tid = sent["text_id"].iloc[0]
        beg = sent["is_sent_beginning"].tolist()
        end = sent["is_sent_end"].tolist()
        for k in range(len(words)):
            if beg[k] == 1 or end[k] == 1:
                continue
            for l in range(scores.shape[1]):
                rows.append((tid, idx[k], l, scores[k, l], method))
    return pd.DataFrame(
        rows, columns=WORD_KEY + ["layer", "attention", "attention_method"])


# ── eye-tracking features + PCA ──────────────────────────────────────────────
def eyetracking_features(rm: pd.DataFrame) -> pd.DataFrame:
    """Per-word mean of each eye-tracking measure across participants.

    Skips (measure == 0) are excluded from the mean. Returns one row per word
    with the six PoTeC-mapped measures plus ``text_domain`` and
    ``sent_index_in_text`` for within-sentence normalization.
    """
    measures = list(dict.fromkeys(ET_MEASURE_MAP.values()))
    parts = []
    meta = (rm[WORD_KEY + ["text_domain", "sent_index_in_text"]]
            .drop_duplicates(WORD_KEY))
    for m in measures:
        col = (rm[rm[m] > 0].groupby(WORD_KEY)[m].mean().rename(m).reset_index())
        parts.append(col)
    out = meta
    for col in parts:
        out = out.merge(col, on=WORD_KEY, how="left")
    return out


def _zscore_within_sentence(df, cols):
    g = df.groupby(["text_id", "sent_index_in_text"])
    return (df[cols] - g[cols].transform("mean")) / g[cols].transform("std")


def pca_eyetracking(et_df: pd.DataFrame, domain: str):
    """Fit a 1-component PCA on the 4 informative measures for one domain.

    Measures are z-scored within sentence first (per plan / Mouratidi & Poesio
    2025). Returns ``(scored_df, explained_variance_ratio)`` where ``scored_df``
    has ``text_id``, ``word_index_in_text`` and ``pca`` (the component score).
    """
    sub = et_df[et_df["text_domain"] == domain].copy()
    z = _zscore_within_sentence(sub, PCA_MEASURES)
    mask = z.notna().all(axis=1)
    z = z[mask]
    pca = PCA(n_components=1)
    comp = pca.fit_transform(z.values).ravel()
    scored = sub.loc[mask, WORD_KEY].copy()
    scored["pca"] = comp
    return scored, float(pca.explained_variance_ratio_[0])
