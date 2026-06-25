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
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

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
    device = next(model.parameters()).device
    out = model(enc.input_ids.to(device), output_attentions=True)
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
def _add_flow_layer(G, A_layer, l, seq, residual=0.5):
    """Add layer ``l``'s edges (l, j) -> (l+1, i) to a layered flow network.

    Capacities = head-avg attention with an identity residual (Abnar & Zeng 2020),
    row-renormalized so each token's incoming capacities form a distribution.
    Node = (layer, token). Mutates ``G`` in place — the graph is built up one
    layer at a time so the network for layer L is the network for L-1 plus this
    layer, never rebuilt from scratch.
    """
    A = residual * np.eye(seq) + (1 - residual) * A_layer
    A = A / A.sum(axis=1, keepdims=True).clip(min=1e-12)
    for i in range(seq):  # query (receiver at layer l+1)
        for j in range(seq):  # key   (source at layer l)
            c = A[i, j]
            if c > 0:
                G.add_edge((l, j), (l + 1, i), capacity=float(c))


def _flow_scores(atts, word_ids, n_words, decay=1.0) -> np.ndarray:
    """Attention-flow score per word per layer from precomputed attentions.

    Pure CPU (networkx max-flow) — no model, so it parallelizes across sentences.
    ``decay`` is the Metzger et al. (2022) position correction: each input token's
    flow is multiplied by ``(position+1) ** decay`` to offset the early-token bias
    before normalization (decay=0 disables it).

    The flow network is grown incrementally: after adding layer L's edges, the
    sink for layer L is node ``(L+1, target)``. Max-flow value to that sink is
    identical whether computed on the L+1-layer subgraph or a full rebuild — the
    network is a layered DAG, so layers above the sink carry no s-t flow.
    """
    n_layers, seq, _ = atts.shape
    positions = _word_positions(word_ids, n_words)
    target = seq - 1  # sentence-final token = sink

    G = nx.DiGraph()
    out = np.zeros((n_words, n_layers))
    for L in range(n_layers):
        _add_flow_layer(G, atts[L], L, seq)
        sink = (L + 1, target)
        tok_flow = np.zeros(seq)
        for i in range(seq):
            src = (0, i)
            if src == sink or not G.has_node(src):
                continue
            val, _ = nx.maximum_flow(
                G, src, sink, flow_func=nx.algorithms.flow.edmonds_karp
            )
            tok_flow[i] = val * ((i + 1) ** decay)  # position-decay correction
        word_score = np.array([tok_flow[p].max() if p else 0.0 for p in positions])
        out[:, L] = _normalize(word_score)
    return out


def attention_flow(word_list, model, tokenizer, decay=1.0) -> np.ndarray:
    """Attention-flow score per word per layer. -> [n_words, n_layers].

    ``decay`` controls the Metzger et al. (2022) position correction (decay=0
    disables it). Forward pass on the model, then ``_flow_scores`` on CPU.
    """
    atts, word_ids = _forward_attentions(word_list, model, tokenizer)
    return _flow_scores(atts, word_ids, len(word_list), decay=decay)


# ── per-sentence extraction over the corpus ──────────────────────────────────
def extract_attention(
    words_df: pd.DataFrame,
    model,
    tokenizer,
    method: str = "raw",
    decay: float = 1.0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run an attention method over every sentence; long-form output.

    Returns columns ``text_id``, ``word_index_in_text``, ``layer``,
    ``attention``, ``attention_method``. First/last words of each sentence are
    dropped to match the surprisal / reading-time cleaning. ``decay`` is the
    Metzger position correction passed to the flow method (decoder default 1.0;
    set 0 for a bidirectional encoder, which has no early-token bias).

    For ``flow`` the per-token Edmonds-Karp max-flow dominates and is pure-CPU,
    single-thread Python — so the GPU forward passes are done first, then the
    flow itself is mapped across ``n_jobs`` cores (``-1`` = all). Scores are
    bit-identical to a serial run; only the schedule changes.
    """
    sents = [
        sent
        for _, sent in words_df.sort_values(
            ["text_id", "sent_index_in_text", "word_index_in_sent"]
        ).groupby(["text_id", "sent_index_in_text"], sort=False)
        if len(sent) >= 3
    ]

    if method == "raw":
        scores_list = [
            raw_attention(s["word"].fillna("").astype(str).tolist(), model, tokenizer)
            for s in sents
        ]
    else:
        # Phase 1 (GPU): forward every sentence, keep head-averaged attentions.
        forwards = [
            _forward_attentions(s["word"].fillna("").astype(str).tolist(), model, tokenizer)
            for s in tqdm(sents, desc="flow forward", leave=False)
        ]
        # Phase 2 (CPU, parallel): max-flow per sentence. joblib preserves order.
        scores_list = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_flow_scores)(atts, word_ids, len(s), decay)
            for s, (atts, word_ids) in zip(sents, forwards)
        )

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
def eyetracking_features(rm: pd.DataFrame) -> pd.DataFrame:
    """Per-word mean of each eye-tracking measure across participants.

    Skips (measure == 0) are excluded from the mean. Returns one row per word
    with the six PoTeC-mapped measures plus ``text_domain`` and
    ``sent_index_in_text`` for within-sentence normalization.
    """
    measures = list(dict.fromkeys(ET_MEASURE_MAP.values()))
    parts = []
    meta = rm[WORD_KEY + ["text_domain", "sent_index_in_text"]].drop_duplicates(
        WORD_KEY
    )
    for m in measures:
        col = rm[rm[m] > 0].groupby(WORD_KEY)[m].mean().rename(m).reset_index()
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
