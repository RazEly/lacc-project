"""Clean a scraped domain corpus: drop off-topic clusters AND off-domain outliers.

The term-targeted Wikipedia scrape (``acquire.scrape``) leaks whole ADJACENT
categories in via the category graph — physics pulls photography-technique,
logic/probability paradoxes, art/craft, and cartoon/fiction articles; biology
pulls cars, autobahns, aircraft, ships, and regulations. A physics-term DENSITY
gate can't separate them: it kills terse real physics (Leidenfrost-Effekt) while
keeping jargon-rich off-topic prose.

Two complementary cuts, because the contamination has two shapes (measured):

  1. CLUSTER drop (BERTopic; Grootendorst 2022). SBERT embed -> UMAP -> HDBSCAN,
     KeyBERTInspired topic labels. Contamination that arrives in bulk forms its
     own NAMED cluster (biology: an autobahn cluster, an aircraft cluster, …) you
     drop wholesale. This is the biology workhorse (~18% off-domain, clustered).

  2. DOC drop (anchor cosine). Every doc is scored by the cosine of its embedding
     to the DOMAIN ANCHOR — the mean embedding of the domain's curriculum terms
     (``scrape.CURRICULUM_TERMS``, the clean whole-subject scrape seeds). Docs
     below ``config.DOMAIN_COS_THRESHOLD[domain]`` are dropped. This catches
     DIFFUSE contamination that never clusters: physics contamination is small
     (~3%) but spread thin, so HDBSCAN buries it inside the one big physics topic
     where no --drop can reach it — the cosine cut removes it directly.

The threshold is per-domain by necessity: the anchor-cosine baseline differs
(physics median ~0.43, biology ~0.35), so a single global cut means different
things in each. Both thresholds live in ``config.DOMAIN_COS_THRESHOLD``.

Two-pass by design (the drop set is a human judgement on the cluster labels):
    # 1. inspect — cluster, print the labelled topic table + cosine histogram,
    #    write the per-doc assignments TSV
    python -m src.acquire.cluster_filter --domain physics

    # 2. filter — drop the topics you judged off-topic (+ HDBSCAN noise) and every
    #    doc under the domain cosine threshold, save the cleaned corpus
    python -m src.acquire.cluster_filter --domain physics --drop 4,7 --drop-noise \
        --out data/wiki_physics_clean

The filter pass never re-clusters or re-embeds: it reads the TSV written by the
inspect pass (topic id + anchor cosine per doc), so it runs in seconds and the
ids/scores are frozen at inspect time.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
from datasets import load_from_disk

from src.acquire.scrape import CURRICULUM_TERMS
from src.config import DEVICE, DOMAIN_COS_THRESHOLD, DOMAIN_DIRS, DOMAINS

# Multilingual SBERT — the corpus is German. MiniLM is the fast default BERTopic
# reaches for; the L12 multilingual variant handles German prose well and keeps
# embedding cheap enough to run on CPU for a few-thousand-doc corpus.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# UMAP/HDBSCAN knobs. min_cluster_size sets granularity: too small shreds physics
# into dozens of micro-topics, too large merges a contamination group back into a
# physics topic. 30 gives ~topic-per-subfield on a ~3k-doc corpus; tune per corpus.
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
MIN_CLUSTER_SIZE = 30
MIN_DF = 2  # c-TF-IDF vocabulary floor (counted over per-topic concatenated docs)
RANDOM_STATE = 42
N_TOPIC_WORDS = 8  # label words shown per topic
N_REP_TITLES = 4  # representative article titles shown per topic


def embed(texts: list[str], model_name: str):
    """SBERT document embeddings (normalised, so dot product == cosine)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=DEVICE)
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def domain_anchor(domain: str, model_name: str) -> np.ndarray:
    """Unit anchor = mean embedding of the domain's curriculum terms."""
    vecs = embed(CURRICULUM_TERMS[domain], model_name)
    anchor = vecs.mean(axis=0)
    return anchor / np.linalg.norm(anchor)


def build_model(embeddings: np.ndarray, model_name: str):
    """BERTopic with seeded UMAP, HDBSCAN, and KeyBERT-style topic labels.

    Embeddings are precomputed and passed to ``fit_transform`` so the same vectors
    drive both the clustering and the anchor cosine (one embed pass, not two).
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    hdbscan = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=SentenceTransformer(model_name, device=DEVICE),
        umap_model=umap,
        hdbscan_model=hdbscan,
        vectorizer_model=CountVectorizer(min_df=MIN_DF),
        representation_model=KeyBERTInspired(),
        calculate_probabilities=False,
        verbose=True,
    )


def topic_relevance(cos: np.ndarray, topics: np.ndarray) -> dict[int, float]:
    """Per-topic domain relevance = mean anchor cosine of the topic's docs."""
    return {int(t): float(cos[topics == t].mean()) for t in set(topics)}


def _topic_label(model, topic_id: int) -> str:
    words = model.get_topic(topic_id) or []
    return ", ".join(w for w, _ in words[:N_TOPIC_WORDS])


def cosine_histogram(cos: np.ndarray, threshold: float, bins: int = 20) -> None:
    """ASCII histogram of anchor cosine with the domain threshold marked."""
    counts, edges = np.histogram(cos, bins=bins)
    scale = 40 / max(counts.max(), 1)
    print(f"\nanchor-cosine histogram (threshold {threshold:.2f} marks off-domain):")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        mark = " <-- threshold" if lo <= threshold < hi else ""
        print(f"  {lo:>6.2f} |{'#' * int(c * scale):40} {c:>4}{mark}")
    n_off = int((cos < threshold).sum())
    print(f"  {n_off} docs below threshold ({100 * n_off / len(cos):.1f}%)")


def report(model, topics, titles, texts, relevance) -> None:
    """Print the labelled topic table, sorted by domain relevance (off-topic last)."""
    rep_doc = model.get_document_info(texts)["Representative_document"].to_numpy()
    order = sorted(set(topics), key=lambda t: relevance[int(t)], reverse=True)
    print(f"\n{'topic':>5} {'n':>5} {'relev':>6}  {'top words':45} representative titles")
    print("-" * 110)
    for t in order:
        n = int((topics == t).sum())
        idx = np.flatnonzero((topics == t) & rep_doc)[:N_REP_TITLES]
        if idx.size == 0:  # noise topic has no representative docs
            idx = np.flatnonzero(topics == t)[:N_REP_TITLES]
        reps = " | ".join(titles[i] for i in idx)
        tag = "  <-- NOISE" if t == -1 else ""
        label = _topic_label(model, t) if t != -1 else "(unclustered outliers)"
        print(f"{t:>5} {n:>5} {relevance[int(t)]:>6.3f}  {label[:45]:45} {reps[:60]}{tag}")
    print(
        "\nInspect the low-relevance / off-topic-labelled topics, then re-run with"
        "\n  --drop <ids>  (comma-separated)   [--drop-noise]   --out <dir>"
        "\n(docs under the domain cosine threshold are dropped automatically too)"
    )


def write_assignments(path, titles, topics, cos, relevance, num_words) -> None:
    """Per-doc TSV — the audit trail AND the filter pass's input (no re-embed)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            ["doc_idx", "title", "topic", "anchor_cos", "topic_relevance", "num_words"]
        )
        for i, (title, t, c, nw) in enumerate(zip(titles, topics, cos, num_words)):
            w.writerow([i, title, int(t), f"{c:.4f}", f"{relevance[int(t)]:.4f}", nw])
    print(f"wrote per-doc assignments -> {path}")


def inspect_pass(ds, domain: str, model_name: str, assignments_path: str) -> None:
    """Embed once, cluster, score by anchor cosine, report, write the TSV."""
    titles = np.array(ds["title"], dtype=object)
    texts = list(ds["text"])
    num_words = ds["num_words"] if "num_words" in ds.column_names else [0] * len(ds)

    embeddings = embed(texts, model_name)
    anchor = domain_anchor(domain, model_name)
    cos = embeddings @ anchor

    model = build_model(embeddings, model_name)
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    topics = np.asarray(topics)
    relevance = topic_relevance(cos, topics)

    n_topics = len({t for t in topics if t != -1})
    n_noise = int((topics == -1).sum())
    print(f"[{domain}] {n_topics} topics, {n_noise} noise docs")
    report(model, topics, titles, texts, relevance)
    cosine_histogram(cos, DOMAIN_COS_THRESHOLD[domain])
    write_assignments(assignments_path, titles, topics, cos, relevance, num_words)


def filter_pass(ds, domain, assignments_path, drop_ids, out) -> None:
    """Drop chosen topics + sub-threshold docs from the inspect TSV — no recompute."""
    try:
        with open(assignments_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    except FileNotFoundError:
        raise SystemExit(
            f"{assignments_path} not found — run the inspect pass first (no --drop)"
        )
    if len(rows) != len(ds) or [r["title"] for r in rows] != [
        str(t) for t in ds["title"]
    ]:
        raise SystemExit(
            f"{assignments_path} doesn't match the corpus (order or length) — the "
            "corpus changed since the inspect pass; re-run it"
        )
    topics = np.array([int(r["topic"]) for r in rows])
    cos = np.array([float(r["anchor_cos"]) for r in rows])

    threshold = DOMAIN_COS_THRESHOLD[domain]
    off_topic = np.isin(topics, sorted(drop_ids))
    off_domain = cos < threshold
    drop = off_topic | off_domain

    cleaned = ds.select(np.flatnonzero(~drop).tolist())
    print(
        f"\n[{domain}] dropped {int(drop.sum())} of {len(ds)} docs "
        f"({100 * drop.sum() / len(ds):.1f}%):"
        f"\n  topics {sorted(drop_ids)}: {int(off_topic.sum())} docs"
        f"\n  cosine < {threshold:.2f}: {int(off_domain.sum())} docs "
        f"({int((off_domain & ~off_topic).sum())} not already in a dropped topic)"
    )
    if "num_words" in ds.column_names:
        print(f"  words {sum(ds['num_words']):,} -> {sum(cleaned['num_words']):,}")
    cleaned.save_to_disk(out)
    print(f"[{domain}] saved cleaned corpus -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="physics", choices=DOMAINS)
    ap.add_argument(
        "corpus", nargs="?", help="corpus dir (default: the --domain wiki corpus)"
    )
    ap.add_argument("--model", default=EMBED_MODEL, help="SBERT embedding model")
    ap.add_argument(
        "--drop", default="", help="comma-separated topic ids to remove (filter pass)"
    )
    ap.add_argument(
        "--drop-noise", action="store_true", help="also remove HDBSCAN noise (topic -1)"
    )
    ap.add_argument("--out", help="save the filtered corpus here (required to filter)")
    ap.add_argument(
        "--assignments",
        default="topic_assignments.tsv",
        help="per-doc topic TSV (written by inspect pass, read by filter pass)",
    )
    args = ap.parse_args()

    # Validate up front — not after minutes of embedding.
    try:
        drop_ids = {int(x) for x in args.drop.split(",") if x.strip()}
    except ValueError:
        ap.error(f"--drop takes comma-separated topic ids, got {args.drop!r}")
    if args.drop_noise:
        drop_ids.add(-1)
    # --out triggers the filter pass; the cosine cut runs even with no --drop.
    if drop_ids and not args.out:
        ap.error("--out is required when dropping topics")

    path = args.corpus or str(DOMAIN_DIRS[args.domain])
    ds = load_from_disk(path)
    print(f"[{args.domain}] {len(ds)} docs from {path}")

    if args.out:
        filter_pass(ds, args.domain, args.assignments, drop_ids, args.out)
    else:
        inspect_pass(ds, args.domain, args.model, args.assignments)


if __name__ == "__main__":
    main()
