"""Cluster a scraped domain corpus and drop the off-topic clusters (BERTopic).

The term-targeted Wikipedia scrape (``acquire.scrape``) leaks whole ADJACENT
categories in via the category graph — physics pulls photography-technique,
logic/probability paradoxes, art/craft, and cartoon/fiction articles (~12-14 % of
the physics corpus). A physics-term DENSITY gate can't separate them: it kills
terse real physics (Leidenfrost-Effekt) while keeping jargon-rich off-topic prose.

The literature answer is to cluster in an embedding space, not count words
(Grootendorst 2022, BERTopic). Pipeline:
    SBERT embed -> UMAP reduce -> HDBSCAN cluster -> c-TF-IDF topic labels.
HDBSCAN also flags docs that fit no dense region as noise (topic -1). Each cluster
gets a keyword label so the contamination groups surface as NAMED topics you drop
wholesale, keeping the physics topics.

To guide which topic ids are off-topic, every topic is scored by the cosine of its
centroid to a DOMAIN ANCHOR — the mean SBERT embedding of the domain's curriculum
terms (``scrape.CURRICULUM_TERMS``). Off-topic clusters float to the bottom of the
ranking; the label + representative titles confirm the call.

Two-pass by design (the drop set is a human judgement on the labels):
    # 1. inspect — cluster, print the labelled topic table, write assignments TSV
    python -m src.acquire.cluster_filter --domain physics

    # 2. filter — drop the topics you judged off-topic (+ HDBSCAN noise), save
    python -m src.acquire.cluster_filter --domain physics --drop 4,7,9 --drop-noise \
        --out data/wiki_physics_clean

Deterministic: UMAP is seeded, so a re-run reproduces the same topic ids.
"""

from __future__ import annotations

import argparse

import numpy as np
from datasets import load_from_disk

from src.acquire.scrape import CURRICULUM_TERMS
from src.config import DEVICE, DOMAIN_DIRS, DOMAINS

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
RANDOM_STATE = 42
N_TOPIC_WORDS = 8  # c-TF-IDF words shown per topic
N_REP_TITLES = 4  # representative article titles shown per topic

# German function words stripped from the c-TF-IDF vocabulary so topic labels are
# content words (photography/paradox/craft/cat…), not "die, der, und". sklearn ships
# no German list; this is the common closed-class set (articles, pronouns, aux/modal
# verbs, prepositions, conjunctions) — enough to surface topical nouns.
GERMAN_STOPWORDS = frozenset(
    """aber alle allem allen aller alles als also am an ander andere anderem anderen
    anderer anderes auch auf aus bei bein beim bin bis bist da damit dann das dass
    dasselbe dazu dein deine dem den denn der derer des dessen dich die dies diese
    diesem diesen dieser dieses dir doch dort du durch ein eine einem einen einer
    eines einig einige einigem einigen einiger einiges einmal er es etwas euer eure
    fr für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich ihm ihn
    ihnen ihr ihre ihrem ihren ihrer ihres im in indem ins ist ja jede jedem jeden
    jeder jedes jene jenem jenen jener jenes jetzt kann kein keine keinem keinen
    keiner keines knnen knnte machen man manche manchem manchen mancher manches mein
    meine meinem meinen meiner meines mich mir mit muss musste nach nicht nichts noch
    nun nur ob oder ohne sehr sein seine seinem seinen seiner seines selbst sich sie
    sind so solche solchem solchen solcher solches soll sollte sondern sonst ber um
    und uns unse unsem unsen unser unses unter viel vom von vor war waren warst was
    weg weil weiter welche welchem welchen welcher welches wenn werde werden wie wieder
    will wir wird wirst wo wollen wollte wrde wrden zu zum zur zwar zwischen ist eine
    einer wurde wurden sowie bzw etc dabei dabing dazu jedoch mehr wobei worden""".split()
)


def _embed(texts: list[str], model_name: str):
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


def _domain_anchor(domain: str, model_name: str) -> np.ndarray:
    """Unit anchor vector for a domain = mean embedding of its curriculum terms.

    ``CURRICULUM_TERMS`` are the whole-subject seeds of the scrape (Quantenmechanik,
    Thermodynamik, …) — a clean domain centroid free of any scraped contamination.
    """
    vecs = _embed(CURRICULUM_TERMS[domain], model_name)
    anchor = vecs.mean(axis=0)
    return anchor / np.linalg.norm(anchor)


def fit_topics(texts: list[str], embeddings: np.ndarray):
    """Fit BERTopic on precomputed embeddings; return the fitted model + topics."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
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
    # Strip German function words from the c-TF-IDF vocabulary so labels are topical.
    vectorizer = CountVectorizer(stop_words=list(GERMAN_STOPWORDS), min_df=2)
    model = BERTopic(
        umap_model=umap,
        hdbscan_model=hdbscan,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    return model, np.asarray(topics)


def topic_relevance(embeddings, topics, anchor: np.ndarray) -> dict[int, float]:
    """Cosine of each topic's mean embedding to the domain anchor (higher = on-topic)."""
    rel: dict[int, float] = {}
    for t in sorted(set(topics)):
        centroid = embeddings[topics == t].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        rel[t] = float(centroid @ anchor)
    return rel


def _topic_label(model, topic_id: int) -> str:
    words = model.get_topic(topic_id) or []
    return ", ".join(w for w, _ in words[:N_TOPIC_WORDS])


def report(model, topics, titles, relevance) -> None:
    """Print the labelled topic table, sorted by domain relevance (off-topic last)."""
    info = model.get_topic_info().set_index("Topic")
    print(f"\n{'topic':>5} {'n':>5} {'relev':>6}  {'top words':45} representative titles")
    print("-" * 110)
    for t in sorted(relevance, key=relevance.get, reverse=True):
        n = int(info.loc[t, "Count"]) if t in info.index else int((topics == t).sum())
        idx = np.flatnonzero(topics == t)[:N_REP_TITLES]
        reps = " | ".join(titles[i] for i in idx)
        tag = "  <-- NOISE" if t == -1 else ""
        label = _topic_label(model, t) if t != -1 else "(unclustered outliers)"
        print(f"{t:>5} {n:>5} {relevance[t]:>6.3f}  {label[:45]:45} {reps[:60]}{tag}")
    print(
        "\nInspect the low-relevance / off-topic-labelled topics, then re-run with"
        "\n  --drop <ids>  (comma-separated)   [--drop-noise]   --out <dir>"
    )


def write_assignments(path: str, titles, topics, relevance, num_words) -> None:
    """Per-doc topic assignment TSV (title, topic, relevance, words) for auditing."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["title", "topic", "topic_relevance", "num_words"])
        for title, t, nw in zip(titles, topics, num_words):
            w.writerow([title, int(t), f"{relevance[t]:.4f}", nw])
    print(f"wrote per-doc assignments -> {path}")


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
    ap.add_argument("--out", help="save the filtered corpus here (required with --drop)")
    ap.add_argument(
        "--assignments",
        default="topic_assignments.tsv",
        help="where to write the per-doc topic TSV",
    )
    args = ap.parse_args()

    path = args.corpus or str(DOMAIN_DIRS[args.domain])
    ds = load_from_disk(path)
    titles = np.array(ds["title"], dtype=object)
    texts = list(ds["text"])
    num_words = list(ds["num_words"]) if "num_words" in ds.column_names else [0] * len(ds)
    print(f"[{args.domain}] {len(ds)} docs from {path}")

    embeddings = _embed(texts, args.model)
    anchor = _domain_anchor(args.domain, args.model)
    model, topics = fit_topics(texts, embeddings)
    relevance = topic_relevance(embeddings, topics, anchor)

    n_topics = len({t for t in topics if t != -1})
    n_noise = int((topics == -1).sum())
    print(f"[{args.domain}] {n_topics} topics, {n_noise} noise docs")
    report(model, topics, titles, relevance)
    write_assignments(args.assignments, titles, topics, relevance, num_words)

    drop_ids = {int(x) for x in args.drop.split(",") if x.strip()}
    if args.drop_noise:
        drop_ids.add(-1)
    if not drop_ids:
        return  # inspect pass only

    if not args.out:
        ap.error("--out is required when dropping topics")
    keep = np.array([t not in drop_ids for t in topics])
    cleaned = ds.select(np.flatnonzero(keep).tolist())
    removed = len(ds) - len(cleaned)
    words_before = sum(num_words)
    words_after = sum(cleaned["num_words"]) if "num_words" in ds.column_names else 0
    print(
        f"\n[{args.domain}] dropped topics {sorted(drop_ids)}: "
        f"-{removed} docs ({100 * removed / len(ds):.1f}%), "
        f"words {words_before:,} -> {words_after:,}"
    )
    cleaned.save_to_disk(args.out)
    print(f"[{args.domain}] saved cleaned corpus -> {args.out}")


if __name__ == "__main__":
    main()
