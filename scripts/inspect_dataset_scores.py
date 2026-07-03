"""Dump the top/bottom NLI-scoring paragraphs per commons-scientific sub-dataset.

The DAPT fine-tuning corpora (``data/domain_phy`` / ``data/domain_bio``) were
selected by the two-pass labeller in ``src.acquire.domain_preprocessing``: TF-IDF
seeds narrow the candidates, then zero-shot NLI (XLM-RoBERTa) scores each and a
calibrated threshold filters them. That per-document ``nli_score`` is NOT persisted
in the saved datasets (they are rebuilt from a cached id list), so this script
recomputes it with the same model + hypotheses on exactly the filtered documents.

For each source (arXiv, OpenAlex, Wikibooks, Wikiversity, DOAB, Polytechnisches
Journal) it prints the first 200 words of the 10 highest- and 10 lowest-scoring
documents, so the selection can be eyeballed. Writes a text file next to itself.

    python scripts/inspect_dataset_scores.py
"""

from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import XLMRobertaTokenizer, pipeline

from src.acquire.domain_preprocessing import BATCH_SIZE, NLI_LABELS, TEXT_CHARS
from src.config import DOMAIN_BIO_DIR, DOMAIN_PHY_DIR

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"  # same model as domain_preprocessing
TOP_N = 10
WORDS = 200
OUT_PATH = Path(__file__).with_name("dataset_score_inspection.txt")


def first_words(text, n=WORDS):
    return " ".join((text or "").split()[:n])


def main():
    domains = {"physics": DOMAIN_PHY_DIR, "biology": DOMAIN_BIO_DIR}
    parts = []
    for domain, path in domains.items():
        ds = load_from_disk(str(path))
        ds = ds.add_column("domain", [domain] * len(ds))
        parts.append(ds)
    ds = concatenate_datasets(parts)
    print(f"loaded {len(ds):,} filtered docs across both fine-tuning corpora")

    device = 0 if torch.cuda.is_available() else -1
    print(f"loading NLI model (device={'cuda' if device == 0 else 'cpu'})...")
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        tokenizer=XLMRobertaTokenizer.from_pretrained(MODEL_NAME),
        device=device,
    )

    hyps = list(NLI_LABELS.values())
    print(f"scoring {len(ds):,} docs (top NLI label score)...")
    results = classifier(
        [(t or "")[:TEXT_CHARS] for t in ds["text"]],
        candidate_labels=hyps,
        multi_label=False,
        batch_size=BATCH_SIZE,
    )
    # nli_score = confidence of the winning label, exactly as the labeller stored it.
    nli_scores = [r["scores"][0] for r in results]
    ds = ds.add_column("nli_score", nli_scores)

    # Group rows by source (the original commons-scientific sub-dataset).
    rows = list(ds)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    lines = []

    def emit(s=""):
        lines.append(s)

    emit("NLI-score inspection of the DAPT fine-tuning corpora")
    emit("(data/domain_phy + data/domain_bio, grouped by commons-scientific source)")
    emit(f"nli_score = winning-label confidence from {MODEL_NAME}")
    emit(f"hypotheses: {NLI_LABELS}")
    emit(f"scored on first {TEXT_CHARS} chars; showing first {WORDS} words per doc")
    emit("=" * 100)

    for source in sorted(by_source):
        group = sorted(by_source[source], key=lambda x: x["nli_score"], reverse=True)
        n = len(group)
        emit("")
        emit("#" * 100)
        emit(f"# SOURCE: {source}   (n={n} docs)")
        emit("#" * 100)

        def dump(label, subset):
            emit("")
            emit(f"----- {label} (by nli_score) -----")
            for i, r in enumerate(subset, 1):
                emit("")
                emit(
                    f"[{label} #{i}] nli_score={r['nli_score']:.4f} "
                    f"domain={r['domain']} id={r['id']} "
                    f"num_tokens={r['num_tokens']} ppl={r['perplexity']:.1f} "
                    f"ocr={r['ocr_score']}"
                )
                emit(first_words(r["text"]))

        if n <= 2 * TOP_N:
            # too few to split cleanly; show them all once, high to low.
            dump(f"ALL {n} DOCS", group)
        else:
            dump(f"TOP {TOP_N}", group[:TOP_N])
            dump(f"BOTTOM {TOP_N}", group[-TOP_N:])

    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
