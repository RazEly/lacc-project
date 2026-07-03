"""Fetch the german-commons scientific corpus.

Downloads ``coral-nlp/german-commons`` [scientific], flattens its per-source
splits into one dataset, and ``save_to_disk`` for ``domain_preprocessing``.

    python -m src.acquire.download_commons
"""
from datasets import concatenate_datasets, load_dataset

from src.config import COMMONS_DIR

COMMONS_HF_REPO = "coral-nlp/german-commons"
COMMONS_HF_CONFIG = "scientific"
# OpenAlex is OCR'd; keep only its high-quality docs. ocr_score is 0-100. This is
# the only OCR gate in the pipeline — the saved corpus is clean downstream.
OPENALEX_SOURCE = "openalex"
OPENALEX_MIN_OCR = 90.0


def _keep_openalex(row) -> bool:
    """OpenAlex is OCR'd: keep only docs with a high ocr_score (0-100)."""
    score = row["ocr_score"]
    return score is not None and score == score and score > OPENALEX_MIN_OCR


def main() -> None:
    if COMMONS_DIR.exists():
        print(f"{COMMONS_DIR} already exists, skipping download.")
        return

    print(f"Downloading {COMMONS_HF_REPO} [{COMMONS_HF_CONFIG}] from the Hugging Face Hub ...")
    dsd = load_dataset(COMMONS_HF_REPO, COMMONS_HF_CONFIG)

    # Gate OpenAlex on OCR quality; keep every other source in full.
    if OPENALEX_SOURCE in dsd:
        before = len(dsd[OPENALEX_SOURCE])
        dsd[OPENALEX_SOURCE] = dsd[OPENALEX_SOURCE].filter(_keep_openalex)
        print(f"  openalex ocr>{OPENALEX_MIN_OCR}: {len(dsd[OPENALEX_SOURCE]):,}/{before:,} kept")

    # Flatten per-source splits into a single dataset.
    ds = concatenate_datasets([dsd[name] for name in dsd.keys()])
    print(f"  {len(ds):,} documents across sources: {list(dsd.keys())}")

    COMMONS_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(COMMONS_DIR))
    print(f"  saved to {COMMONS_DIR}")


if __name__ == "__main__":
    main()
