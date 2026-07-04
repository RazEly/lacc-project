"""Fetch the german-commons scientific corpus.

Downloads ``coral-nlp/german-commons`` [scientific], flattens its per-source
splits into one dataset, and ``save_to_disk`` for ``domain_preprocessing``.
No filtering here — every quality gate (OCR score, length) lives in
``domain_preprocessing.apply_quality_filter`` so the raw corpus stays complete
on disk.

    python -m src.acquire.download_commons
"""
from datasets import concatenate_datasets, load_dataset

from src.config import COMMONS_DIR

COMMONS_HF_REPO = "coral-nlp/german-commons"
COMMONS_HF_CONFIG = "scientific"


def main() -> None:
    if COMMONS_DIR.exists():
        print(f"{COMMONS_DIR} already exists, skipping download.")
        return

    print(f"Downloading {COMMONS_HF_REPO} [{COMMONS_HF_CONFIG}] from the Hugging Face Hub ...")
    dsd = load_dataset(COMMONS_HF_REPO, COMMONS_HF_CONFIG)

    # Flatten per-source splits into a single dataset.
    ds = concatenate_datasets([dsd[name] for name in dsd.keys()])
    print(f"  {len(ds):,} documents across sources: {list(dsd.keys())}")

    COMMONS_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(COMMONS_DIR))
    print(f"  saved to {COMMONS_DIR}")


if __name__ == "__main__":
    main()
