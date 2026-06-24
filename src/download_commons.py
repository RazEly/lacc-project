"""Fetch the german-commons scientific corpus.

Downloads the ``Scientific`` subset of ``coral-nlp/german-commons`` from the
Hugging Face Hub. The hub config returns one split per source (arxiv, doab,
openalex, ...); we concatenate them into a single flat dataset and
``save_to_disk`` so ``domain_preprocessing`` can read it with
``load_from_disk`` as one table.

    python -m src.download_commons
"""
from datasets import concatenate_datasets, load_dataset

from src.config import COMMONS_DIR, COMMONS_HF_CONFIG, COMMONS_HF_REPO


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
