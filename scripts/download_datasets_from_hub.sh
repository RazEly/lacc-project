#!/usr/bin/env bash
#
# Download the wiki_* domain corpora from the Hugging Face Hub.
#
# Inverse of push_datasets_to_hub.sh. Each public dataset repo
# ElyR120/wiki_<domain> (one per domain in `src.config.DOMAINS`) is loaded with
# datasets.load_dataset and written with save_to_disk into data/wiki_<domain>
# (`src.config.DOMAIN_DIRS`), so downstream load_from_disk callers find the
# corpora where the scrape pipeline would have written them. A missing repo is
# warned about, not fatal.
#
# Usage:
#   ./scripts/download_datasets_from_hub.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,15p' "$0"
    exit 0
fi

echo "==> Downloading datasets from ElyR120/wiki_* into data/ ..."

pixi run python - <<'PY'
from datasets import load_dataset
from huggingface_hub.utils import RepositoryNotFoundError

from src.config import DOMAIN_DIRS

NAMESPACE = "ElyR120"

pulled = 0
for domain, local_dir in DOMAIN_DIRS.items():
    repo_id = f"{NAMESPACE}/{local_dir.name}"
    print(f"--> {repo_id}  ->  {local_dir}")
    try:
        ds = load_dataset(repo_id, split="train")
    except RepositoryNotFoundError:
        print(f"    skip: {repo_id} not found (not pushed yet)")
        continue
    ds.save_to_disk(str(local_dir))
    pulled += 1
    print("    done")

print(f"Downloaded {pulled}/{len(DOMAIN_DIRS)} dataset(s).")
PY

echo
echo "Download complete."
