#!/usr/bin/env bash
#
# Download the wiki_* domain corpora from the Hugging Face Hub.
#
# Inverse of push_datasets_to_hub.sh. Each public dataset repo
# ElyR120/wiki_<domain> (one per domain in `src.config.DOMAINS`, plus the
# off-domain `wiki_neutral` corpus) is loaded with
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

from src.config import DOMAIN_DIRS, NEUTRAL_DIR

NAMESPACE = "ElyR120"

# Neutral corpus lives on the Hub as wiki_neutral but is deliberately kept out of
# DOMAINS/DOMAIN_DIRS (not a DAPT domain), so pull it explicitly alongside them.
local_dirs = [*DOMAIN_DIRS.values(), NEUTRAL_DIR]

pulled = 0
for local_dir in local_dirs:
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

print(f"Downloaded {pulled}/{len(local_dirs)} dataset(s).")
PY

echo
echo "Download complete."
