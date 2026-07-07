#!/usr/bin/env bash
#
# Push the wiki_* domain corpora to the Hugging Face Hub as dataset repos.
#
# Each data/wiki_<domain> directory is a `datasets.Dataset` written with
# save_to_disk. This loads each one with load_from_disk and pushes it to
# ElyR120/<dir-name> via Dataset.push_to_hub, which re-serializes to parquet
# and therefore skips the stale cache-*.arrow map caches left in the dirs.
#
# Usage:
#   ./scripts/push_datasets_to_hub.sh <hf-token> [--private]
#
#   <hf-token>   HF access token with write scope (required, positional).
#   --private    create private repos (default: public).
#
# Examples:
#   ./scripts/push_datasets_to_hub.sh hf_xxx
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,17p' "$0"
    exit 1
fi

# Token stays in the environment, never on the command line the child sees, so it
# is not exposed in `ps`.
export HF_UPLOAD_TOKEN="$1"
shift

PRIVATE="False"
for arg in "$@"; do
    case "$arg" in
        --private) PRIVATE="True" ;;
        --public) PRIVATE="False" ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done
export HF_PRIVATE="$PRIVATE"

DATA_DIR="data"
if ! compgen -G "$DATA_DIR/wiki_*" > /dev/null; then
    echo "No datasets ($DATA_DIR/wiki_*) found. Scrape first: pixi run python -m src.acquire.wiki_scrape" >&2
    exit 1
fi

echo "==> Pushing datasets under $DATA_DIR/wiki_* to ElyR120/* ..."

pixi run python - "$DATA_DIR" <<'PY'
import os
import sys
from pathlib import Path

from datasets import load_from_disk

NAMESPACE = "ElyR120"

data_dir = Path(sys.argv[1])
token = os.environ["HF_UPLOAD_TOKEN"]
private = os.environ["HF_PRIVATE"] == "True"

dataset_dirs = sorted(p for p in data_dir.glob("wiki_*") if p.is_dir())
for dataset_dir in dataset_dirs:
    repo_id = f"{NAMESPACE}/{dataset_dir.name}"
    print(f"--> {dataset_dir}  ->  {repo_id}  (private={private})")
    ds = load_from_disk(str(dataset_dir))
    ds.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message=f"Upload domain corpus ({dataset_dir.name})",
    )
    print(f"    done: https://huggingface.co/datasets/{repo_id}")

print(f"Pushed {len(dataset_dirs)} dataset repo(s).")
PY

echo
echo "Push complete."
