#!/usr/bin/env bash
#
# Push the DAPT LoRA checkpoints to the Hugging Face Hub.
#
# Each DAPT run directory (artifacts/<base-model>_<domain>_lora) holds one
# model's full checkpoint schedule (checkpoint-<step>/dapt adapters + manifest.csv);
# `src.modeling.finetune.run_dir_for` builds the name. This uploads each run dir
# as its own public model repo ElyR120/<run-dir-name> via
# huggingface_hub.upload_folder (huggingface_hub >= 0.34; env has 0.36.2).
#
# Usage:
#   ./scripts/push_to_hub.sh <hf-token> [--private]
#
#   <hf-token>   HF access token with write scope (required, positional).
#   --private    create private repos (default: public).
#
# Examples:
#   ./scripts/push_to_hub.sh hf_xxx
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

ARTIFACTS_DIR="artifacts"
if ! compgen -G "$ARTIFACTS_DIR/*_lora" > /dev/null; then
    echo "No DAPT run dirs ($ARTIFACTS_DIR/*_lora) found. Train first: pixi run python -m src.modeling.finetune" >&2
    exit 1
fi

echo "==> Pushing DAPT run dirs under $ARTIFACTS_DIR/*_lora to ElyR120/* ..."

pixi run python - "$ARTIFACTS_DIR" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

NAMESPACE = "ElyR120"

artifacts_dir = Path(sys.argv[1])
token = os.environ["HF_UPLOAD_TOKEN"]
private = os.environ["HF_PRIVATE"] == "True"

api = HfApi(token=token)

run_dirs = sorted(p for p in artifacts_dir.glob("*_lora") if p.is_dir())
for run_dir in run_dirs:
    repo_id = f"{NAMESPACE}/{run_dir.name}"
    print(f"--> {run_dir}  ->  {repo_id}  (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload DAPT LoRA checkpoints ({run_dir.name})",
    )
    print(f"    done: https://huggingface.co/{repo_id}")

print(f"Pushed {len(run_dirs)} model repo(s).")
PY

echo
echo "Push complete."
