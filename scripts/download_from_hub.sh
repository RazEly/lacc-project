#!/usr/bin/env bash
#
# Download the DAPT LoRA checkpoints from the Hugging Face Hub.
#
# Inverse of push_to_hub.sh. The expected repo set is the model x domain grid
# from `src.modeling.finetune` (run_dir_for builds each run-dir name); each public
# repo ElyR120/<run-dir-name> is pulled into artifacts/<run-dir-name> so the
# surprisal loader (`src.modeling.lm.load_causal_lm`) finds the checkpoints where
# training would have written them. Uses huggingface_hub.snapshot_download
# (env has 0.36.2). A missing repo is warned about, not fatal.
#
# Usage:
#   ./scripts/download_from_hub.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,17p' "$0"
    exit 0
fi

echo "==> Downloading DAPT run dirs from ElyR120/* into artifacts/ ..."

pixi run python - <<'PY'
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from src.config import ARTIFACTS_DIR, DOMAINS
from src.modeling.finetune import MODELS, run_dir_for

NAMESPACE = "ElyR120"

# Same run-dir names training would have produced, one per base model x domain.
run_names = [
    run_dir_for(hf_repo, domain).name
    for hf_repo in MODELS.values()
    for domain in DOMAINS
]

pulled = 0
for name in run_names:
    repo_id = f"{NAMESPACE}/{name}"
    local_dir = ARTIFACTS_DIR / name
    print(f"--> {repo_id}  ->  {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
        )
        pulled += 1
        print("    done")
    except RepositoryNotFoundError:
        print(f"    skip: {repo_id} not found (not pushed yet)")

print(f"Downloaded {pulled}/{len(run_names)} run dir(s).")
PY

echo
echo "Download complete."
