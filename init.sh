#!/usr/bin/env bash
#
# One-shot project setup for a fresh (remote) machine.
#   - creates the uv-managed env from the lockfile (uv sync)
#   - clones PoTeC and downloads PoTeC eye-tracking + german-commons corpora
#
# Usage:
#   ./init.sh                 # full setup incl. PoTeC eye-tracking data
#   ./init.sh --no-eyetracking  # skip the large OSF eye-tracking download
#
set -euo pipefail

cd "$(dirname "$0")"

POTEC_REPO_URL="https://github.com/DiLi-Lab/PoTeC"
POTEC_DIR="data/potec"

EYETRACKING=1
if [[ "${1:-}" == "--no-eyetracking" ]]; then
    EYETRACKING=0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "==> Syncing dependencies (uv sync) ..."
uv sync

# Run subsequent python through the uv-managed env.
PY="uv run python"

echo "==> Cloning PoTeC ..."
if [[ -d "$POTEC_DIR" ]]; then
    echo "    $POTEC_DIR already exists, skipping clone."
else
    git clone --depth 1 "$POTEC_REPO_URL" "$POTEC_DIR"
fi

if [[ "$EYETRACKING" -eq 1 ]]; then
    echo "==> Downloading PoTeC eye-tracking data ..."
    $PY -m src.acquire.download_potec
else
    echo "==> Skipping PoTeC eye-tracking download (--no-eyetracking)."
fi

echo "==> Downloading german-commons (scientific) ..."
$PY -m src.acquire.download_commons

echo "==> Labelling corpus by domain (physics / biology) ..."
$PY -m src.acquire.domain_preprocessing

echo
echo "Setup complete."
