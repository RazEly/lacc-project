#!/usr/bin/env bash
#
# One-shot project setup for a fresh (remote) machine.
#   - builds the pixi env from pixi.lock (Python deps + R + lme4/lmerTest)
#   - clones PoTeC, downloads PoTeC eye-tracking data, scrapes the domain corpora
#
# Usage:
#   ./init.sh                 # full setup incl. PoTeC eye-tracking data
#
set -euo pipefail

cd "$(dirname "$0")"

POTEC_REPO_URL="https://github.com/DiLi-Lab/PoTeC"
POTEC_DIR="data/potec"

if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi not found. Install it: https://pixi.sh/latest/#installation"
    echo "  curl -fsSL https://pixi.sh/install.sh | bash   (then restart your shell)"
    exit 1
fi

echo "==> Building env from pixi.lock (Python + R backend) ..."
pixi install

# Run subsequent python through the pixi-managed env.
PY="pixi run python"

echo "==> Cloning PoTeC ..."
if [[ -d "$POTEC_DIR" ]]; then
    echo "    $POTEC_DIR already exists, skipping clone."
else
    git clone --depth 1 "$POTEC_REPO_URL" "$POTEC_DIR"
fi

echo "==> Downloading PoTeC eye-tracking data ..."
$PY -m src.acquire.download_potec

echo "==> Downloading Datasets ..."
scripts/download_datasets_from_hub.sh

echo
echo "Setup complete."
