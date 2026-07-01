#!/usr/bin/env bash
#
# One-shot project setup for a fresh (remote) machine.
#   - builds the pixi env from pixi.lock (Python deps + R + lme4/lmerTest)
#   - clones PoTeC and downloads PoTeC eye-tracking + german-commons corpora
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

# ── Hugging Face Hub checkpoint mirror ────────────────────────────────────────
# DAPT checkpoints are mirrored to fixed PUBLIC repos (src.modeling.hub
# HF_NAMESPACE) so expensive GPU runs are reused across machines — the download
# tier after local artifacts/. Download needs NO credentials; it always pulls the
# owner's public checkpoints. A write HF_TOKEN for that account additionally
# enables PUSH of freshly trained runs (owner only) — leave blank for everyone else.
if [[ -z "${HF_TOKEN:-}" ]]; then
    read -r -s -p "Hugging Face write token to PUSH new checkpoints (blank = download-only): " HF_TOKEN
    echo
    export HF_TOKEN
fi
echo "    Hub mirror: download=always, push=$([[ -n "${HF_TOKEN:-}" ]] && echo yes || echo 'no (download-only)')"
[[ -n "${HF_TOKEN:-}" ]] && echo "    (note: HF_TOKEN export lives only in this shell; re-set it for later runs.)"

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

echo "==> Downloading german-commons (scientific) ..."
$PY -m src.acquire.download_commons

echo "==> Labelling corpus by domain (physics / biology) ..."
$PY -m src.acquire.domain_preprocessing

echo
echo "Setup complete."
