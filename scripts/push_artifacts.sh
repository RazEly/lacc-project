#!/usr/bin/env bash
# Push completed DAPT run dirs in artifacts/ to the Hugging Face Hub via the `hf` CLI.
#
# A "run" is one finetune_dapt output dir (artifacts/<model>_<domain>_lora): the
# checkpoint_NN/ dirs plus manifest.csv + run_signature.json. Each is mirrored to
# its canonical public repo  ElyR120/potec-dapt-<dir>  — the same id hub.repo_id_for
# computes — so runs trained before Hub mirroring was wired up get published without
# retraining. The transient Trainer _hf/ dir and __pycache__ are excluded, matching
# hub.upload_run's ignore_patterns.
#
# Needs a write token for the HF_NAMESPACE (owner only): export HF_TOKEN=hf_xxx, or
# run `hf auth login` first. Usage:
#
#     HF_TOKEN=hf_xxx scripts/push_artifacts.sh
#
set -euo pipefail

NAMESPACE="ElyR120"
PREFIX="potec-dapt-"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$ROOT/artifacts"

# Prefer the project venv's hf, fall back to whatever is on PATH.
HF="$ROOT/.venv/bin/hf"
[ -x "$HF" ] || HF="hf"
command -v "$HF" >/dev/null 2>&1 || { echo "error: 'hf' CLI not found (.venv/bin/hf or PATH)"; exit 1; }

if [ -z "${HF_TOKEN:-}" ] && ! "$HF" auth whoami >/dev/null 2>&1; then
  echo "error: no HF_TOKEN and not logged in. export HF_TOKEN=hf_xxx or run '$HF auth login'"; exit 1
fi

shopt -s nullglob
runs=()
for d in "$ARTIFACTS"/*/; do
  [ -f "${d}run_signature.json" ] && runs+=("${d%/}")
done

if [ "${#runs[@]}" -eq 0 ]; then
  echo "no run dirs under $ARTIFACTS"; exit 0
fi

# Retry each upload a few times: large transfers can hit connection timeouts
# mid-flight. `hf upload` hashes files and skips already-pushed ones, so a
# retry resumes where the last attempt died instead of restarting.
MAX_TRIES="${MAX_TRIES:-5}"

# Mirror NUM_CKPTS checkpoints per run (sorted), starting at index CKPT_START,
# not all of them. Each checkpoint_NN/ is uploaded as its own commit so one
# connection drop only loses that checkpoint, not the whole run. Checkpoints
# 0-2 are already pushed; default to the next batch (3-7).
CKPT_START="${CKPT_START:-3}"
NUM_CKPTS="${NUM_CKPTS:-5}"

# push REPO LOCAL_PATH PATH_IN_REPO MSG — upload one path with retry/backoff.
push() {
  local repo_id="$1" local_path="$2" path_in_repo="$3" msg="$4" try=1
  until "$HF" upload "$repo_id" "$local_path" "$path_in_repo" \
        --repo-type model \
        --exclude "_hf/*" "**/__pycache__/*" \
        --commit-message "$msg"; do
    if [ "$try" -ge "$MAX_TRIES" ]; then
      echo "error: $repo_id <- $path_in_repo failed after $MAX_TRIES tries" >&2
      exit 1
    fi
    echo "  retry $((++try))/$MAX_TRIES for $path_in_repo in ${try}s..." >&2
    sleep "$try"
  done
}

echo "pushing ${#runs[@]} run(s) to $NAMESPACE ($NUM_CKPTS checkpoint(s) each from index $CKPT_START, up to $MAX_TRIES tries):"
for d in "${runs[@]}"; do
  name="$(basename "$d")"
  repo_id="$NAMESPACE/$PREFIX$name"
  echo "  -> $repo_id"

  ckpts=("$d"/checkpoint_*/)
  for ckpt in "${ckpts[@]:$CKPT_START:$NUM_CKPTS}"; do
    cname="$(basename "$ckpt")"
    echo "    checkpoint $cname"
    push "$repo_id" "${ckpt%/}" "$cname" "mirror $name/$cname"
  done

  # run-level metadata so the partial mirror is still self-describing
  for f in run_signature.json manifest.csv; do
    [ -f "$d/$f" ] && { echo "    $f"; push "$repo_id" "$d/$f" "$f" "mirror $name/$f"; }
  done
done
