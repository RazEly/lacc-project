"""Hugging Face Hub mirror for DAPT checkpoint runs.

A "run" is one ``finetune_dapt`` output directory (e.g.
``artifacts/german-gpt2_physics_lora``): the per-index ``checkpoint_NN/`` dirs
plus ``manifest.csv`` and ``run_signature.json``. This module mirrors a whole
run dir to/from one public Hub repo so an expensive GPU run is reused across
machines, not just re-run-to-run on the same disk.

Resolution order in ``finetune_dapt`` is local cache -> Hub -> train. The Hub is
only consulted when the local cache misses; a downloaded run is reused under the
*same* signature check as a local one (``_load_cached_manifest``), so a config
change still retrains. Everything here degrades gracefully: offline or a missing
repo returns ``None`` (download) or skips (upload) rather than raising, so the
pipeline always falls back to plain local training.

The repos live under a FIXED namespace (``HF_NAMESPACE``) — the project owner's
account — regardless of who runs the code, so anyone always downloads the same
canonical public checkpoints with zero configuration. Upload, by contrast, needs
a write token for that namespace, so only the owner pushes. Behaviour is
token-driven: a write ``HF_TOKEN`` enables download AND upload (push a fresh run);
no token is download-only (reuse the public runs, never push). Env vars:

* ``HF_TOKEN`` (or ``huggingface-cli login``) — write token for ``HF_NAMESPACE``;
  presence enables upload. Absent -> download-only.
* ``HF_AUTO_CHECKPOINTS=0`` — disable both the Hub download and upload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Fixed owner of the canonical public checkpoint repos. Hardcoded (not env) so a
# fresh machine downloads the same weights no matter who runs it; only a token
# holder for this account can push. Repo ids: <HF_NAMESPACE>/<prefix><run-dir>,
# e.g. ElyR120/potec-dapt-german-gpt2_physics_lora.
HF_NAMESPACE = "ElyR120"
REPO_PREFIX = "potec-dapt-"


def _enabled() -> bool:
    """Hub mirroring is on unless explicitly disabled."""
    return os.environ.get("HF_AUTO_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "")


def _api():
    """``(HfApi, has_token)`` or ``None`` if the Hub is disabled.

    The namespace is always the hardcoded ``HF_NAMESPACE``; ``has_token`` gates
    upload (write needs a token), while download works tokenless on the public
    repos — so checkpoints are fetched even with no env vars set at all.
    """
    if not _enabled():
        return None
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or None
        return HfApi(token=token), token is not None
    except Exception as e:  # noqa: BLE001 — Hub is best-effort, never fatal
        print(f"  [hub] disabled: {type(e).__name__}: {e}")
        return None


def repo_id_for(out_dir: Path) -> str:
    """Hub repo id mirroring run directory ``out_dir`` under the fixed namespace."""
    return f"{HF_NAMESPACE}/{REPO_PREFIX}{Path(out_dir).name}"


def try_download_run(out_dir: Path, signature: dict) -> pd.DataFrame | None:
    """Pull a matching run from the Hub into ``out_dir``; return its manifest.

    Returns ``None`` (and downloads nothing kept) when the Hub is disabled, the
    repo is absent, or the downloaded run's signature/checkpoints don't match the
    current call — leaving the caller to train locally. On a match the manifest's
    checkpoint paths are rewritten to ``out_dir`` and persisted, so the standard
    local cache check passes on this and every later run.
    """
    info = _api()
    if info is None:
        return None
    api, _has_token = info
    repo_id = repo_id_for(out_dir)
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(out_dir),
                token=api.token,
            )
        except RepositoryNotFoundError:
            return None  # never trained+uploaded yet -> train locally

        # Signature gate: only reuse weights produced by this exact config.
        sig_path = out_dir / "run_signature.json"
        if not sig_path.exists() or json.loads(sig_path.read_text()) != json.loads(
            json.dumps(signature)
        ):
            print(f"  [hub] {repo_id} exists but signature differs — retraining")
            return None

        manifest_path = out_dir / "manifest.csv"
        if not manifest_path.exists():
            return None
        df = pd.read_csv(manifest_path)
        # Stored paths are relative to the producer's cwd; repoint them at the
        # local out_dir so existence checks + downstream loads resolve here.
        df["checkpoint"] = [str(out_dir / Path(c).name) for c in df["checkpoint"]]
        if not all(Path(c).exists() for c in df["checkpoint"]):
            return None
        df.to_csv(manifest_path, index=False)
        print(f"  [hub] reusing {len(df)} checkpoints from {repo_id}")
        return df
    except Exception as e:  # noqa: BLE001
        print(f"  [hub] download skipped: {type(e).__name__}: {e}")
        return None


def upload_run(out_dir: Path) -> None:
    """Mirror run directory ``out_dir`` to its public Hub repo (best effort).

    Upload needs a write token: with no ``HF_TOKEN`` the pipeline is download-only
    and this returns quietly. Otherwise creates the repo if needed and uploads the
    checkpoints + manifest + signature, skipping the transient Trainer ``_hf/``
    output. A network/permission error prints a hint and returns without raising.
    """
    info = _api()
    if info is None:
        return
    api, has_token = info
    if not has_token:
        print("  [hub] download-only (no HF_TOKEN) — skipping upload")
        return
    out_dir = Path(out_dir)
    repo_id = repo_id_for(out_dir)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(out_dir),
            # _hf/ is the live Trainer dir (optimizer state, transient); the run's
            # reusable artifacts are the checkpoint_NN dirs + manifest + signature.
            ignore_patterns=["_hf/*", "_hf", "**/__pycache__/*"],
        )
        print(f"  [hub] uploaded {out_dir.name} -> {repo_id}")
    except Exception as e:  # noqa: BLE001
        print(
            f"  [hub] upload skipped ({type(e).__name__}: {e}); "
            "set HF_TOKEN with write access to enable"
        )
