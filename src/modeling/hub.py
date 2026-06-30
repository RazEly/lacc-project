"""Hugging Face Hub mirror for DAPT checkpoint runs.

Mirrors a whole ``finetune_dapt`` run dir (checkpoint_NN/ + manifest.csv +
run_signature.json) to/from one public Hub repo, so a GPU run is reused across
machines. Resolution: local cache -> Hub -> train; a downloaded run passes the
same signature check as a local one. Best-effort — offline / missing repo returns
``None`` (download) or skips (upload), never raising.

Repos live under a fixed namespace (anyone downloads tokenless); upload needs a
write token, so only the owner pushes. Env vars:
* ``HF_TOKEN`` — write token; present enables upload, absent = download-only.
* ``HF_AUTO_CHECKPOINTS=0`` — disable Hub download + upload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Fixed owner of the public checkpoint repos. Repo ids:
# <HF_NAMESPACE>/<prefix><run-dir>, e.g. ElyR120/potec-dapt-german-gpt2_physics_lora.
HF_NAMESPACE = "ElyR120"
REPO_PREFIX = "potec-dapt-"


def _enabled() -> bool:
    """Hub mirroring is on unless explicitly disabled."""
    return os.environ.get("HF_AUTO_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "")


def _api():
    """``(HfApi, has_token)`` or ``None`` if disabled. ``has_token`` gates upload."""
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

    ``None`` when disabled, repo absent, or signature/checkpoints mismatch (caller
    trains locally). On a match the manifest paths are rewritten to ``out_dir``.
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
        # repoint stored paths (producer's cwd) at the local out_dir.
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

    No-op without a write ``HF_TOKEN``. Overwrites the repo to exactly this run
    (``delete_patterns`` prunes stale remote files in the same commit).
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
            ignore_patterns=["_hf/*", "_hf", "**/__pycache__/*"],  # skip transient Trainer dir
            # delete remote files absent from this run (upsert wins within the commit).
            delete_patterns=["*"],
        )
        print(f"  [hub] uploaded {out_dir.name} -> {repo_id}")
    except Exception as e:  # noqa: BLE001
        print(
            f"  [hub] upload skipped ({type(e).__name__}: {e}); "
            "set HF_TOKEN with write access to enable"
        )
