"""Push existing DAPT run dirs in artifacts/ to the Hugging Face Hub.

Mirrors each completed run directory (the *_lora dirs holding checkpoint_NN/ +
manifest.csv + run_signature.json) to its canonical public Hub repo via the same
hub.upload_run path the pipeline uses after a fresh train. Use this to publish
runs that were trained before Hub mirroring was wired up, without retraining.

Needs a write HF_TOKEN for HF_NAMESPACE (owner only):

    HF_TOKEN=hf_xxx python scripts/push_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.modeling import hub  # noqa: E402 — needs ROOT on sys.path first

ARTIFACTS = ROOT / "artifacts"


def _run_dirs() -> list[Path]:
    """Completed run dirs: subdirs of artifacts/ carrying a run_signature.json."""
    return sorted(p for p in ARTIFACTS.iterdir() if (p / "run_signature.json").exists())


def main() -> None:
    runs = _run_dirs()
    if not runs:
        print(f"no run dirs under {ARTIFACTS}")
        return
    print(f"pushing {len(runs)} run(s) to {hub.HF_NAMESPACE}:")
    for d in runs:
        print(f"  -> {hub.repo_id_for(d)}")
        hub.upload_run(d)


if __name__ == "__main__":
    main()
