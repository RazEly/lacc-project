"""Slim bloated german-gpt2 DAPT checkpoints in place (adapter_model.safetensors).

german-gpt2 grows its token embedding for DAPT, so PEFT saves ``wte`` via
``modules_to_save``. save_pretrained then serialises the [50266,768] embedding
FOUR times in fp32 per checkpoint (~636 MB):

    wte.modules_to_save.weight   trained rows            KEEP (the DAPT signal)
    wte.original_module.weight   frozen base copy        drop (== base model)
    wte.weight                   active-forward dup      drop (== modules_to_save)
    lm_head.weight               tied to wte             drop (recovered on load)

We keep the LoRA deltas + the single trained embedding, cast every kept float
tensor to fp16 (load_causal_lm already runs fp16 inference, so lossless there),
and drop the three redundant copies. ~636 MB -> ~80 MB per checkpoint.

LLaMmlein is not resized (modules_to_save=None) -> already adapter-only; skip it.

    uv run python scripts/slim_german_checkpoints.py verify <run_dir>   # one ckpt, compare surprisal
    uv run python scripts/slim_german_checkpoints.py apply  <run_dir>   # rewrite every ckpt in place
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Redundant embedding copies to drop (keys end with these).
DROP_SUFFIXES = (
    "wte.original_module.weight",  # frozen base — provided by base model on load
    "lm_head.weight",              # tied to wte — rematerialised on load
    "transformer.wte.weight",      # active-forward dup of modules_to_save
)


def _keep(key: str) -> bool:
    return not any(key.endswith(s) for s in DROP_SUFFIXES)


def slim_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop redundant embedding copies; cast remaining float tensors to fp16."""
    out = {}
    for k, v in sd.items():
        if not _keep(k):
            continue
        out[k] = v.to(torch.float16) if v.is_floating_point() else v
    return out


def slim_file(src: Path, dst: Path) -> tuple[int, int]:
    """Rewrite one adapter_model.safetensors; return (src_bytes, dst_bytes)."""
    sd = load_file(str(src))
    slim = slim_state_dict(sd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # PEFT reads the format metadata; safetensors needs it to round-trip.
    save_file(slim, str(dst), metadata={"format": "pt"})
    return src.stat().st_size, dst.stat().st_size


def _checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("checkpoint_*"))


def _surprisal_sample(ckpt: Path, n_words: int = 200):
    """Load the checkpoint via the real inference path and return a surprisal vector."""
    from src.features import data, surprisal as su

    words = data.load_word_features().head(n_words)
    model, tok = su.load_causal_lm(str(ckpt))
    s = su.compute_surprisal(words, model, tok)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return s["surprisal"].to_numpy()


def verify(run_dir: Path) -> None:
    """Slim checkpoint_00 into a temp dir, compare surprisal against the original."""
    ck = _checkpoints(run_dir)[0]
    src = ck / "adapter_model.safetensors"
    print(f"verify: {ck}")

    base = _surprisal_sample(ck)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / ck.name
        shutil.copytree(ck, tmp)  # tokenizer/config alongside the adapter
        sb, db = slim_file(src, tmp / "adapter_model.safetensors")
        print(f"  size {sb/1e6:.0f} MB -> {db/1e6:.0f} MB  ({sb/db:.1f}x smaller)")
        slim = _surprisal_sample(tmp)

    import numpy as np

    diff = np.abs(base - slim)
    ok = np.allclose(base, slim, atol=1e-3, rtol=1e-3, equal_nan=True)
    print(f"  surprisal max|Δ|={np.nanmax(diff):.2e} mean|Δ|={np.nanmean(diff):.2e}")
    print("  RESULT:", "MATCH — safe to apply" if ok else "MISMATCH — do NOT apply")


def apply(run_dir: Path) -> None:
    """Rewrite every checkpoint's adapter file in place (originals live on the Hub)."""
    total_src = total_dst = 0
    for ck in _checkpoints(run_dir):
        f = ck / "adapter_model.safetensors"
        if not f.is_file():
            print(f"  [skip] {ck.name}: no adapter file")
            continue
        sb, db = slim_file(f, f)  # overwrite in place
        total_src += sb
        total_dst += db
        print(f"  {ck.name}: {sb/1e6:.0f} MB -> {db/1e6:.0f} MB")
    print(f"run total {total_src/1e6:.0f} MB -> {total_dst/1e6:.0f} MB")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("verify", "apply"):
        print(__doc__)
        sys.exit(2)
    mode, run_dir = sys.argv[1], Path(sys.argv[2])
    if not run_dir.is_dir():
        print(f"no such run dir: {run_dir}")
        sys.exit(1)
    (verify if mode == "verify" else apply)(run_dir)


if __name__ == "__main__":
    main()
