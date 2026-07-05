"""Load decoder LMs (base checkpoints or LoRA adapters) for scoring and DAPT.

Shared LM infrastructure: tokenizer loading, the deterministic embedding resize,
and the surprisal-ready model loader. Lives in ``modeling`` so both training
(``finetune``) and scoring (``features.surprisal``) draw from one place and the
dependency arrow points features -> modeling only.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from src.config import DEFAULT_MODEL, DEVICE

# Name of the single LoRA adapter trained by ``finetune``; also the checkpoint
# subdirectory the ``adapters`` library saves it under.
ADAPTER_NAME = "dapt"


def load_tokenizer(name_or_path: str = DEFAULT_MODEL):
    """The scoring tokenizer (``add_prefix_space`` matches word-level surprisal)."""
    return AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=True)


def resize_with_mean_init(model, tokenizer) -> bool:
    """Grow embeddings to cover every tokenizer id; new rows = mean embedding.

    german-gpt2's eos/pad id sits one past its embedding rows, so training (which
    appends eos as a document separator) needs the resize. The default resize
    initialises new rows randomly — nondeterministic across loads and never
    trained under LoRA (embeddings stay frozen); mean-init is deterministic and a
    sane prior. Returns whether a resize happened.
    """
    if len(tokenizer) <= model.config.vocab_size:
        return False
    old_n = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[old_n:] = emb[:old_n].mean(dim=0)
        out = model.get_output_embeddings()
        if out is not None and out.weight.data_ptr() != emb.data_ptr():
            out.weight[old_n:] = out.weight[:old_n].mean(dim=0)
    # `adapters` flex heads keep their own vocab_size in the head config dict;
    # resize_token_embeddings grows the weights but not that entry, and the
    # head's loss reshape reads it — sync it or training crashes.
    for head in getattr(model, "heads", {}).values():
        if "vocab_size" in head.config:
            head.config["vocab_size"] = len(tokenizer)
    return True


def load_causal_lm(name_or_path: str = DEFAULT_MODEL):
    """Load a causal LM + tokenizer for surprisal.

    Accepts an HF model id, a full fine-tuned checkpoint, or a LoRA adapter
    checkpoint (``adapters`` library) — the latter is detected by its
    ``ADAPTER_NAME`` subdirectory and merged onto its base model so callers get
    a plain model either way.
    """
    adapter_dir = Path(name_or_path) / ADAPTER_NAME
    if (adapter_dir / "adapter_config.json").is_file():
        # LoRA checkpoint: load the adapter's base model, match embedding size,
        # then merge the adapter so the forward carries no LoRA overhead.
        base = json.loads((adapter_dir / "adapter_config.json").read_text())[
            "model_name"
        ]
        tokenizer = load_tokenizer(base)
        model = AutoAdapterModel.from_pretrained(base)
        resize_with_mean_init(model, tokenizer)
        model.load_adapter(str(adapter_dir), set_active=True)
        model.merge_adapter(ADAPTER_NAME)
    else:
        tokenizer = load_tokenizer(name_or_path)
        model = AutoAdapterModel.from_pretrained(name_or_path)
        resize_with_mean_init(model, tokenizer)
    model.eval()
    # fp16 on GPU: halves VRAM, inference-safe (log_softmax upcasts via .float()).
    # Cast only after the adapter merge: `adapters` builds its LoRA modules in
    # fp32 regardless of the base model's dtype, so an fp16 base + adapter load
    # leaves mixed Half/Float weights and the forward crashes.
    if DEVICE == "cuda":
        model.to(dtype=torch.float16)
    model.to(DEVICE)
    return model, tokenizer
