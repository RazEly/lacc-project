"""Load decoder LMs (base checkpoints or LoRA adapters) for scoring and DAPT.

Shared LM infrastructure: tokenizer loading, the deterministic embedding resize,
and the surprisal-ready model loader. Lives in ``modeling`` so both training
(``finetune``) and scoring (``features.surprisal``) draw from one place and the
dependency arrow points features -> modeling only.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEFAULT_MODEL, DEVICE


def _load_tokenizer(name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=True)
    except (TypeError, ValueError):
        return AutoTokenizer.from_pretrained(name_or_path)


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
    return True


def load_causal_lm(name_or_path: str = DEFAULT_MODEL):
    """Load a causal LM + tokenizer for surprisal.

    Accepts an HF model id, a full fine-tuned checkpoint, or a LoRA (PEFT) adapter
    checkpoint — the latter is detected by ``adapter_config.json`` and folded onto
    its base model so callers get a plain merged model either way.
    """
    kwargs = {}
    # fp16 on GPU: halves VRAM, inference-safe (log_softmax upcasts via .float()).
    if DEVICE == "cuda":
        kwargs["dtype"] = torch.float16

    if (Path(name_or_path) / "adapter_config.json").is_file():
        # LoRA checkpoint: load the adapter's base model, match embedding size,
        # then merge the adapter so the forward needs no PEFT machinery.

        base = PeftConfig.from_pretrained(name_or_path).base_model_name_or_path
        tokenizer = _load_tokenizer(base)
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        resize_with_mean_init(model, tokenizer)
        model = PeftModel.from_pretrained(model, name_or_path).merge_and_unload()
    else:
        tokenizer = _load_tokenizer(name_or_path)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        resize_with_mean_init(model, tokenizer)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer
