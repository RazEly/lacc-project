"""Domain-adaptive continued pre-training (DAPT) of a causal LM (step 4).

Continued next-token pre-training (Gururangan et al. 2020) of a German decoder LM
on the ``german-commons`` domain splits (``data/domain_phy`` / ``data/domain_bio``),
disjoint from the PoTeC stimuli (no leakage). Physics and biology train
independently on a shared *token* budget (``max_tokens``), so both see the same
tokens at the same checkpoint index despite different corpus sizes. Saves
checkpoints by training step with validation perplexity, for a progress curve.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_MODEL,
    DOMAIN_BIO_DIR,
    DOMAIN_PHY_DIR,
)
from src.features.surprisal import compute_surprisal, load_causal_lm

DOMAIN_DIRS = {"physics": DOMAIN_PHY_DIR, "biology": DOMAIN_BIO_DIR}


def _slim_adapter(ckpt: Path) -> None:
    """Drop the redundant embedding copies PEFT writes for ``modules_to_save``.

    Resizing the vocab forces ``modules_to_save=["wte"]``; PEFT then serialises the
    full embedding four times in fp32 (top-level ``wte.weight``, ``original_module``,
    ``modules_to_save``, plus the tied ``lm_head``) — ~4x the real signal for a
    ~600MB adapter. Only the top-level ``wte.weight`` is read back at load; the
    ``original_module``/``lm_head`` copies equal the frozen base and reconstruct from
    it. Keep the LoRA deltas + trained top-level embedding, drop the rest. No-op for
    pure-LoRA adapters (no ``modules_to_save``).
    """
    from safetensors.torch import load_file, save_file

    path = ckpt / "adapter_model.safetensors"
    sd = load_file(str(path))
    if not any(k.endswith(".modules_to_save.weight") for k in sd):
        return  # pure-LoRA adapter, nothing to slim
    keep = {
        k: v
        for k, v in sd.items()
        if not (
            k.endswith(".modules_to_save.weight")
            or k.endswith(".original_module.weight")
            or k == "base_model.model.lm_head.weight"
        )
    }
    save_file(keep, str(path), metadata={"format": "pt"})


def _lora_targets(model, override):
    """Per-architecture LoRA target + embedding modules for DAPT.

    Targets MLP as well as attention (PEFT's attention-only default is too thin for
    domain adaptation). Embeddings returned separately for ``modules_to_save`` when
    grown. ``override`` short-circuits the targets (embeddings stay per-arch).
    """
    mt = model.config.model_type
    if mt == "gpt2":
        targets = ["c_attn", "c_fc", "c_proj"]  # attn qkv/out + mlp in/out
        embed = ["wte"]                          # lm_head is tied to wte
    elif mt == "llama":
        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
        embed = ["embed_tokens", "lm_head"]
    else:  # unknown arch: fall back to PEFT's per-arch defaults
        targets, embed = None, None
    return (override or targets), embed


def _tokenize_and_chunk(ds, tokenizer, block_size, text_col="text"):
    """Tokenize a text dataset and pack into fixed-size causal-LM blocks.

    A separator token is appended per document before packing so blocks carry a
    document boundary (GPT-2's tokenizer adds none). ``labels`` = inputs.
    """
    sep_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    def tok(batch):
        out = tokenizer(batch[text_col])
        if sep_id is not None:
            for ids in out["input_ids"]:
                ids.append(sep_id)
        return out

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

    def group(batch):
        concat = sum(batch["input_ids"], [])
        n = (len(concat) // block_size) * block_size
        ids = [concat[i : i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}

    return ds.map(group, batched=True, remove_columns=ds.column_names)


def _prepare_splits(domain, tokenizer, block_size, val_frac, seed, max_docs):
    """Load a domain corpus and build packed train/test LM blocks.

    Returns the split dict plus ``words_per_epoch`` (whitespace words in the
    train docs, for the legacy words-seen column).
    """
    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    if max_docs:
        raw = raw.select(range(min(max_docs, len(raw))))

    # Split at document level before packing (block-level split leaks adjacent
    # text into val and inflates eval perplexity).
    raw_split = raw.train_test_split(test_size=val_frac, seed=seed)

    # Legacy words-seen x-axis: train docs only.
    words_per_epoch = sum(len(str(t).split()) for t in raw_split["train"]["text"])

    split = {
        "train": _tokenize_and_chunk(raw_split["train"], tokenizer, block_size),
        "test": _tokenize_and_chunk(raw_split["test"], tokenizer, block_size),
    }
    return split, words_per_epoch


class _CheckpointSchedule(TrainerCallback):
    """Save checkpoints by training step.

    Default: ``n_checkpoints`` evenly spaced over ``max_steps``. ``checkpoint_steps``
    saves at exactly those steps (e.g. Škrjanec et al.'s 4ⁿ schedule).
    ``include_baseline`` makes the step-0 model index 0; listed steps follow as 1, 2, …
    """

    def __init__(self, trainer, out_dir, n_checkpoints, words_per_epoch,
                 tokens_per_step, manifest, include_baseline=True,
                 checkpoint_steps=None):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.n_checkpoints = n_checkpoints
        self.words_per_epoch = words_per_epoch
        self.tokens_per_step = tokens_per_step
        self.manifest = manifest
        self.include_baseline = include_baseline
        self.checkpoint_steps = checkpoint_steps
        self._targets: dict[int, int] = {}  # global_step -> checkpoint index

    def on_train_begin(self, args, state, control, **kwargs):
        if self.checkpoint_steps is not None:
            # explicit step targets (indices 1..), clamped onto the final step.
            steps = {min(int(s), state.max_steps): i
                     for i, s in enumerate(self.checkpoint_steps, start=1)}
            self._targets = steps
        else:
            # evenly spaced; include_baseline puts the first at step 0.
            lo = 0 if self.include_baseline else 1
            steps = {
                round(i / (self.n_checkpoints - 1) * state.max_steps): i
                for i in range(lo, self.n_checkpoints)
            }
            self._targets = steps
            wanted = self.n_checkpoints - lo
            if len(steps) < wanted:
                print(
                    f"  [warn] {wanted} checkpoints requested but only {len(steps)} "
                    f"distinct steps fit in max_steps={state.max_steps}; some "
                    "collided (raise max_steps / lower n_checkpoints)."
                )
        if self.include_baseline:
            self._save(state, 0)  # baseline: weights still un-fine-tuned

    def on_step_end(self, args, state, control, **kwargs):
        idx = self._targets.get(state.global_step)
        if idx is not None:
            self._save(state, idx)

    def _save(self, state, idx):
        ckpt = self.out_dir / f"checkpoint_{idx:02d}"
        self.trainer.save_model(str(ckpt))
        _slim_adapter(ckpt)  # strip PEFT's redundant fp32 embedding copies
        metrics = self.trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"])
        # tokens_seen: cross-domain-comparable x-axis (depends only on global_step).
        tokens_seen = state.global_step * self.tokens_per_step
        words_seen = round(state.epoch * self.words_per_epoch)
        self.manifest.append(
            {
                "checkpoint": str(ckpt),
                "index": idx,
                "epoch": round(state.epoch, 3),
                "step": state.global_step,
                "tokens_seen": tokens_seen,
                "words_seen": words_seen,
                "perplexity": ppl,
            }
        )
        print(
            f"  [{ckpt.name}] step={state.global_step} tokens_seen={tokens_seen:,} "
            f"epoch={state.epoch:.2f} perplexity={ppl:.2f}"
        )


class _StepProgress(TrainerCallback):
    """tqdm bar over training steps (the run is budgeted by tokens, not epochs)."""

    def __init__(self, domain):
        self.domain = domain
        self.bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.bar = tqdm(total=state.max_steps, desc=f"DAPT {self.domain}", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        self.bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.bar.close()


def _load_cached_manifest(out_dir: Path) -> pd.DataFrame | None:
    """Return the saved manifest if the run dir has one — trusts its contents."""
    manifest_path = out_dir / "manifest.csv"
    return pd.read_csv(manifest_path) if manifest_path.exists() else None


def finetune_dapt(
    domain: str,
    base_model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
    max_steps: int | None = None,
    checkpoint_steps=None,
    n_checkpoints: int = 10,
    include_baseline: bool = True,
    block_size: int = 512,
    batch_size: int = 64,
    grad_accum: int = 1,
    val_frac: float = 0.05,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    seed: int = 0,
    out_dir=None,
    max_docs=None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules=None,
) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    LoRA continued causal pre-training (the only method). Checkpoints store the
    adapter only; ``surprisal.load_causal_lm`` reattaches it. ``lora_target_modules``
    defaults per-arch via ``_lora_targets`` (attention + MLP).

    Budgeted by tokens: ``max_tokens`` (one step = block_size·batch_size·grad_accum
    tokens); the same ``max_tokens`` equalises two domains. ``None`` = one pass.
    ``max_steps`` overrides with a fixed step count. Checkpoints saved by step
    (``n_checkpoints`` evenly spaced, or exactly ``checkpoint_steps``).
    ``max_docs`` truncates the corpus (smoke testing).

    Resumable: an existing cached run (local or Hub) is reused instead of
    retraining. Columns: ``domain``, ``checkpoint``, ``index``,
    ``epoch``, ``step``, ``tokens_seen``, ``words_seen``, ``perplexity``.
    """
    out_dir = Path(
        out_dir or CHECKPOINTS_DIR / f"{Path(base_model).name}_{domain}_lora"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from a cached run if the dir already holds one (trusted as-is).
    cached = _load_cached_manifest(out_dir)
    if cached is not None:
        print(f"  [{domain}] reusing {len(cached)} saved checkpoints in {out_dir} "
              "(skipping training)")
        return cached

    # Local miss: try the Hub before paying for a GPU run.
    from src.modeling import hub

    remote = hub.try_download_run(out_dir)
    if remote is not None:
        print(f"  [{domain}] reusing {len(remote)} checkpoints pulled from the Hub "
              "(skipping training)")
        return remote

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    # german-gpt2's eos/pad id sits one past its embedding rows; grow embeddings to
    # cover every tokenizer id (the new rows train during DAPT) to avoid OOB indexing.
    resized = len(tokenizer) > model.config.vocab_size
    if resized:
        model.resize_token_embeddings(len(tokenizer))

    from peft import LoraConfig, TaskType, get_peft_model

    targets, embed_modules = _lora_targets(model, lora_target_modules)
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,  # attn + MLP (None -> PEFT per-arch default)
        # save grown embeddings: LoRA freezes the base, so the new rows would be
        # re-randomised at load time otherwise.
        modules_to_save=embed_modules if resized else None,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    split, words_per_epoch = _prepare_splits(
        domain, tokenizer, block_size, val_frac, seed, max_docs
    )

    # Optimiser steps: from max_steps if set, else derived from the token budget.
    tokens_per_step = block_size * batch_size * grad_accum
    available_tokens = len(split["train"]) * block_size
    if max_steps is None:
        budget = max_tokens if max_tokens is not None else available_tokens
        max_steps = max(1, math.ceil(budget / tokens_per_step))

    # bf16 on Ampere+, else fp16; tf32 matmuls free on Ampere+.
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir / "_hf"),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        # low LR + short warmup: avoid catastrophic forgetting from the base weights.
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        seed=seed,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        tf32=use_cuda,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="no",  # checkpointing handled by the callback
        logging_steps=50,
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    manifest: list[dict] = []
    trainer.add_callback(
        _CheckpointSchedule(
            trainer, out_dir, n_checkpoints, words_per_epoch, tokens_per_step,
            manifest, include_baseline=include_baseline,
            checkpoint_steps=checkpoint_steps,
        )
    )
    trainer.add_callback(_StepProgress(domain))
    trainer.train()

    df = pd.DataFrame(manifest)
    df.insert(0, "domain", domain)
    # Persist manifest for reuse (see _load_cached_manifest).
    df.to_csv(out_dir / "manifest.csv", index=False)
    # Mirror to the Hub for future runs. Best effort — never fails the run.
    hub.upload_run(out_dir)
    return df


def recompute_surprisal_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame, prompt=None
) -> pd.DataFrame:
    """Recompute step-2 surprisal with each fine-tuned checkpoint.

    Concatenated surprisal table tagged with ``checkpoint`` / ``index`` / ``epoch``
    / ``domain``. ``index`` (0 = baseline) is the stable key for pairing domains.
    """
    frames = []
    # iterrows: the manifest has a column named "index" that itertuples would rename.
    for _, row in manifest.iterrows():
        model, tok = load_causal_lm(row.checkpoint)
        sup = compute_surprisal(words_df, model, tok, prompt=prompt)
        sup["checkpoint"] = row.checkpoint
        sup["index"] = row["index"]  # checkpoint index: stable across domains
        sup["epoch"] = row.epoch
        sup["domain"] = row.domain
        frames.append(sup)
    return pd.concat(frames, ignore_index=True)
