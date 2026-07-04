"""Domain-adaptive continued pre-training (DAPT) of a causal LM (step 4).

Continued next-token pre-training (Gururangan et al. 2020) of a German decoder LM
on the ``german-commons`` domain splits (``data/domain_physics`` / ``data/domain_biology``),
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
from adapters import AdapterTrainer, LoRAConfig
from adapters import init as adapters_init
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
)

from src.config import ARTIFACTS_DIR, DEFAULT_MODEL, DEVICE, DOMAIN_DIRS
from src.modeling import hub
from src.modeling.lm import ADAPTER_NAME, resize_with_mean_init

# Blocks used for the checkpoint perplexity readout. The held-out split is huge
# (val_frac of a multi-M-token corpus); a fixed subset gives a stable perplexity
# at a fraction of the eval cost. Set to None to eval on the whole test split.
EVAL_SUBSET_SIZE = 256


def _tokenize_and_chunk(ds, tokenizer, block_size, text_col="text"):
    """Tokenize a text dataset and pack into fixed-size causal-LM blocks.

    A separator token is appended per document before packing so blocks carry a
    document boundary (GPT-2's tokenizer adds none). Labels come from the
    collator (inputs with pad==eos loss-masked, so separators aren't predicted).
    """
    sep_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    def tok(batch):
        out = tokenizer(batch[text_col])
        if sep_id is not None:
            for ids in out["input_ids"]:
                ids.append(sep_id)
        return out

    ds = ds.map(
        tok, batched=True, remove_columns=ds.column_names, num_proc=4, desc="tokenize"
    )

    def group(batch):
        concat = sum(batch["input_ids"], [])
        n = (len(concat) // block_size) * block_size
        ids = [concat[i : i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": ids}

    return ds.map(
        group, batched=True, remove_columns=ds.column_names, num_proc=4, desc="pack"
    )


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

    def __init__(
        self,
        trainer,
        out_dir,
        n_checkpoints,
        words_per_epoch,
        tokens_per_step,
        manifest,
        include_baseline=True,
        checkpoint_steps=None,
        eval_samples=EVAL_SUBSET_SIZE,
    ):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.n_checkpoints = n_checkpoints
        self.words_per_epoch = words_per_epoch
        self.tokens_per_step = tokens_per_step
        self.manifest = manifest
        self.include_baseline = include_baseline
        self.checkpoint_steps = checkpoint_steps
        self._targets: dict[int, int] = {}  # global_step -> checkpoint index
        # fixed eval subset for the perplexity readout (cheaper than the full split).
        ds = trainer.eval_dataset
        self._eval_subset = (
            ds.select(range(min(eval_samples, len(ds)))) if eval_samples else ds
        )

    def on_train_begin(self, args, state, control, **kwargs):
        if self.checkpoint_steps is not None:
            # explicit step targets (indices 1..), clamped onto the final step.
            self._targets = {
                min(int(s), state.max_steps): i
                for i, s in enumerate(self.checkpoint_steps, start=1)
            }
        else:
            # evenly spaced; include_baseline puts the first at step 0.
            lo = 0 if self.include_baseline else 1
            self._targets = {
                round(i / (self.n_checkpoints - 1) * state.max_steps): i
                for i in range(lo, self.n_checkpoints)
            }
            collided = (self.n_checkpoints - lo) - len(self._targets)
            if collided:
                print(
                    f"  [warn] {collided} checkpoint step(s) collided in "
                    f"max_steps={state.max_steps}; some indices dropped "
                    "(raise max_steps / lower n_checkpoints)."
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
        # perplexity on the fixed eval subset (not the full held-out split).
        metrics = self.trainer.evaluate(eval_dataset=self._eval_subset)
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


def run_dir_for(base_model: str, domain: str, out_dir=None) -> Path:
    """Artifact directory for a DAPT run — one per base model × domain."""
    return Path(out_dir or ARTIFACTS_DIR / f"{Path(base_model).name}_{domain}_lora")


def load_cached_run(out_dir) -> pd.DataFrame | None:
    """A finished run's manifest — local ``manifest.csv``, else the Hub, else None.

    A cached run is trusted as-is. The caller (``main``) decides whether to reuse
    it or call ``finetune_dapt``; ``finetune_dapt`` itself always trains.
    """
    out_dir = Path(out_dir)
    manifest_csv = out_dir / "manifest.csv"
    if manifest_csv.exists():
        return pd.read_csv(manifest_csv)
    return hub.try_download_run(out_dir)


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
) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    LoRA continued causal pre-training (the only method), via the AdapterHub
    ``adapters`` library. Embeddings stay frozen for every arch (LoRA paradigm
    only), so differently sized models get identical adaptation capacity classes.
    Checkpoints store the adapter only; ``modeling.lm.load_causal_lm`` reattaches
    it. LoRA is injected into attention (q, k, v) and both MLP projections on
    every arch (head and embeddings excluded).

    Budgeted by tokens: ``max_tokens`` (one step = block_size·batch_size·grad_accum
    tokens); the same ``max_tokens`` equalises two domains. ``None`` = one pass.
    ``max_steps`` overrides with a fixed step count. Checkpoints saved by step
    (``n_checkpoints`` evenly spaced, or exactly ``checkpoint_steps``).
    ``max_docs`` truncates the corpus (smoke testing).

    Always trains — the skip-if-cached check lives in the caller (``main`` via
    ``load_cached_run``). Columns: ``domain``, ``checkpoint``, ``index``,
    ``epoch``, ``step``, ``tokens_seen``, ``words_seen``, ``perplexity``.
    """
    out_dir = run_dir_for(base_model, domain, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    # german-gpt2's eos/pad id sits one past its embedding rows; grow embeddings
    # (deterministic mean-init, frozen under LoRA) to avoid OOB indexing. The
    # surprisal loader applies the same resize, so train and load weights match.
    resize_with_mean_init(model, tokenizer)

    adapters_init(model)
    model.add_adapter(
        ADAPTER_NAME,
        config=LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            # attn (q, k, v) + both MLP projections on any arch; the attention output
            # projection is not adaptable in `adapters`, unlike PEFT's "all-linear".
            attn_matrices=["q", "k", "v"],
            intermediate_lora=True,
            output_lora=True,
        ),
    )
    model.train_adapter(
        ADAPTER_NAME
    )  # freeze everything but the adapter; sets it active
    print(model.adapter_summary())

    split, words_per_epoch = _prepare_splits(
        domain, tokenizer, block_size, val_frac, seed, max_docs
    )

    # Optimiser steps: from max_steps if set, else derived from the token budget
    # (max_tokens, or one full pass over the packed train blocks).
    tokens_per_step = block_size * batch_size * grad_accum
    if max_steps is None:
        budget = (
            max_tokens if max_tokens is not None else len(split["train"]) * block_size
        )
        max_steps = max(1, math.ceil(budget / tokens_per_step))

    # bf16 on Ampere+, else fp16; tf32 matmuls free on Ampere+.
    use_cuda = DEVICE == "cuda"
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
        # no automatic epoch evals — the checkpoint callback does the only eval we
        # need (perplexity on a fixed subset), avoiding redundant full-split passes.
        eval_strategy="no",
        save_strategy="no",  # checkpointing handled by the callback
        logging_steps=50,
        disable_tqdm=True,  # _StepProgress replaces the built-in bar
        report_to=[],
    )
    # AdapterTrainer: save_model writes the active adapter only, not full weights.
    trainer = AdapterTrainer(
        model=model,
        args=args,
        # mlm=False: labels = inputs with pad(=eos) loss-masked
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,  # saved into each checkpoint dir
        callbacks=[_StepProgress(domain)],
    )

    manifest: list[dict] = []
    trainer.add_callback(
        _CheckpointSchedule(
            trainer,
            out_dir,
            n_checkpoints,
            words_per_epoch,
            tokens_per_step,
            manifest,
            include_baseline=include_baseline,
            checkpoint_steps=checkpoint_steps,
        )
    )
    trainer.train()

    df = pd.DataFrame(manifest)
    df.insert(0, "domain", domain)
    # Persist manifest so the next run resumes from cache instead of retraining.
    df.to_csv(out_dir / "manifest.csv", index=False)
    # Mirror to the Hub for future runs. Best effort — never fails the run.
    hub.upload_run(out_dir)
    return df
