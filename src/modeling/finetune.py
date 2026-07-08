"""Domain-adaptive continued pre-training (DAPT) of a causal LM (step 4).

Continued next-token pre-training (Gururangan et al. 2020) of a German decoder LM
on the term-targeted Wikipedia domain corpora (``data/wiki_physics`` /
``data/wiki_biology``), disjoint from the PoTeC stimuli (no leakage). Physics and
biology train independently for the same number of steps, so both see the same
tokens at the same checkpoint index despite different corpus sizes. Saves
checkpoints by training step with validation perplexity, for a progress curve.

Run as a module to train every model × domain and save the LoRA adapters under
``artifacts/`` (the default ``adapters``-library checkpoint layout, reloaded by
``modeling.lm.load_causal_lm``)::

    python -m src.modeling.finetune
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from adapters import AdapterTrainer, AutoAdapterModel, LoRAConfig
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
)

from src.config import (
    ARTIFACTS_DIR,
    DAPT_CHECKPOINT_STEPS,
    DEFAULT_MODEL,
    DEVICE,
    DOMAIN_DIRS,
    DOMAINS,
    MODELS,
)
from src.modeling.lm import ADAPTER_NAME, resize_with_mean_init

# Blocks used for the checkpoint perplexity readout. The held-out split is huge
# (val_frac of a multi-M-token corpus); a fixed subset gives a stable perplexity
# at a fraction of the eval cost. Set to None to eval on the whole test split.
EVAL_SUBSET_SIZE = 256

# ---------------------------------------------------------------------------
# Fine-tuning hyperparameters (every knob `train_all` uses)
# ---------------------------------------------------------------------------
# MODELS + DAPT_CHECKPOINT_STEPS are shared across modules -> config.
BLOCK_SIZE = 512  # causal-LM block length (tokens)
VAL_FRAC = 0.05  # doc-level held-out fraction for the perplexity eval
WARMUP_RATIO = 0.05  # short warmup + low LR: limit catastrophic forgetting
SEED = 0
LORA_R = 16  # LoRA rank / alpha / dropout — same adaptation capacity every arch
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# --- per model (german-gpt2 vs llammlein-1b differ here) ---
# Effective batch = BATCH_SIZE × GRAD_ACCUM MUST match across models (= 8, i.e.
# 4096 tokens/step at BLOCK_SIZE=512) so a checkpoint step is the same #tokens.
BATCH_SIZE = {
    "german-gpt2": 8,
    "llammlein-1b": 4,
}  # per-device train batch (~16 GB VRAM)
GRAD_ACCUM = {"german-gpt2": 1, "llammlein-1b": 2}  # 8×1 == 2×4 effective batch
LEARNING_RATE = {"german-gpt2": 1e-4, "llammlein-1b": 1e-4}  # larger model, smaller LR


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
    """Load a domain corpus and build packed train/test LM blocks."""
    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    if max_docs:
        raw = raw.select(range(min(max_docs, len(raw))))

    # Split at document level before packing (block-level split leaks adjacent
    # text into val and inflates eval perplexity).
    raw_split = raw.train_test_split(test_size=val_frac, seed=seed)

    return {
        "train": _tokenize_and_chunk(raw_split["train"], tokenizer, block_size),
        "test": _tokenize_and_chunk(raw_split["test"], tokenizer, block_size),
    }


def _prepare_eval_split(domain, tokenizer, block_size, val_frac, seed, max_docs=None):
    """Packed test blocks of ``domain``'s held-out split only (no train tokenize).

    Same doc-level ``train_test_split(val_frac, seed)`` as ``_prepare_splits``,
    so the blocks are exactly the held-out set that domain's own DAPT run evals
    on — used for the cross-domain perplexity readout (fig 1 dotted lines).
    """
    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    if max_docs:
        raw = raw.select(range(min(max_docs, len(raw))))
    raw_split = raw.train_test_split(test_size=val_frac, seed=seed)
    return _tokenize_and_chunk(raw_split["test"], tokenizer, block_size)


class _CheckpointSchedule(TrainerCallback):
    """Save the adapter at ``checkpoint_steps`` + record a per-checkpoint perplexity manifest.

    Flags each target step with ``control.should_save`` so the ``AdapterTrainer``
    saves the adapter its own default way (``<out_dir>/checkpoint-<step>/<adapter>``)
    and logs that checkpoint's perplexity. Index 0 is the un-fine-tuned base model
    (LoRA delta is zero at init, so no adapter is saved) — its manifest row points
    at ``base_model`` and ``load_causal_lm`` scores it as the plain model.
    """

    def __init__(
        self,
        trainer,
        out_dir,
        base_model,
        checkpoint_steps,
        tokens_per_step,
        manifest,
        eval_samples=EVAL_SUBSET_SIZE,
        cross_eval=None,
    ):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.base_model = base_model
        self.tokens_per_step = tokens_per_step
        self.manifest = manifest
        # step targets (indices 1..), clamped onto the final step.
        self._index = {
            min(int(s), trainer.args.max_steps): i
            for i, s in enumerate(checkpoint_steps, start=1)
        }

        def _subset(ds):
            return ds.select(range(min(eval_samples, len(ds)))) if eval_samples else ds

        # fixed eval subset for the perplexity readout (cheaper than the full split).
        self._eval_subset = _subset(trainer.eval_dataset)
        # out-of-domain held-out sets ({domain: packed test blocks}) scored at each
        # checkpoint into ``perplexity_<domain>`` manifest columns (fig 1 dotted).
        self._cross_eval = {d: _subset(ds) for d, ds in (cross_eval or {}).items()}

    def on_train_begin(self, args, state, control, **kwargs):
        # index 0: the base model, scored before any training step.
        self._log(state, step=0, idx=0, checkpoint=self.base_model)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self._index:
            control.should_save = True  # AdapterTrainer saves the adapter (default)
            self._log(
                state, step, self._index[step], self.out_dir / f"checkpoint-{step}"
            )
        return control

    def _log(self, state, step, idx, checkpoint):
        # perplexity on the fixed eval subset (not the full held-out split).
        metrics = self.trainer.evaluate(eval_dataset=self._eval_subset)
        ppl = math.exp(metrics["eval_loss"])
        # tokens_seen: cross-domain-comparable x-axis (depends only on global_step).
        tokens_seen = step * self.tokens_per_step
        row = {
            "checkpoint": str(checkpoint),
            "index": idx,
            "epoch": round(state.epoch, 3),
            "step": step,
            "tokens_seen": tokens_seen,
            "perplexity": ppl,
        }
        for dom, ds in self._cross_eval.items():
            m = self.trainer.evaluate(eval_dataset=ds)
            row[f"perplexity_{dom}"] = math.exp(m["eval_loss"])
        self.manifest.append(row)
        extra = " ".join(
            f"perplexity_{d}={row[f'perplexity_{d}']:.2f}" for d in self._cross_eval
        )
        print(
            f"  [idx {idx:02d}] step={step} tokens_seen={tokens_seen:,} "
            f"epoch={state.epoch:.2f} perplexity={ppl:.2f} {extra}".rstrip()
        )


def run_dir_for(base_model: str, domain: str, out_dir=None) -> Path:
    """Artifact directory for a DAPT run — one per base model × domain."""
    return Path(out_dir or ARTIFACTS_DIR / f"{Path(base_model).name}_{domain}_lora")


def load_cached_run(out_dir) -> pd.DataFrame | None:
    """A finished run's manifest (``manifest.csv``) if present, else None.

    A cached run is trusted as-is. The caller decides whether to reuse it or
    train; ``finetune_dapt`` itself always trains.
    """
    manifest_csv = Path(out_dir) / "manifest.csv"
    if manifest_csv.exists():
        return pd.read_csv(manifest_csv)
    return None


def finetune_dapt(
    domain: str,
    base_model: str = DEFAULT_MODEL,
    max_steps: int | None = DAPT_CHECKPOINT_STEPS[-1],
    checkpoint_steps=DAPT_CHECKPOINT_STEPS,
    block_size: int = BLOCK_SIZE,
    batch_size: int = 8,
    grad_accum: int = 1,
    val_frac: float = VAL_FRAC,
    learning_rate: float = 2e-4,
    warmup_ratio: float = WARMUP_RATIO,
    seed: int = SEED,
    out_dir=None,
    max_docs=None,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    LoRA continued causal pre-training (the only method), via the AdapterHub
    ``adapters`` library. Embeddings stay frozen for every arch (LoRA paradigm
    only), so differently sized models get identical adaptation capacity classes.
    Checkpoints store the adapter only; ``modeling.lm.load_causal_lm`` reattaches
    it. LoRA is injected into attention (q, k, v) and both MLP projections on
    every arch (head and embeddings excluded).

    Runs ``max_steps`` optimiser steps (one step = block_size·batch_size·grad_accum
    tokens); the shared step count equalises two domains. Adapters are saved at
    exactly ``checkpoint_steps`` (indices 1..); index 0 is the un-fine-tuned base
    model. ``max_docs`` truncates the corpus (smoke testing).

    Always trains — the skip-if-cached check lives in the caller (``train_all``
    via ``load_cached_run``). Columns: ``domain``, ``checkpoint``, ``index``,
    ``epoch``, ``step``, ``tokens_seen``, ``perplexity`` (own
    held-out split), plus ``perplexity_<domain>`` per other domain (that
    domain's held-out split — the fig 1 out-of-domain dotted line).
    """
    out_dir = run_dir_for(base_model, domain, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoAdapterModel.from_pretrained(base_model)

    # german-gpt2's eos/pad id sits one past its embedding rows; grow embeddings
    # (deterministic mean-init, frozen under LoRA) to avoid OOB indexing. The
    # surprisal loader applies the same resize, so train and load weights match.
    resize_with_mean_init(model, tokenizer)

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
    model.train_adapter(ADAPTER_NAME)
    print(model.adapter_summary())

    split = _prepare_splits(domain, tokenizer, block_size, val_frac, seed, max_docs)
    # Other domains' held-out splits, scored at every checkpoint for the
    # out-of-domain perplexity curve (manifest ``perplexity_<domain>``, fig 1).
    cross_eval = {
        d: _prepare_eval_split(d, tokenizer, block_size, val_frac, seed, max_docs)
        for d in DOMAINS
        if d != domain
    }

    tokens_per_step = block_size * batch_size * grad_accum
    use_cuda = DEVICE == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        seed=seed,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        tf32=use_cuda,
        dataloader_num_workers=4,
        eval_strategy="no",
        save_strategy="no",
        save_only_model=True,
        logging_steps=50,
        report_to=[],
    )
    trainer = AdapterTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    manifest: list[dict] = []
    trainer.add_callback(
        _CheckpointSchedule(
            trainer,
            out_dir,
            base_model,
            checkpoint_steps,
            tokens_per_step,
            manifest,
            cross_eval=cross_eval,
        )
    )
    trainer.train()

    df = pd.DataFrame(manifest)
    df.insert(0, "domain", domain)
    df.to_csv(out_dir / "manifest.csv", index=False)
    return df


def train_all() -> None:
    """DAPT every model × domain; save LoRA checkpoints under ``artifacts/``.

    Shared knobs come from ``finetune_dapt``'s defaults (the module globals); the
    per-model batch/grad-accum/LR are passed from the ``BATCH_SIZE`` /
    ``GRAD_ACCUM`` / ``LEARNING_RATE`` tables. Skips a run whose ``manifest.csv``
    already exists (delete the run dir to force a retrain).

    Entry point: ``python -m src.modeling.finetune``.
    """
    for slug, name in MODELS.items():
        for domain in DOMAINS:
            out_dir = run_dir_for(name, domain)
            if load_cached_run(out_dir) is not None:
                print(f"[{slug}/{domain}] already trained -> {out_dir}")
                continue
            print(f"\n=== DAPT {slug} / {domain} ===")
            finetune_dapt(
                domain,
                base_model=name,
                out_dir=out_dir,
                batch_size=BATCH_SIZE[slug],
                grad_accum=GRAD_ACCUM[slug],
                learning_rate=LEARNING_RATE[slug],
            )


if __name__ == "__main__":
    train_all()
