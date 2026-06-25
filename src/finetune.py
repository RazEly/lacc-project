"""Domain-adaptive continued pre-training (DAPT) of a causal LM (step 4).

Continues causal next-token pre-training (Gururangan et al. 2020) of a baseline
German decoder LM on the domain-labelled ``german-commons`` splits
(``data/domain_phy`` / ``data/domain_bio``), which are disjoint from the PoTeC
stimuli used for the reading-time analysis — so there is no leakage.

Physics and biology are fine-tuned independently. The run saves ``n_checkpoints``
models evenly spaced by training step (including a step-0 baseline), recording
validation perplexity + cumulative words processed at each, giving a fine-tuning
progress curve with several points inside a single epoch.
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

from src.config import CHECKPOINTS_DIR, DEFAULT_MODEL, DOMAIN_BIO_DIR, DOMAIN_PHY_DIR
from src.surprisal import compute_surprisal, load_causal_lm

DOMAIN_DIRS = {"physics": DOMAIN_PHY_DIR, "biology": DOMAIN_BIO_DIR}


def _tokenize_and_chunk(ds, tokenizer, block_size, text_col="text"):
    """Tokenize a text dataset and pack into fixed-size LM blocks.

    An ``eos`` token is appended to every document before packing so concatenated
    blocks carry a boundary between documents (GPT-2's tokenizer adds none on its
    own), preventing the model from training across unrelated documents as if they
    were continuous text.
    """

    def tok(batch):
        out = tokenizer(batch[text_col])
        if tokenizer.eos_token_id is not None:
            for ids in out["input_ids"]:
                ids.append(tokenizer.eos_token_id)
        return out

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

    def group(batch):
        concat = sum(batch["input_ids"], [])
        n = (len(concat) // block_size) * block_size
        ids = [concat[i : i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}

    return ds.map(group, batched=True, remove_columns=ds.column_names)


class _CheckpointSchedule(TrainerCallback):
    """Save ``n_checkpoints`` models evenly spaced by training step.

    The schedule spans the whole run: ``n_checkpoints`` global steps evenly
    spaced from 0 to ``max_steps``. With ``include_baseline`` the first
    checkpoint (step 0) is the un-fine-tuned model, so the manifest carries a
    baseline anchor and the remaining ones march up to the final epoch. Saving by
    step (not epoch) allows several checkpoints inside a single epoch.
    """

    def __init__(self, trainer, out_dir, n_checkpoints, words_per_epoch, manifest,
                 include_baseline=True):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.n_checkpoints = n_checkpoints
        self.words_per_epoch = words_per_epoch
        self.manifest = manifest
        self.include_baseline = include_baseline
        self._targets: dict[int, int] = {}  # global_step -> checkpoint index

    def on_train_begin(self, args, state, control, **kwargs):
        # Evenly space the checkpoints across the whole run. include_baseline puts
        # the first at step 0; otherwise the first lands after the first chunk.
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
                f"distinct steps fit in max_steps={state.max_steps}; some collided "
                "and were dropped (raise max_steps / lower n_checkpoints)."
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
        metrics = self.trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"])
        words_seen = round(state.epoch * self.words_per_epoch)
        self.manifest.append(
            {
                "checkpoint": str(ckpt),
                "index": idx,
                "epoch": round(state.epoch, 3),
                "step": state.global_step,
                "words_seen": words_seen,
                "perplexity": ppl,
            }
        )
        print(
            f"  [{ckpt.name}] epoch={state.epoch:.2f} step={state.global_step} "
            f"words_seen={words_seen:,} perplexity={ppl:.2f}"
        )


class _EpochProgress(TrainerCallback):
    """tqdm bar over training epochs."""

    def __init__(self, epochs, domain):
        self.bar = tqdm(total=epochs, desc=f"DAPT {domain}", unit="epoch")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.bar.close()


def finetune_dapt(
    domain: str,
    base_model: str = DEFAULT_MODEL,
    epochs: int = 3,
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
) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    Trains for ``epochs`` epochs and saves ``n_checkpoints`` models evenly spaced
    by training step across the whole run (so several land inside one epoch). With
    ``include_baseline`` the first checkpoint is the un-fine-tuned model (step 0).
    ``max_docs`` truncates the corpus (smoke testing). The returned DataFrame has
    columns ``domain``, ``checkpoint``, ``index``, ``epoch``, ``step``,
    ``words_seen``, ``perplexity``.
    """
    out_dir = Path(out_dir or CHECKPOINTS_DIR / f"{Path(base_model).name}_{domain}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    # german-gpt2 ships an eos/pad id (50265) one past its embedding rows
    # (vocab_size 50265); using it as a document separator / pad would index out
    # of range on the GPU (device-side assert). Grow the embeddings to cover every
    # tokenizer id; the new rows train during DAPT.
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    if max_docs:
        raw = raw.select(range(min(max_docs, len(raw))))

    # Split at the document level *before* packing so no document contributes
    # blocks to both train and validation (block-level splitting leaks adjacent
    # text and makes eval perplexity optimistic).
    raw_split = raw.train_test_split(test_size=val_frac, seed=seed)

    # Progress-curve x-axis: words the model actually trains on -> count the
    # train docs only (the held-out val fraction is never seen).
    words_per_epoch = sum(len(str(t).split()) for t in raw_split["train"]["text"])

    split = {
        "train": _tokenize_and_chunk(raw_split["train"], tokenizer, block_size),
        "test": _tokenize_and_chunk(raw_split["test"], tokenizer, block_size),
    }

    # 12 GB VRAM, 124M-param GPT-2: VRAM is slack, so optimise for throughput.
    # bf16 where the GPU supports it (Ampere+), else fp16; tf32 matmuls are free
    # on Ampere+. grad_accum lifts the effective batch without more memory.
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir / "_hf"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        # Continued pre-training: low LR + short warmup to avoid an early loss
        # spike / catastrophic forgetting from the baseline weights.
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
            trainer, out_dir, n_checkpoints, words_per_epoch, manifest,
            include_baseline=include_baseline,
        )
    )
    trainer.add_callback(_EpochProgress(epochs, domain))
    trainer.train()

    df = pd.DataFrame(manifest)
    df.insert(0, "domain", domain)
    return df


def recompute_surprisal_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame, prompt=None
) -> pd.DataFrame:
    """Recompute step-2 surprisal with each fine-tuned checkpoint.

    Returns the surprisal table for every checkpoint concatenated, tagged with
    ``checkpoint`` / ``index`` / ``epoch`` / ``domain`` so versions can be
    compared. ``index`` is the per-domain checkpoint number (0 = baseline) and is
    the stable key for pairing physics vs biology checkpoints (``epoch`` floats
    can differ slightly between domains).
    """
    frames = []
    # iterrows (not itertuples): the manifest has a column literally named
    # "index" which itertuples renames (clashes with tuple.index).
    for _, row in manifest.iterrows():
        model, tok = load_causal_lm(row.checkpoint)
        sup = compute_surprisal(words_df, model, tok, prompt=prompt)
        sup["checkpoint"] = row.checkpoint
        sup["index"] = row["index"]  # checkpoint index: stable across domains
        sup["epoch"] = row.epoch
        sup["domain"] = row.domain
        frames.append(sup)
    return pd.concat(frames, ignore_index=True)
