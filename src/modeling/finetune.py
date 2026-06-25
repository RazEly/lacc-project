"""Domain-adaptive continued pre-training (DAPT) of a causal LM (step 4).

Continues causal next-token pre-training (Gururangan et al. 2020) of a baseline
German decoder LM on the domain-labelled ``german-commons`` splits
(``data/domain_phy`` / ``data/domain_bio``), which are disjoint from the PoTeC
stimuli used for the reading-time analysis — so there is no leakage.

Physics and biology are fine-tuned independently, on a shared *token* budget
rather than a shared epoch count: training runs for a fixed number of training
tokens (``max_tokens``), not a fixed number of passes. Because the two domain
corpora differ in size, equal epochs would mean unequal token exposure; equal
``max_tokens`` (with identical batch/block/grad-accum, so identical tokens per
step) makes both domains see the *same* number of tokens at the same checkpoint
index. ``count_domain_tokens`` reports a domain's available training tokens so a
caller can pick the largest equal budget that fits one pass of each.

The run saves ``n_checkpoints`` models evenly spaced by training step (including
a step-0 baseline), recording validation perplexity + cumulative tokens
processed at each, giving a fine-tuning progress curve.
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
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.config import CHECKPOINTS_DIR, DEFAULT_MODEL, DOMAIN_BIO_DIR, DOMAIN_PHY_DIR
from src.features.surprisal import compute_surprisal, load_causal_lm

DOMAIN_DIRS = {"physics": DOMAIN_PHY_DIR, "biology": DOMAIN_BIO_DIR}


def _tokenize_and_chunk(ds, tokenizer, block_size, text_col="text", mlm=False):
    """Tokenize a text dataset and pack into fixed-size LM blocks.

    A separator token is appended to every document before packing so concatenated
    blocks carry a boundary between documents (GPT-2's tokenizer adds none on its
    own), preventing the model from training across unrelated documents as if they
    were continuous text. For causal training ``labels`` are the shifted inputs;
    for ``mlm`` they are omitted so the MLM collator can mask dynamically.
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
        out = {"input_ids": ids}
        if not mlm:  # MLM collator builds masked labels itself
            out["labels"] = [x[:] for x in ids]
        return out

    return ds.map(group, batched=True, remove_columns=ds.column_names)


def _prepare_splits(domain, tokenizer, block_size, val_frac, seed, max_docs, is_mlm):
    """Load a domain corpus and build packed train/test LM blocks.

    Shared by ``finetune_dapt`` and ``count_domain_tokens`` so the token count a
    caller budgets against is exactly the count the trainer will see (same load,
    split, tokenizer and packing). Returns the split dict plus ``words_per_epoch``
    (whitespace words in the train docs, for the legacy words-seen column).
    """
    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    if max_docs:
        raw = raw.select(range(min(max_docs, len(raw))))

    # Split at the document level *before* packing so no document contributes
    # blocks to both train and validation (block-level splitting leaks adjacent
    # text and makes eval perplexity optimistic).
    raw_split = raw.train_test_split(test_size=val_frac, seed=seed)

    # Legacy words-seen x-axis: words the model actually trains on -> count the
    # train docs only (the held-out val fraction is never seen).
    words_per_epoch = sum(len(str(t).split()) for t in raw_split["train"]["text"])

    split = {
        "train": _tokenize_and_chunk(raw_split["train"], tokenizer, block_size, mlm=is_mlm),
        "test": _tokenize_and_chunk(raw_split["test"], tokenizer, block_size, mlm=is_mlm),
    }
    return split, words_per_epoch


def count_domain_tokens(
    domain: str,
    base_model: str = DEFAULT_MODEL,
    block_size: int = 512,
    val_frac: float = 0.05,
    seed: int = 0,
    max_docs=None,
    objective: str = "causal",
) -> int:
    """Available training tokens for ``domain`` (= packed train blocks × block_size).

    Use ``min(count_domain_tokens("physics"), count_domain_tokens("biology"))`` as
    a shared ``max_tokens`` so both domains train on an equal token budget that
    fits within one pass of each corpus. Args must match the ``finetune_dapt`` call
    they budget for (``base_model``/``block_size``/``val_frac``/``seed``/``max_docs``),
    since those determine the packed block count.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    split, _ = _prepare_splits(
        domain, tokenizer, block_size, val_frac, seed, max_docs, objective == "mlm"
    )
    return len(split["train"]) * block_size


class _CheckpointSchedule(TrainerCallback):
    """Save ``n_checkpoints`` models evenly spaced by training step.

    The schedule spans the whole run: ``n_checkpoints`` global steps evenly
    spaced from 0 to ``max_steps``. With ``include_baseline`` the first
    checkpoint (step 0) is the un-fine-tuned model, so the manifest carries a
    baseline anchor and the remaining ones march up to the final epoch. Saving by
    step (not epoch) allows several checkpoints inside a single epoch.
    """

    def __init__(self, trainer, out_dir, n_checkpoints, words_per_epoch,
                 tokens_per_step, manifest, include_baseline=True):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.n_checkpoints = n_checkpoints
        self.words_per_epoch = words_per_epoch
        self.tokens_per_step = tokens_per_step
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
        # tokens_seen is the cross-domain-comparable x-axis: with identical tokens
        # per step it depends only on global_step, so the same checkpoint index
        # carries the same tokens_seen in both domains. words_seen is kept (legacy).
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


def finetune_dapt(
    domain: str,
    base_model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
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
    objective: str = "causal",
) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    ``objective`` is ``"causal"`` (decoder, next-token; the surprisal pipeline) or
    ``"mlm"`` (encoder, masked-LM; the attention experiment).

    The run is budgeted by training *tokens*, not epochs: it trains for
    ``max_tokens`` tokens (rounded up to whole optimiser steps), where one step
    processes ``block_size * batch_size * grad_accum`` tokens. Passing the *same*
    ``max_tokens`` to two domains makes both see the same number of tokens at the
    same checkpoint index — use ``count_domain_tokens`` to size an equal budget.
    ``max_tokens=None`` defaults to a single pass over the domain's own training
    tokens (per-domain, hence *not* equalised across domains).

    Saves ``n_checkpoints`` models evenly spaced by training step across the whole
    run. With ``include_baseline`` the first checkpoint is the un-fine-tuned model
    (step 0). ``max_docs`` truncates the corpus (smoke testing). The returned
    DataFrame has columns ``domain``, ``checkpoint``, ``index``, ``epoch``,
    ``step``, ``tokens_seen``, ``words_seen``, ``perplexity`` (pseudo-perplexity
    over masked tokens when ``objective="mlm"``).
    """
    is_mlm = objective == "mlm"
    out_dir = Path(out_dir or CHECKPOINTS_DIR / f"{Path(base_model).name}_{domain}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model_cls = AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM
    model = model_cls.from_pretrained(base_model)
    # german-gpt2 ships an eos/pad id (50265) one past its embedding rows
    # (vocab_size 50265); using it as a document separator / pad would index out
    # of range on the GPU (device-side assert). Grow the embeddings to cover every
    # tokenizer id; the new rows train during DAPT.
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    split, words_per_epoch = _prepare_splits(
        domain, tokenizer, block_size, val_frac, seed, max_docs, is_mlm
    )

    # Token budget -> optimiser steps. One step trains on tokens_per_step real
    # tokens (blocks are packed to exactly block_size, no padding). Default budget
    # is one full pass over this domain's train blocks.
    tokens_per_step = block_size * batch_size * grad_accum
    available_tokens = len(split["train"]) * block_size
    budget = max_tokens if max_tokens is not None else available_tokens
    max_steps = max(1, math.ceil(budget / tokens_per_step))

    # 12 GB VRAM, 124M-param GPT-2: VRAM is slack, so optimise for throughput.
    # bf16 where the GPU supports it (Ampere+), else fp16; tf32 matmuls are free
    # on Ampere+. grad_accum lifts the effective batch without more memory.
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir / "_hf"),
        max_steps=max_steps,
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
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=is_mlm, mlm_probability=0.15 if is_mlm else None
    )
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
        )
    )
    trainer.add_callback(_StepProgress(domain))
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
