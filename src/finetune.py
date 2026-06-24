"""Domain-adaptive continued pre-training (DAPT) of a causal LM (step 4).

Continues causal next-token pre-training (Gururangan et al. 2020) of a baseline
German decoder LM on the domain-labelled ``german-commons`` splits
(``data/domain_phy`` / ``data/domain_bio``), which are disjoint from the PoTeC
stimuli used for the reading-time analysis — so there is no leakage.

Physics and biology are fine-tuned independently. Every ``save_every`` epochs the
model is checkpointed and its validation perplexity + cumulative words processed
are recorded, giving a fine-tuning progress curve.
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
    """Tokenize a text dataset and pack into fixed-size LM blocks."""
    def tok(batch):
        return tokenizer(batch[text_col])

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

    def group(batch):
        concat = sum(batch["input_ids"], [])
        n = (len(concat) // block_size) * block_size
        ids = [concat[i:i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}

    return ds.map(group, batched=True, remove_columns=ds.column_names)


class _CheckpointEveryN(TrainerCallback):
    """Save model + record perplexity / words_seen every ``save_every`` epochs."""

    def __init__(self, trainer, out_dir, save_every, words_per_epoch, manifest):
        self.trainer = trainer
        self.out_dir = Path(out_dir)
        self.save_every = save_every
        self.words_per_epoch = words_per_epoch
        self.manifest = manifest

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if epoch == 0 or epoch % self.save_every:
            return
        ckpt = self.out_dir / f"epoch_{epoch}"
        self.trainer.save_model(str(ckpt))
        metrics = self.trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"])
        self.manifest.append({
            "checkpoint": str(ckpt),
            "epoch": epoch,
            "words_seen": epoch * self.words_per_epoch,
            "perplexity": ppl,
        })
        print(f"  [{ckpt.name}] words_seen={epoch * self.words_per_epoch:,} "
              f"perplexity={ppl:.2f}")


class _EpochProgress(TrainerCallback):
    """tqdm bar over training epochs."""

    def __init__(self, epochs, domain):
        self.bar = tqdm(total=epochs, desc=f"DAPT {domain}", unit="epoch")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.bar.close()


def finetune_dapt(domain: str, base_model: str = DEFAULT_MODEL, epochs: int = 10,
                  save_every: int = 2, block_size: int = 512,
                  batch_size: int = 64, grad_accum: int = 1, val_frac: float = 0.05,
                  out_dir=None, max_docs=None) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    ``max_docs`` truncates the corpus (smoke testing). The returned DataFrame has
    columns ``domain``, ``checkpoint``, ``epoch``, ``words_seen``, ``perplexity``.
    """
    out_dir = Path(out_dir or CHECKPOINTS_DIR / f"{Path(base_model).name}_{domain}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)

    raw = load_from_disk(str(DOMAIN_DIRS[domain]))
    if max_docs:
        raw = raw.select(range(min(max_docs, len(raw))))
    words_per_epoch = sum(len(str(t).split()) for t in raw["text"])

    lm = _tokenize_and_chunk(raw, tokenizer, block_size)
    split = lm.train_test_split(test_size=val_frac, seed=0)

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
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        tf32=use_cuda,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="no",            # checkpointing handled by the callback
        logging_steps=50,
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, data_collator=collator,
                      train_dataset=split["train"], eval_dataset=split["test"])

    manifest: list[dict] = []
    trainer.add_callback(_CheckpointEveryN(
        trainer, out_dir, save_every, words_per_epoch, manifest))
    trainer.add_callback(_EpochProgress(epochs, domain))
    trainer.train()

    df = pd.DataFrame(manifest)
    df.insert(0, "domain", domain)
    return df


def recompute_surprisal_over_checkpoints(words_df: pd.DataFrame,
                                         manifest: pd.DataFrame,
                                         prompt=None) -> pd.DataFrame:
    """Recompute step-2 surprisal with each fine-tuned checkpoint.

    Returns the surprisal table for every checkpoint concatenated, tagged with
    ``checkpoint`` / ``epoch`` / ``domain`` so versions can be compared.
    """
    frames = []
    for row in manifest.itertuples():
        model, tok = load_causal_lm(row.checkpoint)
        sup = compute_surprisal(words_df, model, tok, prompt=prompt)
        sup["checkpoint"] = row.checkpoint
        sup["epoch"] = row.epoch
        sup["domain"] = row.domain
        frames.append(sup)
    return pd.concat(frames, ignore_index=True)
