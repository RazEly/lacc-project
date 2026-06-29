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

import json
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

from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_MODEL,
    DOMAIN_BIO_DIR,
    DOMAIN_PHY_DIR,
    GENERAL_HF_CONFIG,
    GENERAL_HF_REPO,
)
from src.features.surprisal import compute_surprisal, load_causal_lm

DOMAIN_DIRS = {"physics": DOMAIN_PHY_DIR, "biology": DOMAIN_BIO_DIR}


def _lora_targets(model, override):
    """Per-architecture LoRA target + embedding modules for DAPT.

    Domain-adaptive pre-training shifts vocabulary and feed-forward knowledge, so
    LoRA targets the MLP as well as attention (PEFT's default is attention-only,
    too thin for domain adaptation). The embedding modules are returned separately
    so the caller can ``modules_to_save`` them when the embeddings were grown.
    ``override`` short-circuits the target choice (embeddings still per-arch).
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
    """Save model checkpoints by training step (NOT by epoch).

    Default: ``n_checkpoints`` global steps evenly spaced from 0 to ``max_steps``.
    If ``checkpoint_steps`` is given, save at exactly those steps instead — e.g.
    Škrjanec et al. (PoTeC reader-experience) save at 4ⁿ steps
    ``[4, 16, 64, 256, 1024, 4096, 16384]``. With ``include_baseline`` the step-0
    un-fine-tuned model is checkpoint index 0; the listed steps then get indices
    1, 2, … in order.
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
            # Explicit step targets (e.g. the paper's 4ⁿ schedule); baseline is
            # index 0, the listed steps are indices 1.. in order. Steps past
            # max_steps are clamped onto the final step.
            steps = {min(int(s), state.max_steps): i
                     for i, s in enumerate(self.checkpoint_steps, start=1)}
            self._targets = steps
        else:
            # Evenly space the checkpoints across the whole run. include_baseline
            # puts the first at step 0; otherwise the first lands after one chunk.
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


def _run_signature(**kwargs) -> dict:
    """Serialise the args that determine a DAPT run's checkpoints.

    Any change to one of these (budget, schedule, optimiser, LoRA config, …) makes
    the saved checkpoints stale, so a cached run is only reused when its stored
    signature matches the current call exactly. Lists are JSON-normalised to tuples
    so ``lora_target_modules`` compares by value.
    """
    return {k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in sorted(kwargs.items())}


def _load_cached_manifest(out_dir: Path, signature: dict) -> pd.DataFrame | None:
    """Return the saved manifest iff it matches ``signature`` and is complete.

    A run is reusable only when (1) the signature sidecar matches the current call
    and (2) every checkpoint dir the manifest references still exists on disk.
    Otherwise the cache is ignored and training reruns from scratch.
    """
    manifest_path = out_dir / "manifest.csv"
    sig_path = out_dir / "run_signature.json"
    if not (manifest_path.exists() and sig_path.exists()):
        return None
    if json.loads(sig_path.read_text()) != json.loads(json.dumps(signature)):
        return None
    df = pd.read_csv(manifest_path)
    if not all(Path(c).exists() for c in df["checkpoint"]):
        return None
    return df


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
    objective: str = "causal",
    lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules=None,
) -> pd.DataFrame:
    """DAPT-fine-tune ``base_model`` on one domain; return a checkpoint manifest.

    ``objective`` is ``"causal"`` (decoder, next-token; the surprisal pipeline) or
    ``"mlm"`` (encoder, masked-LM; the attention experiment).

    ``lora=True`` trains LoRA adapters (PEFT) instead of all weights — the default
    method driven from ``main.py``. Checkpoints then store the adapter only;
    ``surprisal.load_causal_lm`` reattaches it to the base model. ``target_modules``
    defaults (``None``) to a per-architecture attention+MLP set via ``_lora_targets``
    (GPT-2 ``c_attn``/``c_fc``/``c_proj``; Llama q/k/v/o + gate/up/down), so DAPT
    adapts feed-forward knowledge, not just attention, for every model.

    The run is budgeted by training *tokens*, not epochs: it trains for
    ``max_tokens`` tokens (rounded up to whole optimiser steps), where one step
    processes ``block_size * batch_size * grad_accum`` tokens. Passing the *same*
    ``max_tokens`` to two domains makes both see the same number of tokens at the
    same checkpoint index — use ``count_domain_tokens`` to size an equal budget.
    ``max_tokens=None`` defaults to a single pass over the domain's own training
    tokens (per-domain, hence *not* equalised across domains).

    ``max_steps`` overrides the token budget to train for a fixed step count.
    Checkpoints are saved by step, NOT by epoch: ``n_checkpoints`` evenly spaced by
    default, or — when ``checkpoint_steps`` is given — at exactly those steps (e.g.
    Škrjanec et al.'s 4ⁿ schedule ``[4, 16, 64, 256, 1024, 4096, 16384]``). With
    ``include_baseline`` the first checkpoint is the un-fine-tuned model (step 0).
    ``max_docs`` truncates the corpus (smoke testing).

    Resumable: each run writes its manifest + a signature of every arg that affects
    the result to ``out_dir``. A later call with the same signature whose checkpoints
    still exist returns the saved manifest and skips training; changing any such arg
    retrains (stale checkpoints are not reused). The returned
    DataFrame has columns ``domain``, ``checkpoint``, ``index``, ``epoch``,
    ``step``, ``tokens_seen``, ``words_seen``, ``perplexity`` (pseudo-perplexity
    over masked tokens when ``objective="mlm"``).
    """
    is_mlm = objective == "mlm"
    # Suffix the run dir by method so LoRA and full-FT checkpoints never clobber.
    suffix = "_lora" if lora else ""
    out_dir = Path(
        out_dir or CHECKPOINTS_DIR / f"{Path(base_model).name}_{domain}{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: if this exact run already produced checkpoints, reload its manifest
    # and skip training. The signature pins every arg that changes the result, so a
    # different budget/schedule/LoRA config retrains rather than reusing stale weights.
    # max_steps / checkpoint_steps enter the signature ONLY when set, so existing
    # cached runs (which predate these args) keep matching and are still reused.
    _sched = {}
    if max_steps is not None:
        _sched["max_steps"] = max_steps
    if checkpoint_steps is not None:
        _sched["checkpoint_steps"] = list(checkpoint_steps)
    signature = _run_signature(
        base_model=base_model, domain=domain, max_tokens=max_tokens, **_sched,
        n_checkpoints=n_checkpoints, include_baseline=include_baseline,
        block_size=block_size, batch_size=batch_size, grad_accum=grad_accum,
        val_frac=val_frac, learning_rate=learning_rate, warmup_ratio=warmup_ratio,
        seed=seed, max_docs=max_docs, objective=objective, lora=lora, lora_r=lora_r,
        lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    cached = _load_cached_manifest(out_dir, signature)
    if cached is not None:
        print(f"  [{domain}] reusing {len(cached)} saved checkpoints in {out_dir} "
              "(skipping training)")
        return cached

    # Local cache missed: try the Hub before paying for a GPU run. A matching
    # remote run is downloaded into out_dir and reused under the same signature
    # gate; a miss (disabled / no repo / config drift) falls through to training.
    from src.modeling import hub

    remote = hub.try_download_run(out_dir, signature)
    if remote is not None:
        print(f"  [{domain}] reusing {len(remote)} checkpoints pulled from the Hub "
              "(skipping training)")
        return remote

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model_cls = AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM
    model = model_cls.from_pretrained(base_model)
    # german-gpt2 ships an eos/pad id (50265) one past its embedding rows
    # (vocab_size 50265); using it as a document separator / pad would index out
    # of range on the GPU (device-side assert). Grow the embeddings to cover every
    # tokenizer id; the new rows train during DAPT.
    resized = len(tokenizer) > model.config.vocab_size
    if resized:
        model.resize_token_embeddings(len(tokenizer))

    if lora:
        from peft import LoraConfig, TaskType, get_peft_model

        targets, embed_modules = _lora_targets(model, lora_target_modules)
        peft_cfg = LoraConfig(
            task_type=None if is_mlm else TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,  # attn + MLP (None -> PEFT per-arch default)
            # When embeddings were grown, the new rows are randomly initialised and
            # LoRA freezes the base — without saving them they would be re-randomised
            # (differently) at load time. Save the embedding module so they survive.
            modules_to_save=embed_modules if resized else None,
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    split, words_per_epoch = _prepare_splits(
        domain, tokenizer, block_size, val_frac, seed, max_docs, is_mlm
    )

    # Optimiser steps. One step trains on tokens_per_step real tokens (blocks are
    # packed to exactly block_size, no padding). ``max_steps`` overrides the token
    # budget directly (e.g. the paper's fixed 16,384-step schedule); otherwise it
    # is derived from ``max_tokens`` (default: one pass over the train blocks).
    tokens_per_step = block_size * batch_size * grad_accum
    available_tokens = len(split["train"]) * block_size
    if max_steps is None:
        budget = max_tokens if max_tokens is not None else available_tokens
        max_steps = max(1, math.ceil(budget / tokens_per_step))

    # Batch sizes (config.DAPT_BATCH_SIZE) target ~16 GB VRAM; for 124M-param
    # GPT-2 VRAM is slack, so optimise for throughput. bf16 where the GPU supports
    # it (Ampere+), else fp16; tf32 matmuls are free on Ampere+. grad_accum lifts
    # the effective batch without more memory.
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
            checkpoint_steps=checkpoint_steps,
        )
    )
    trainer.add_callback(_StepProgress(domain))
    trainer.train()

    df = pd.DataFrame(manifest)
    df.insert(0, "domain", domain)
    # Persist manifest + signature so a later run reuses these checkpoints instead
    # of retraining (see _load_cached_manifest).
    df.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "run_signature.json").write_text(json.dumps(signature))
    # Mirror the finished run to the Hub so future runs (this disk or another)
    # download instead of retraining. Best effort — never fails the run.
    hub.upload_run(out_dir)
    return df


# ── cross-domain perplexity (lite pipeline) ──────────────────────────────────
# How DAPT for one field shifts the model's fit on the OTHER field and on plain
# general text: score the baseline and both DAPT models on the SAME held-out
# general / physics / biology validation blocks.

EVAL_GENERAL_DOCS = 400  # streamed German-Wikipedia docs for the general val set
EVAL_MAX_BLOCKS = 200    # cap packed eval blocks per domain (keep the run light)


def _general_eval_corpus(max_docs: int = EVAL_GENERAL_DOCS):
    """A small German-Wikipedia slice as the general-domain eval corpus.

    Streamed so nothing downloads beyond the first ``max_docs`` articles, then
    materialised into an in-memory ``text`` dataset that packs exactly like the
    domain corpora. This is the out-of-domain reference for cross-domain ppl.
    """
    import itertools

    from datasets import Dataset, load_dataset

    stream = load_dataset(
        GENERAL_HF_REPO, GENERAL_HF_CONFIG, split="train", streaming=True
    )
    docs = [d["text"] for d in itertools.islice(stream, max_docs)]
    return Dataset.from_dict({"text": docs})


def build_eval_sets(
    base_model: str = DEFAULT_MODEL,
    block_size: int = 512,
    val_frac: float = 0.05,
    seed: int = 0,
    general_docs: int = EVAL_GENERAL_DOCS,
    max_blocks: int = EVAL_MAX_BLOCKS,
) -> tuple:
    """Packed validation blocks for general/physics/biology under one tokenizer.

    Physics and biology reuse each domain's *held-out* training split (same
    ``val_frac``/``seed`` as ``finetune_dapt``), so the eval text never overlaps
    that domain's DAPT data; general is a German-Wikipedia slice. All three are
    tokenized once with the base model's tokenizer — every checkpoint shares that
    tokenizer, so the identical blocks score all models (the apples-to-apples
    requirement for cross-domain perplexity). Returns ``(tokenizer, {domain: blocks})``.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    sets = {}
    for domain in ("physics", "biology"):
        split, _ = _prepare_splits(
            domain, tokenizer, block_size, val_frac, seed, None, False
        )
        sets[domain] = split["test"]
    sets["general"] = _tokenize_and_chunk(
        _general_eval_corpus(general_docs), tokenizer, block_size
    )
    if max_blocks:
        sets = {d: b.select(range(min(max_blocks, len(b)))) for d, b in sets.items()}
    return tokenizer, sets


@torch.no_grad()
def _perplexity(model, blocks, batch_size: int = 8) -> float:
    """Token-weighted causal-LM perplexity of ``model`` over packed ``blocks``.

    Sums next-token cross-entropy over all blocks and exponentiates the per-token
    mean, so blocks contribute in proportion to their token count. Logits are
    upcast to fp32 before the loss (fp16 inference weights are numerically fine
    for the forward, not for the reduction).
    """
    import torch.nn.functional as F

    device = model.device
    ids_all = blocks["input_ids"]
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(ids_all), batch_size):
        ids = torch.tensor(ids_all[i : i + batch_size], device=device)
        logits = model(ids).logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tok += shift_labels.numel()
    return math.exp(total_loss / total_tok)


def cross_domain_perplexity(
    phys_manifest: pd.DataFrame,
    bio_manifest: pd.DataFrame,
    base_model: str = DEFAULT_MODEL,
    **eval_kw,
) -> pd.DataFrame:
    """Perplexity of baseline / physics-DAPT / biology-DAPT on all three domains.

    The baseline is each run's step-0 checkpoint (the un-fine-tuned model, but
    saved with the grown embeddings so it loads identically to the adapted
    checkpoints); physics/biology are the final checkpoint of their run. Every
    model is scored on the *same* general/physics/biology validation blocks
    (``build_eval_sets``), so the table reads off how DAPT for physics moves the
    fit on biology and general text. Returns rows ``model``, ``eval_domain``,
    ``perplexity``.
    """
    baseline = phys_manifest.loc[phys_manifest["index"].idxmin(), "checkpoint"]
    physics = phys_manifest.loc[phys_manifest["index"].idxmax(), "checkpoint"]
    biology = bio_manifest.loc[bio_manifest["index"].idxmax(), "checkpoint"]
    models = {"baseline": baseline, "physics-DAPT": physics, "biology-DAPT": biology}

    _tok, sets = build_eval_sets(base_model=base_model, **eval_kw)
    rows = []
    for label, ckpt in models.items():
        model, _ = load_causal_lm(ckpt)
        for domain, blocks in sets.items():
            ppl = _perplexity(model, blocks)
            rows.append({"model": label, "eval_domain": domain, "perplexity": ppl})
            print(f"  {label:>13s} on {domain:<8s}: ppl={ppl:.2f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


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
