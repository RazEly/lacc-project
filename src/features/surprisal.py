"""Model surprisal from a decoder-only causal LM (step 2) + its wide cache.

Word surprisal = sum of sub-token surprisals (-log2 p(token | left context)),
aligned via the tokenizer's ``word_ids``. Each text is scored as one sequence so
context spans sentence boundaries (Eq. 1). An optional ``prompt`` prepends
domain-matched context.

The bottom half persists the (expensive) forwards to one wide CSV so reruns
reload instead of re-scoring — see the "wide surprisal cache" section.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEFAULT_MODEL, SURPRISAL_CACHE_PATH, WORD_KEY


def _load_tokenizer(name_or_path: str):
    # add_prefix_space (BPE/german-gpt2): first word tokenizes like a mid-sentence
    # word. SentencePiece (Llama) adds it itself and may reject the kwarg.
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
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    if (Path(name_or_path) / "adapter_config.json").is_file():
        # LoRA checkpoint: load the adapter's base model, match embedding size,
        # then merge the adapter so the forward needs no PEFT machinery.
        from peft import PeftConfig, PeftModel

        base = PeftConfig.from_pretrained(name_or_path).base_model_name_or_path
        tokenizer = _load_tokenizer(base)
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        resize_with_mean_init(model, tokenizer)
        model = PeftModel.from_pretrained(model, name_or_path).merge_and_unload()
    else:
        tokenizer = _load_tokenizer(name_or_path)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        # Same resize as the adapter branch: german-gpt2's eos id sits one past its
        # embedding rows, and the document-boundary separator (below, in
        # score_words) feeds that id in as CONTEXT — an un-resized base model would
        # index out of bounds. No-op for Llama (eos already in vocab).
        resize_with_mean_init(model, tokenizer)
    model.eval()
    # from_pretrained leaves the model on CPU; move to GPU once here.
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def _resolve_prompt(prompt, domain: str | None):
    """Resolve ``prompt`` for ``domain``: None, str, list[str], or {domain: ...} dict.

    A dict selects the entry for ``domain`` (missing -> None). Anything else passes
    through unchanged, so a bare str or list is shared across every domain.
    """
    if prompt is None:
        return None
    if isinstance(prompt, dict):
        return prompt.get(domain)
    return prompt


def _as_prompt_list(prompt) -> list[str]:
    """Normalise a resolved prompt to a (possibly empty) list of passages."""
    if not prompt:
        return []
    return [prompt] if isinstance(prompt, str) else list(prompt)


def _mean_bits(per_prompt: list[list]) -> list:
    """Element-wise mean over equal-length per-word bit lists, ignoring ``None``.

    Averaging in surprisal (bits) space equals the surprisal of the geometric-mean
    predictive distribution over the priors — a pooled domain estimate no single
    passage dominates. A word left ``None`` (no left context) by every prior stays
    ``None``.
    """
    out = []
    for vals in zip(*per_prompt):
        present = [v for v in vals if v is not None]
        out.append(sum(present) / len(present) if present else None)
    return out


def _doc_separator_id(tokenizer) -> int:
    """The tokenizer's NATIVE end-of-document token id (eos, else sep).

    This is the real special token both base LMs were pretrained to read as a
    document break — NOT a newline. german-gpt2 exposes ``eos_token_id`` (the id
    just past its original embedding rows, which is why the model is resized on
    load); Llama/TinyLlama exposes ``eos_token_id`` == 2. Raises if neither
    exists, rather than silently degrading the boundary to whitespace.
    """
    sep_id = tokenizer.eos_token_id
    if sep_id is None:
        sep_id = tokenizer.sep_token_id
    if sep_id is None:
        raise ValueError(
            "tokenizer exposes no eos/sep token for the document boundary; "
            "cannot separate the prior passage from the stimulus natively"
        )
    return int(sep_id)


@torch.no_grad()
def score_words(
    word_list, model, tokenizer, prompt=None, domain=None, max_prompt_tokens=None
):
    """Per-word surprisal (bits) for an ordered word sequence (a full text).

    With a ``prompt`` (a prior-reading passage), the sequence is
    ``[prior tokens] <eos> [stimulus tokens]`` — the prior and the stimulus are
    two documents joined by the tokenizer's NATIVE end-of-document token
    (``_doc_separator_id``), so the model treats the stimulus as a fresh document
    that merely follows domain-primed context (never as a continuation mislabelled
    with the prior's domain). ``max_prompt_tokens`` truncates the prior to a fixed
    budget so every prompt condition shares one context length.

    Returns a list the same length as ``word_list``; the first word is ``None``
    when it has no left context (no prompt prepended).
    """
    enc = tokenizer(word_list, is_split_into_words=True, return_tensors="pt")
    sent_ids = enc.input_ids[0]
    word_ids = enc.word_ids(0)

    prompt_text = _resolve_prompt(prompt, domain)
    if prompt_text:
        prior_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        if max_prompt_tokens is not None and len(prior_ids) > max_prompt_tokens:
            prior_ids = prior_ids[:max_prompt_tokens]
        # native document boundary between the prior passage and the stimulus.
        sep = torch.tensor([_doc_separator_id(tokenizer)], dtype=prior_ids.dtype)
        # drop the stimulus encoding's leading specials (e.g. Llama's BOS) — they
        # would otherwise land mid-sequence, after the boundary.
        n_lead = 0
        while n_lead < len(word_ids) and word_ids[n_lead] is None:
            n_lead += 1
        sent_ids = sent_ids[n_lead:]
        word_ids = word_ids[n_lead:]
        input_ids = torch.cat([prior_ids, sep, sent_ids])
        offset = len(prior_ids) + 1  # prior + the boundary token
    else:
        input_ids = sent_ids
        offset = 0

    input_ids = input_ids.to(next(model.parameters()).device)
    logits = model(input_ids.unsqueeze(0)).logits[0]  # [seq, vocab]
    logprobs = torch.log_softmax(logits.float(), dim=-1)  # natural log

    word_bits: dict[int, float] = {}
    for j, wid in enumerate(word_ids):
        if wid is None:
            continue
        pos = offset + j  # absolute position of this sentence token
        if pos == 0:
            continue  # first token, no predictor -> first word NaN
        # log2 p(token_pos | tokens < pos) = logprobs[pos-1, token_pos]
        lp = logprobs[pos - 1, int(sent_ids[j])].item() / math.log(2)
        word_bits[wid] = word_bits.get(wid, 0.0) - lp

    return [word_bits.get(i) for i in range(len(word_list))]


def compute_surprisal(
    words_df: pd.DataFrame, model, tokenizer, prompt=None, max_prompt_tokens=None
) -> pd.DataFrame:
    """Per-word surprisal for every PoTeC text.

    Each text is rebuilt as one sequence (ordered by ``word_index_in_text``) so
    context carries across sentences (Eq. 1). The text's first/last word is
    dropped to match the reading-time cleaning. ``prompt`` may be None, a string,
    a list of strings, or a ``{text_domain: str | list[str]}`` dict; each entry is
    a prior-reading passage joined to the stimulus by the native document boundary
    (see ``score_words``). When a domain resolves to SEVERAL priors, the stimulus
    is scored under each and the per-word surprisals are averaged (``_mean_bits``),
    so the column is a mean over priors. ``max_prompt_tokens`` fixes the prior's
    context length across conditions.

    Returns columns: ``text_id``, ``word_index_in_text``, ``surprisal`` (the mean
    over priors when several are given).
    """
    rows = []
    for text_id, text in words_df.sort_values(
        ["text_id", "word_index_in_text"]
    ).groupby("text_id", sort=False):
        domain = text["text_domain"].iloc[0]
        words = text["word"].fillna("").astype(str).tolist()
        prompts = _as_prompt_list(_resolve_prompt(prompt, domain))
        if len(prompts) > 1:
            bits = _mean_bits([
                score_words(
                    words, model, tokenizer, prompt=p, domain=domain,
                    max_prompt_tokens=max_prompt_tokens,
                )
                for p in prompts
            ])
        else:
            # None / single prior: score once (keeps the no-prompt path unchanged).
            bits = score_words(
                words, model, tokenizer,
                prompt=prompts[0] if prompts else None, domain=domain,
                max_prompt_tokens=max_prompt_tokens,
            )
        idx = text["word_index_in_text"].tolist()
        last = len(words) - 1
        for k, (wi, b) in enumerate(zip(idx, bits)):
            if b is None or k == last:  # text first (no context) / last word
                continue
            rows.append((text_id, wi, b))

    return pd.DataFrame(rows, columns=WORD_KEY + ["surprisal"])


# ── wide surprisal cache ─────────────────────────────────────────────────────
# One CSV for every model + checkpoint. One row per word (``WORD_KEY``), one
# column per source named ``<prefix>_<what>`` (prefix from ``config.MODEL_PREFIX``,
# e.g. ``gpt`` / ``llama``):
#
#     <p>_0                            baseline (un-adapted, checkpoint index 0)
#     <p>_<i>_phys / <p>_<i>_bio       domain-adapted checkpoint i>=1 (physics/biology)
#     <p>_prompt_phys / _bio           discipline-matched prompted baseline
#
# Each model fills its own columns; a model already present is reused while
# missing ones are computed and merged in.
_DOMAIN_SHORT = {"physics": "phys", "biology": "bio"}
# prompt_surp column (``s_prompt_*``) suffixes -> cache column ``<prefix>_prompt_*``.
# phys/bio = reader-aligned domain prior; neutral = length-matched non-domain prior.
_PROMPT_SUFFIXES = ["phys", "bio", "neutral"]


def _prompt_map(prefix: str) -> dict[str, str]:
    return {f"s_prompt_{s}": f"{prefix}_prompt_{s}" for s in _PROMPT_SUFFIXES}


def load_cache(path: Path = SURPRISAL_CACHE_PATH) -> pd.DataFrame | None:
    """Return the cached wide table, or ``None`` when no file exists yet."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def save_cache(cache: pd.DataFrame, path: Path = SURPRISAL_CACHE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)


def build_wide(
    prefix: str, surp_versions: pd.DataFrame, prompt_surp: pd.DataFrame
) -> pd.DataFrame:
    """Fold one model's ``surp_versions`` + ``prompt_surp`` into wide columns."""
    base_index = surp_versions["index"].min()

    def _col(sel, col):
        return sel[WORD_KEY + ["surprisal"]].rename(columns={"surprisal": col})

    base = surp_versions[surp_versions["index"] == base_index].drop_duplicates(WORD_KEY)
    wide = _col(base, f"{prefix}_0")
    for idx in sorted(surp_versions["index"].unique()):
        if idx == base_index:
            continue
        for domain, short in _DOMAIN_SHORT.items():
            sel = surp_versions[
                (surp_versions["index"] == idx) & (surp_versions["domain"] == domain)
            ]
            wide = wide.merge(
                _col(sel, f"{prefix}_{idx}_{short}"), on=WORD_KEY, how="outer"
            )

    # keep only mapped prompt columns (drops any extra scratch columns callers merged in)
    prompts = prompt_surp.rename(columns=_prompt_map(prefix))
    keep = [c for c in prompts.columns if c.startswith(f"{prefix}_prompt_")]
    return wide.merge(prompts[WORD_KEY + keep], on=WORD_KEY, how="outer")


def merge_model(cache: pd.DataFrame | None, wide: pd.DataFrame) -> pd.DataFrame:
    """Add/replace one model's columns in the cache and return the combined table."""
    if cache is None:
        return wide
    new_cols = [c for c in wide.columns if c not in WORD_KEY]
    cache = cache.drop(columns=[c for c in new_cols if c in cache.columns])
    return cache.merge(wide, on=WORD_KEY, how="outer")
