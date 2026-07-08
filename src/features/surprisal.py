"""Model surprisal from a decoder-only causal LM (step 2).

Word surprisal = sum of sub-token surprisals (-log2 p(token | left context)),
aligned via the tokenizer's ``word_ids``. Each SENTENCE is scored as its own
sequence, so context resets at sentence boundaries (never spans them). An
optional ``prompt`` prepends domain-matched context to each sentence.

Model/tokenizer loading lives in ``modeling.lm``. ``save_cache`` /
``load_cache`` persist these (expensive) forwards as one long CSV
(``surprisal.csv``), written once after every model has finished.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch

from src.config import DEVICE, SURPRISAL_CACHE_PATH, WORD_KEY
from src.modeling.lm import load_causal_lm


def save_cache(bundles: list[dict], path: Path = SURPRISAL_CACHE_PATH) -> None:
    """Write every model's surprisal to a CSV"""
    frames = []
    for b in bundles:
        frames.append(b["surp_versions"].assign(model=b["slug"], condition="dapt"))
        frames.append(
            b["prompt_surp"]
            .melt(id_vars=WORD_KEY, var_name="condition", value_name="surprisal")
            .assign(model=b["slug"])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def load_cache(slugs, path: Path = SURPRISAL_CACHE_PATH) -> list[dict] | None:
    """Rebuild the fit bundles from the CSV, or ``None`` when no file exists.

    ``manifest`` is None — perplexity diagnostics need the training manifests.
    """
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    bundles = []
    for slug in slugs:
        m = df[df["model"] == slug]
        sv = m[m["condition"] == "dapt"].drop(columns=["model", "condition"])
        # epoch is fractional (0.001..~2); an int cast would truncate it to 0/1.
        sv = sv.astype({"index": int, "epoch": float}).reset_index(drop=True)
        ps = (
            m[m["condition"] != "dapt"]
            .pivot(index=WORD_KEY, columns="condition", values="surprisal")
            .reset_index()
            .rename_axis(columns=None)
        )
        bundles.append(
            {"slug": slug, "prompt_surp": ps, "surp_versions": sv, "manifest": None}
        )
    return bundles


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
def score_words(word_list, model, tokenizer, prompt=None, max_prompt_tokens=None):
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

    if prompt:
        prior_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
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

    input_ids = input_ids.to(DEVICE)
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
    """Per-word surprisal for every PoTeC text, scored ONE SENTENCE AT A TIME.

    Each sentence (``sent_index_in_text``) is rebuilt as its OWN sequence, so
    context resets at every sentence boundary and never carries across sentences.
    A sentence's first word therefore has no left context (``None``) and drops
    out; with a ``prompt`` the prior passage supplies that context instead. The
    text's final word is still dropped to match the reading-time cleaning.

    ``prompt`` may be None, a string, or a list of strings; each is a
    prior-reading passage joined to EACH sentence by the native document boundary
    (see ``score_words``). Given SEVERAL priors the sentence is scored under each
    and the per-word surprisals are averaged (``_mean_bits``), so the column is a
    mean over priors. ``max_prompt_tokens`` fixes the prior's context length.

    Returns columns: ``text_id``, ``word_index_in_text``, ``surprisal`` (the mean
    over priors when several are given).
    """
    prompts = _as_prompt_list(prompt)

    def _score(words: list[str]) -> list:
        if len(prompts) > 1:
            return _mean_bits(
                [
                    score_words(
                        words,
                        model,
                        tokenizer,
                        prompt=p,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    for p in prompts
                ]
            )
        # None / single prior: score once (keeps the no-prompt path unchanged).
        return score_words(
            words,
            model,
            tokenizer,
            prompt=prompts[0] if prompts else None,
            max_prompt_tokens=max_prompt_tokens,
        )

    rows = []
    for _text_id, text in words_df.sort_values(
        ["text_id", "word_index_in_text"]
    ).groupby("text_id", sort=False):
        last = text["word_index_in_text"].iloc[-1]  # text-final word (RT cleaning)
        for (text_id, _sent), sent in text.groupby(
            ["text_id", "sent_index_in_text"], sort=False
        ):
            words = sent["word"].fillna("").astype(str).tolist()
            bits = _score(words)
            for wi, b in zip(sent["word_index_in_text"].tolist(), bits):
                # b is None -> sentence first word (no context) or text first word.
                if b is None or wi == last:
                    continue
                rows.append((text_id, wi, b))

    return pd.DataFrame(rows, columns=WORD_KEY + ["surprisal"])


def stimuli_perplexity(
    surp_versions: pd.DataFrame, words_df: pd.DataFrame, tokenizer
) -> pd.DataFrame:
    """Token-level perplexity of every DAPT checkpoint on the PoTeC stimuli.

    Needs no extra forward passes: word surprisal is already the SUM of sub-token
    bits (``score_words``), so summing it over words and dividing by the matching
    sub-token count recovers the exact mean per-token surprisal, and
    ``2 ** mean`` is the same per-token perplexity the training manifest reports
    as ``exp(eval_loss)`` (perplexity is base-independent). Words absent from
    ``surp_versions`` (each text's first/last word) drop out of the token count
    too, keeping numerator and denominator aligned.

    Returns one row per ``domain`` (fine-tune) × ``index`` × ``text_domain``
    with the ``perplexity`` on that domain's stimulus texts (fig 1 dotted lines).
    """
    counts = []
    for text_id, text in words_df.sort_values(
        ["text_id", "word_index_in_text"]
    ).groupby("text_id", sort=False):
        words = text["word"].fillna("").astype(str).tolist()
        word_ids = tokenizer(words, is_split_into_words=True).word_ids(0)
        per_word: dict[int, int] = {}
        for wid in word_ids:
            if wid is not None:
                per_word[wid] = per_word.get(wid, 0) + 1
        counts += [
            (text_id, wi, per_word.get(k, 0))
            for k, wi in enumerate(text["word_index_in_text"])
        ]
    n_tok = pd.DataFrame(counts, columns=WORD_KEY + ["n_tokens"])
    dom = words_df[WORD_KEY + ["text_domain"]].drop_duplicates(WORD_KEY)
    g = (
        surp_versions.merge(n_tok, on=WORD_KEY)
        .merge(dom, on=WORD_KEY)
        .groupby(["domain", "index", "text_domain"])
        .agg(bits=("surprisal", "sum"), n_tokens=("n_tokens", "sum"))
        .reset_index()
    )
    g["perplexity"] = 2 ** (g["bits"] / g["n_tokens"])
    return g[["domain", "index", "text_domain", "perplexity"]]


def recompute_surprisal_over_checkpoints(
    words_df: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Recompute step-2 surprisal with each fine-tuned checkpoint.

    Concatenated surprisal table tagged with ``checkpoint`` / ``index`` / ``epoch``
    / ``domain``. ``index`` (0 = baseline) is the stable key for pairing domains.
    """
    frames = []
    # iterrows: the manifest has a column named "index" that itertuples would rename.
    for _, row in manifest.iterrows():
        model, tok = load_causal_lm(row.checkpoint)
        sup = compute_surprisal(words_df, model, tok)
        sup["checkpoint"] = row.checkpoint
        sup["index"] = row["index"]  # checkpoint index: stable across domains
        sup["epoch"] = row.epoch
        sup["domain"] = row.domain
        frames.append(sup)
    return pd.concat(frames, ignore_index=True)
