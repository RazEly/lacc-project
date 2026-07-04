"""Wide per-word surprisal cache — the CSV schema's single home.

One CSV for every model + checkpoint, persisting the expensive forwards so
reruns reload instead of re-scoring. One row per word (``WORD_KEY``), one
column per source named ``<prefix>_<what>`` (prefix per model, e.g. ``gpt`` /
``llama``):

    <p>_0                              baseline (un-adapted, checkpoint index 0)
    <p>_<i>_physics / <p>_<i>_biology  domain-adapted checkpoint i>=1
    <p>_prompt_physics / _biology      discipline-matched prompted baseline
    <p>_prompt_neutral                 off-domain scientific prompted control

Each model fills its own columns; a model already present is reused while
missing ones are computed and merged in (``merge_model``). ``build_wide`` folds
a run's long tables into these columns; ``bundle_from_cache`` reads them back
into the shapes the fitting stage expects — both directions live here so the
column convention has exactly one home.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DOMAINS, SURPRISAL_CACHE_PATH, WORD_KEY

# prompt_surp column (``s_prompt_*``) suffixes -> cache column ``<prefix>_prompt_*``.
# domains = reader-aligned domain prior; neutral = off-domain scientific prior
# (register-matched control from the german-commons off-domain pool).
_PROMPT_SUFFIXES = [*DOMAINS, "neutral"]


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
        for domain in DOMAINS:
            sel = surp_versions[
                (surp_versions["index"] == idx) & (surp_versions["domain"] == domain)
            ]
            wide = wide.merge(
                _col(sel, f"{prefix}_{idx}_{domain}"), on=WORD_KEY, how="outer"
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


def bundle_from_cache(
    cache: pd.DataFrame,
    slug: str,
    prefix: str,
    indices: list[int],
    checkpoint_steps: list[int],
) -> dict:
    """Rebuild a fit bundle straight from the wide cache — no model load, no finetune.

    Pulls only what the fitting stage needs: baseline surprisal, the prompted
    columns, and the ``indices`` checkpoint(s) for both domains (index ``i`` >= 1
    pairs to ``checkpoint_steps[i - 1]``; 0 = baseline). ``epoch`` in
    surp_versions is the checkpoint step (display only). ``manifest`` is None —
    perplexity diagnostics need the training manifests.
    """
    print(f"\n=== reload from cache: {slug} ({prefix}) ===")

    def _col(name: str, new: str) -> pd.DataFrame:
        return (
            cache[WORD_KEY + [name]]
            .dropna(subset=[name])
            .rename(columns={name: new})
            .reset_index(drop=True)
        )

    prompt_surp = _col(f"{prefix}_prompt_physics", "s_prompt_physics")
    prompt_surp = prompt_surp.merge(
        _col(f"{prefix}_prompt_biology", "s_prompt_biology"), on=WORD_KEY
    )
    # neutral prior is optional: older caches predate it. Merge only if present.
    if f"{prefix}_prompt_neutral" in cache.columns:
        prompt_surp = prompt_surp.merge(
            _col(f"{prefix}_prompt_neutral", "s_prompt_neutral"), on=WORD_KEY
        )

    # Long per-checkpoint table for model_comparison: index 0 (baseline, shared by
    # both domains) + the requested checkpoint(s), each domain.
    frames = []
    for idx in [0, *indices]:
        step = 0 if idx == 0 else checkpoint_steps[idx - 1]
        for domain in DOMAINS:
            col = f"{prefix}_0" if idx == 0 else f"{prefix}_{idx}_{domain}"
            sv = _col(col, "surprisal")
            sv["index"], sv["domain"], sv["epoch"] = idx, domain, step
            frames.append(sv)
    surp_versions = pd.concat(frames, ignore_index=True)

    return {
        "slug": slug,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "manifest": None,
    }
