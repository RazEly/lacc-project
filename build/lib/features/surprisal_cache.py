"""Wide per-word surprisal cache (one CSV for every model + checkpoint).

Surprisal forwards are the pipeline's expensive step; this persists them so reruns
reload instead of re-scoring. One row per word (``text_id``, ``word_index_in_text``)
and one column per surprisal source, named ``<prefix>_<what>`` (prefix from
``config.MODEL_PREFIX``, e.g. ``gpt`` / ``llama``):

    <p>_0                          baseline (un-adapted, checkpoint index 0)
    <p>_<i>_phys / <p>_<i>_bio     domain-adapted checkpoint i>=1 (physics/biology)
    <p>_prompt_phys / _bio         discipline-matched prompted baseline
    <p>_prompt_<phys|bio>_<ug|grad>  field x level prompted baseline

Each model contributes its own columns, so the cache is filled per model and a
model already present is reused while missing ones are computed and merged in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import SURPRISAL_CACHE_PATH, WORD_KEY

_DOMAIN_SHORT = {"physics": "phys", "biology": "bio"}


def load_cache(path: Path = SURPRISAL_CACHE_PATH) -> pd.DataFrame | None:
    """Return the cached wide table, or ``None`` when no file exists yet."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def has_model(cache: pd.DataFrame | None, prefix: str) -> bool:
    """True when ``cache`` already holds this model's columns (baseline present)."""
    return cache is not None and f"{prefix}_0" in cache.columns


def _prompt_map(prefix: str) -> dict[str, str]:
    """prompt_surp column (``s_prompt_*``) -> cache column (``<prefix>_prompt_*``)."""
    cols = [
        "s_prompt_phys",
        "s_prompt_bio",
        "s_prompt_phys_ug",
        "s_prompt_phys_grad",
        "s_prompt_bio_ug",
        "s_prompt_bio_grad",
    ]
    return {c: f"{prefix}_{c[2:]}" for c in cols}  # drop the "s_" head


def build_wide(
    prefix: str, surp_versions: pd.DataFrame, prompt_surp: pd.DataFrame
) -> pd.DataFrame:
    """Fold one model's ``surp_versions`` + ``prompt_surp`` into wide columns."""
    base_index = surp_versions["index"].min()
    base = surp_versions[surp_versions["index"] == base_index]
    wide = base.drop_duplicates(WORD_KEY)[WORD_KEY + ["surprisal"]].rename(
        columns={"surprisal": f"{prefix}_0"}
    )
    for idx in sorted(surp_versions["index"].unique()):
        if idx == base_index:
            continue
        for domain, short in _DOMAIN_SHORT.items():
            sel = surp_versions[
                (surp_versions["index"] == idx) & (surp_versions["domain"] == domain)
            ]
            col = f"{prefix}_{idx}_{short}"
            wide = wide.merge(
                sel[WORD_KEY + ["surprisal"]].rename(columns={"surprisal": col}),
                on=WORD_KEY,
                how="outer",
            )
    prompts = prompt_surp.rename(columns=_prompt_map(prefix))
    return wide.merge(prompts, on=WORD_KEY, how="outer")


def merge_model(cache: pd.DataFrame | None, wide: pd.DataFrame) -> pd.DataFrame:
    """Add/replace one model's columns in the cache and return the combined table."""
    if cache is None:
        return wide
    new_cols = [c for c in wide.columns if c not in WORD_KEY]
    cache = cache.drop(columns=[c for c in new_cols if c in cache.columns])
    return cache.merge(wide, on=WORD_KEY, how="outer")


def save_cache(cache: pd.DataFrame, path: Path = SURPRISAL_CACHE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)


# ── reload: reconstruct the pieces the pipeline consumes ─────────────────────
def baseline_surp(cache: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """The baseline (prompt=None, un-adapted) surprisal table for step 5."""
    col = f"{prefix}_0"
    return (
        cache[WORD_KEY + [col]]
        .dropna(subset=[col])
        .rename(columns={col: "surprisal"})
        .reset_index(drop=True)
    )


def prompt_surp(cache: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """The checkpoint-independent prompted columns (``s_prompt_*``)."""
    rev = {v: k for k, v in _prompt_map(prefix).items()}
    cols = list(rev)
    return (
        cache[WORD_KEY + cols]
        .dropna(subset=cols)
        .rename(columns=rev)
        .reset_index(drop=True)
    )


def surp_versions(
    cache: pd.DataFrame, prefix: str, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Rebuild the long per-checkpoint table (``recompute_surprisal_over_checkpoints``).

    ``manifest`` supplies ``checkpoint`` / ``index`` / ``epoch`` / ``domain`` per row;
    surprisal is read from the matching cache column (baseline shared across domains).
    """
    frames = []
    for _, row in manifest.iterrows():
        idx = int(row["index"])
        col = (
            f"{prefix}_0" if idx == 0 else f"{prefix}_{idx}_{_DOMAIN_SHORT[row.domain]}"
        )
        sup = (
            cache[WORD_KEY + [col]]
            .dropna(subset=[col])
            .rename(columns={col: "surprisal"})
        )
        sup["checkpoint"] = row.checkpoint
        sup["index"] = idx
        sup["epoch"] = row.epoch
        sup["domain"] = row.domain
        frames.append(sup)
    return pd.concat(frames, ignore_index=True)
