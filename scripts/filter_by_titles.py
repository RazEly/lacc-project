"""Filter wiki domain datasets to only titles listed in wiki_titles_translated.txt.

Reads the (hand-edited) translated titles file, extracts the German titles,
then filters each domain dataset to keep only matching articles and drops
duplicates (keeping first occurrence by title).

Usage:
    pixi run python scripts/filter_by_titles.py [--titles PATH] [--out-suffix SUFFIX]

Outputs: data/wiki_physics_filtered, data/wiki_biology_filtered  (by default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk

from src.config import DATA_DIR, DOMAIN_DIRS, DOMAINS


def load_titles(path: Path) -> dict[str, set[str]]:
    """Parse the translated titles file; return {domain: {title, ...}}."""
    titles: dict[str, set[str]] = {"physics": set(), "biology": set()}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "=== PHYSICS ===":
            current = "physics"
        elif line == "=== BIOLOGY ===":
            current = "biology"
        elif line and current and "|" in line:
            de_title = line.split("|", 1)[0].strip()
            if de_title:
                titles[current].add(de_title)
    return titles


def filter_domain(domain: str, keep: set[str], out: Path) -> None:
    ds = load_from_disk(str(DOMAIN_DIRS[domain]))
    print(f"[{domain}] loaded {len(ds)} docs")

    seen: set[str] = set()
    keep_idx: list[int] = []
    for i, title in enumerate(ds["title"]):
        if title in keep and title not in seen:
            keep_idx.append(i)
            seen.add(title)

    dropped_not_listed = len(ds) - sum(1 for t in ds["title"] if t in keep)
    dropped_dup = sum(1 for t in ds["title"] if t in keep) - len(keep_idx)

    filtered = ds.select(keep_idx)
    print(
        f"[{domain}] kept {len(filtered)} / {len(ds)} docs "
        f"(removed {dropped_not_listed} not in titles file, {dropped_dup} duplicates)"
    )
    if "num_words" in ds.column_names:
        print(f"  words {sum(ds['num_words']):,} -> {sum(filtered['num_words']):,}")
    filtered.save_to_disk(str(out))
    print(f"[{domain}] saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--titles",
        default=str(DATA_DIR / "wiki_titles_translated.txt"),
        help="path to (edited) wiki_titles_translated.txt",
    )
    ap.add_argument(
        "--out-suffix",
        default="_filtered",
        help="suffix appended to wiki_<domain> for output dirs",
    )
    args = ap.parse_args()

    titles_path = Path(args.titles)
    if not titles_path.exists():
        raise SystemExit(f"titles file not found: {titles_path}")

    domain_titles = load_titles(titles_path)
    for domain in DOMAINS:
        out = DATA_DIR / f"wiki_{domain}{args.out_suffix}"
        filter_domain(domain, domain_titles[domain], out)


if __name__ == "__main__":
    main()
