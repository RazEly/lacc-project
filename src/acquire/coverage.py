"""Expert-term coverage of a domain corpus (data quality check for DAPT).

Reports, per corpus, how many PoTeC level-2 (expert) technical terms the corpus
actually contains — the metric that decides whether DAPT can lower PoTeC surprisal
(see notes: term-targeted scrape > doc-classification selection). Density
(hits per million tokens) matters more than raw counts when corpora differ in size.

Run:
    python -m src.acquire.coverage                       # default corpora, physics
    python -m src.acquire.coverage --domain biology
    python -m src.acquire.coverage data/wiki_physics data/domain_physics
"""

from __future__ import annotations

import argparse
import re

from datasets import load_from_disk

from src.acquire.scrape import SCRAPE_DIRS, level2_terms
from src.config import DOMAIN_DIRS, DOMAINS

# German inflection tails stripped for stem coverage (longest first).
_SUFFIXES = ("enaufspaltungen", "ungen", "en", "es", "er", "e", "n", "s")


def _stem(term: str) -> str:
    t = term.lower()
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 5:
            return t[: -len(suf)]
    return t


def analyze(path: str, terms: list[str]) -> dict:
    """Coverage stats for one corpus dir against a term list."""
    d = load_from_disk(path)
    corpus = "\n".join(list(d["text"])).lower()
    tokens = sum(list(d["num_tokens"])) if "num_tokens" in d.column_names else len(corpus) // 4
    exact = sum(1 for t in terms if re.search(rf"\b{re.escape(t.lower())}\b", corpus))
    stem = sum(1 for t in terms if _stem(t) in corpus or _stem(t)[:7] in corpus)
    hits = sum(len(re.findall(rf"\b{re.escape(t.lower())}\b", corpus)) for t in terms)
    absent = [t for t in terms if _stem(t) not in corpus and _stem(t)[:7] not in corpus]
    return {
        "rows": len(d),
        "tokens": tokens,
        "exact": exact,
        "stem": stem,
        "hits": hits,
        "density": round(hits / (tokens / 1e6), 1) if tokens else 0.0,
        "absent": absent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpora", nargs="*", help="corpus dirs (default: wiki_ + domain_ for --domain)")
    ap.add_argument("--domain", default="physics", choices=DOMAINS)
    args = ap.parse_args()

    terms = level2_terms(args.domain)
    corpora = args.corpora or [
        str(SCRAPE_DIRS[args.domain]),
        str(DOMAIN_DIRS[args.domain]),
    ]

    print(f"{len(terms)} level-2 expert terms for '{args.domain}'\n")
    print(f"{'corpus':30} {'rows':>5} {'tokens':>10} {'exact':>7} {'stem':>7} {'hits':>6} {'hits/Mtok':>9}")
    for path in corpora:
        try:
            r = analyze(path, terms)
        except FileNotFoundError:
            print(f"{path:30} MISSING")
            continue
        print(
            f"{path:30} {r['rows']:>5} {r['tokens']:>10} "
            f"{r['exact']:>4}/{len(terms)} {r['stem']:>4}/{len(terms)} "
            f"{r['hits']:>6} {r['density']:>9}"
        )
        if r["absent"]:
            print(f"    absent (stem): {', '.join(sorted(r['absent']))}")


if __name__ == "__main__":
    main()
