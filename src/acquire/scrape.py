"""Scrape German Wikipedia for the domain-adaptation corpora (the paper's approach).

Replicates Škrjanec & Demberg (2026), §"Training data for the adaptation to the
physics and biology domains": seed German-Wikipedia categories from the PoTeC
level-2 technical terms, walk their sub-categories, and scrape the plain text of
every member article. One corpus per domain (physics / biology), saved as an HF
dataset that ``modeling.finetune`` can train on directly.

Differences from the paper, kept deliberately simple:
  • categories are derived automatically from the term pages (the paper searched
    with the terms, then HAND-PICKED 32/35 categories); we keep every content
    category a seed term sits in, filtered of admin/maintenance categories.
  • Spektrum.de (the paper's second source, ~9k biology / ~3k physics articles) is
    left as a TODO — Wikipedia is the larger source (>15k / ~11k) and needs no
    site-specific HTML parsing.

Run:
    python -m src.acquire.scrape            # full scrape, both domains
    python -m src.acquire.scrape --test     # tiny smoke run (few terms, depth 1)
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter

import requests
import wikipediaapi
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.config import DATA_DIR, DEFAULT_MODEL, DOMAINS
from src.features.potec import load_word_features

LANG = "de"
# Wikimedia blocks the default urllib/requests UA; a descriptive UA is required.
USER_AGENT = "PoTeC-DAPT-research/0.1 (ely.raz@campus.technion.ac.il)"

# Scrape knobs (paper-scale defaults; --test shrinks them).
MAX_DEPTH = 2  # sub-category recursion depth from each seed category
MAX_TOKENS = 5_000_000  # per-domain token budget (DEFAULT_MODEL tokenizer); stop here
ARTICLE_POOL = 3_500  # cap on titles gathered before the token budget cuts the scrape
MIN_ARTICLE_CHARS = 200  # drop stubs / disambiguation pages
REQUEST_DELAY = 0.1  # polite pause between category/article fetches (s)

# Trailing reference/navigation sections — citation lists, not prose. Dropped
# whole (matched by German Wikipedia heading title, incl. common variants).
DROP_SECTIONS = frozenset({
    "Einzelnachweise", "Weblinks", "Literatur", "Siehe auch", "Quellen",
    "Fußnoten", "Anmerkungen", "Belege", "Weblink", "Nachweise",
})

# Math cruft left by the plain-text extractor when it renders <math> blocks
# (e.g. ``{\displaystyle f(x)=\frac{1}{2}}``). The paper does the same cleaning
# ("removal of math symbols"). Strip LaTeX control words + the braces they leave.
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_MATH_BRACE_RE = re.compile(r"[{}]")
_WS_RE = re.compile(r"[ \t]{2,}")

_TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Sub-token count under the DAPT base tokenizer (matches finetune's budget)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    return len(_TOKENIZER(text, truncation=False)["input_ids"])


# Output dirs — SEPARATE from data/domain_<domain> so a scrape never clobbers a
# corpus the DAPT checkpoints were trained on. Point DOMAIN_DIRS here to use them.
SCRAPE_DIRS = {d: DATA_DIR / f"wiki_{d}" for d in DOMAINS}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


def _wiki() -> wikipediaapi.Wikipedia:
    """German-Wikipedia client that returns full plain-text article bodies."""
    return wikipediaapi.Wikipedia(user_agent=USER_AGENT, language=LANG)


def level2_terms(domain: str) -> list[str]:
    """The PoTeC level-2 (expert) technical terms for one domain.

    Read from the stimulus word features (``is_expert_technical_term``), the exact
    seed the paper used. Deduplicated, longest first (specific compounds first).
    """
    wf = load_word_features()
    exp = wf[(wf["is_expert_technical_term"] == 1) & (wf["text_domain"] == domain)]
    terms = {
        w.strip() for w in exp["word"] if isinstance(w, str) and len(w.strip()) > 2
    }
    return sorted(terms, key=len, reverse=True)


def _is_content_category(cat_title: str) -> bool:
    """Keep topical categories; drop admin/maintenance ones.

    wikipediaapi yields keys like ``Kategorie:Quantenmechanik``. Maintenance
    categories carry a second colon (``Kategorie:Wikipedia:…``) or known admin
    tokens — those are not domain content.
    """
    body = cat_title.split(":", 1)[-1]
    if ":" in body:  # e.g. Kategorie:Wikipedia:Redundanz
        return False
    admin = ("Wikipedia", "Versteckte", "Vorlage", "Navigationsleiste", "Liste")
    return not any(tok in body for tok in admin)


def _search_title(term: str) -> str | None:
    """Nearest German-Wikipedia article title for a term (opensearch), or None.

    Tolerates transient non-200 / non-JSON responses (rate limits, blips): the
    term is simply skipped rather than aborting the whole scrape.
    """
    try:
        r = _SESSION.get(
            f"https://{LANG}.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": term,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            timeout=20,
        )
        r.raise_for_status()
        hits = r.json()[1]
    except (requests.RequestException, ValueError, IndexError):
        return None
    return hits[0] if hits else None


def seed_categories(terms: list[str], wiki, max_terms: int | None = None) -> list[str]:
    """Content categories the seed terms' articles belong to (frequency-ranked).

    For each term: its own article if it is a page title, else the nearest
    opensearch hit. Every content category of that article becomes a seed, ranked
    by how many terms land in it (shared categories = the domain's core topics).
    """
    cats: Counter[str] = Counter()
    picked = terms[: max_terms or len(terms)]
    for term in tqdm(picked, desc="seed cats", unit="term"):
        page = wiki.page(term)
        if not page.exists():
            alt = _search_title(term)
            page = wiki.page(alt) if alt else None
        if not page or not page.exists():
            continue
        for cat in page.categories:
            if _is_content_category(cat):
                cats[cat] += 1
        time.sleep(REQUEST_DELAY)
    return [c for c, _ in cats.most_common()]


def collect_articles(seed_cats, wiki, max_depth, pool_size) -> list[str]:
    """BFS the seed categories' sub-category tree; return member article titles.

    Namespace 0 members are articles; namespace 14 members are sub-categories,
    followed until ``max_depth``. Gathers up to ``pool_size`` titles — the token
    budget in ``fetch_texts`` is what actually sizes the corpus, so the pool is
    kept larger than needed. Interleaves members breadth-first so early titles are
    spread across seed categories, not exhausted from the first one.
    """
    seen_cat: set[str] = set()
    seen_art: set[str] = set()
    articles: list[str] = []
    frontier = [(c, 0) for c in seed_cats]
    bar = tqdm(total=pool_size, desc="pool titles", unit="art")
    while frontier and len(articles) < pool_size:
        cat, depth = frontier.pop(0)
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        for title, member in wiki.page(cat).categorymembers.items():
            if member.ns == wikipediaapi.Namespace.MAIN:
                if title not in seen_art:
                    seen_art.add(title)
                    articles.append(title)
                    bar.update(1)
                    if len(articles) >= pool_size:
                        break
            elif member.ns == wikipediaapi.Namespace.CATEGORY and depth < max_depth:
                frontier.append((title, depth + 1))
        time.sleep(REQUEST_DELAY)
    bar.close()
    return articles


def _section_text(section) -> str:
    """Recursive plain text of a section, minus ``DROP_SECTIONS`` subtrees."""
    if section.title in DROP_SECTIONS:
        return ""
    parts = [section.text, *(_section_text(sub) for sub in section.sections)]
    return "\n".join(p for p in parts if p)


def _clean_text(page) -> str:
    """Article prose: lead + body, minus reference sections and math cruft.

    ``wikipediaapi.text`` concatenates every section (incl. Einzelnachweise /
    Weblinks / Literatur — citation noise) and renders <math> as LaTeX residue.
    We rebuild from ``page.sections`` dropping the reference tail, then strip the
    LaTeX control words and their leftover braces.
    """
    parts = [page.summary, *(_section_text(s) for s in page.sections)]
    text = "\n\n".join(p for p in parts if p)
    text = _LATEX_CMD_RE.sub(" ", text)  # \displaystyle, \frac, \mathrm, ...
    text = _MATH_BRACE_RE.sub(" ", text)  # braces the math blocks leave behind
    return _WS_RE.sub(" ", text).strip()


def fetch_texts(titles, wiki, max_tokens) -> list[dict]:
    """Article bodies until the token budget is reached (short/missing dropped).

    Each kept row carries ``num_tokens`` (DEFAULT_MODEL sub-tokens). Fetching stops
    once the cumulative token count crosses ``max_tokens``, so the corpus is sized
    by tokens (comparable across domains), not article count.
    """
    rows = []
    total = 0
    bar = tqdm(total=max_tokens, desc="fetch text", unit="tok", unit_scale=True)
    for title in titles:
        page = wiki.page(title)
        text = _clean_text(page) if page.exists() else ""
        if len(text) >= MIN_ARTICLE_CHARS:
            n = _count_tokens(text)
            rows.append(
                {"title": title, "text": text, "source": "wikipedia", "num_tokens": n}
            )
            total += n
            bar.update(n)
            if total >= max_tokens:
                break
        time.sleep(REQUEST_DELAY)
    bar.close()
    return rows


def scrape_domain(
    domain: str,
    max_depth: int = MAX_DEPTH,
    max_tokens: int = MAX_TOKENS,
    pool_size: int = ARTICLE_POOL,
    max_terms: int | None = None,
) -> Dataset:
    """Scrape one domain's Wikipedia corpus (to a token budget) into a ``Dataset``."""
    wiki = _wiki()
    terms = level2_terms(domain)
    print(f"[{domain}] {len(terms)} level-2 seed terms")

    cats = seed_categories(terms, wiki, max_terms=max_terms)
    print(f"[{domain}] {len(cats)} seed content categories (top: {cats[:8]})")

    titles = collect_articles(cats, wiki, max_depth, pool_size)
    print(f"[{domain}] {len(titles)} article titles pooled (depth<= {max_depth})")

    rows = fetch_texts(titles, wiki, max_tokens)
    ds = Dataset.from_list([{**r, "domain": domain} for r in rows])
    total_tokens = sum(r["num_tokens"] for r in rows)
    print(f"[{domain}] kept {len(ds)} articles, {total_tokens:,} tokens")
    return ds


def main() -> None:
    test = "--test" in sys.argv
    depth = 1 if test else MAX_DEPTH
    tokens = 30_000 if test else MAX_TOKENS
    pool = 15 if test else ARTICLE_POOL
    terms = 4 if test else None
    doms = ("physics",) if test else DOMAINS

    for domain in doms:
        ds = scrape_domain(
            domain, max_depth=depth, max_tokens=tokens, pool_size=pool, max_terms=terms
        )
        if not test:
            out = SCRAPE_DIRS[domain]
            ds.save_to_disk(str(out))
            print(f"[{domain}] saved -> {out}")
        elif len(ds):
            print(
                f"\n--- {domain} sample ---\n{ds[0]['title']}: {ds[0]['text'][:300]!r}"
            )


if __name__ == "__main__":
    main()
