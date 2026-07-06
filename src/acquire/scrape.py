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
    python -m src.acquire.scrape                    # full scrape, both domains
    python -m src.acquire.scrape --domain biology   # one domain only
    python -m src.acquire.scrape --test             # tiny smoke run (few terms, depth 1)
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

import wikipediaapi
from datasets import Dataset
from tqdm.auto import tqdm

from src.config import DATA_DIR, DOMAINS
from src.features.potec import load_word_features

LANG = "de"
USER_AGENT = "PoTeC-DAPT-research/0.1 (ely.raz@campus.technion.ac.il)"

# Scrape knobs (paper-scale defaults; --test shrinks them).
MAX_DEPTH = 2  # sub-category recursion depth from each seed category
MAX_WORDS = 3_000_000  # per-domain word budget (whitespace words); stop here
ARTICLE_POOL = 9_000  # cap on titles gathered before the word budget cuts the scrape
MIN_ARTICLE_CHARS = 1000  # drop stubs / disambiguation / infobox-only pages
REQUEST_DELAY = 0.1  # polite pause between category-listing fetches (s)
FETCH_WORKERS = 4  # parallel article-text fetches; lower = fewer 429s under throttle
FETCH_RETRIES = 3  # per-article retries on transient API failure (429/network)
RETRY_BACKOFF = 1.5  # seconds, exponential base between retries

# Trailing reference/navigation sections — citation lists, not prose. Dropped
# whole (matched by German Wikipedia heading title, incl. common variants).
DROP_SECTIONS = frozenset(
    {
        "Einzelnachweise",
        "Weblinks",
        "Literatur",
        "Siehe auch",
        "Quellen",
        "Fußnoten",
        "Anmerkungen",
        "Belege",
        "Weblink",
        "Nachweise",
    }
)

_WS_RE = re.compile(r"[ \t]{2,}")
# Category-body substrings that mark a NON-content grouping: admin/maintenance,
# disambiguation, and chemical hazard/regulatory classes (infobox-stub farms).
DROP_CAT_TOKENS = (
    "Wikipedia",
    "Versteckte",
    "Vorlage",
    "Navigationsleiste",
    "Liste",
    "Begriffsklärung",
    "CLP",
    "REACH",
    "SVHC",
    "CMR",
)
# Standalone word "Stoff" — the chemical hazard cats (Gefährlicher Stoff, CMR-Stoff,
# …). Word boundary spares "Stoffwechsel…"/"Stoffgruppe" (metabolism prose we keep).
_STOFF_RE = re.compile(r"\bStoff\b")


class _TextExtractor(HTMLParser):
    """Visible text from TextExtracts HTML, dropping ``<math>``/``<style>`` subtrees.

    With ``extract_format=HTML`` the API renders formulas as ``<math>`` (MathML)
    blocks; we skip everything inside them — the paper's "removal of math
    symbols" — so no LaTeX/unicode residue leaks into the corpus.
    """

    _SKIP = {"math", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def _count_words(text: str) -> int:
    """Whitespace word count — cheap corpus-size proxy, no tokenizer load needed."""
    return len(text.split())


# Output dirs — SEPARATE from data/domain_<domain> so a scrape never clobbers a
# corpus the DAPT checkpoints were trained on. Point DOMAIN_DIRS here to use them.
SCRAPE_DIRS = {d: DATA_DIR / f"wiki_{d}" for d in DOMAINS}


def _wiki() -> wikipediaapi.Wikipedia:
    """German-Wikipedia client; HTML extracts so ``<math>`` can be dropped cleanly."""
    return wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language=LANG,
        extract_format=wikipediaapi.ExtractFormat.HTML,
    )


# Standard German BSc curriculum subjects per domain — added to the PoTeC level-2
# seeds so the scrape also anchors on whole-course topics (each is a real German
# Wikipedia article whose categories are broad, prose-rich domain cores, not the
# molecule/substance stub-farms the expert terms land in). Sourced from the Goethe-
# Uni Frankfurt B.Sc. Physik and Uni Bonn B.Sc. Biologie module handbooks.
CURRICULUM_TERMS = {
    "physics": [
        "Experimentalphysik",
        "Theoretische Physik",
        "Klassische Mechanik",
        "Elektrodynamik",
        "Thermodynamik",
        "Statistische Physik",
        "Quantenmechanik",
        "Optik",
        "Atomphysik",
        "Kernphysik",
        "Teilchenphysik",
        "Festkörperphysik",
        "Kondensierte Materie",
        "Astrophysik",
        "Kosmologie",
        "Relativitätstheorie",
        "Elektromagnetismus",
        "Plasmaphysik",
        "Biophysik",
        "Quantenfeldtheorie",
        "Spektroskopie",
        "Halbleiterphysik",
    ],
    "biology": [
        "Zellbiologie",
        "Zoologie",
        "Botanik",
        "Genetik",
        "Mikrobiologie",
        "Biochemie",
        "Ökologie",
        "Evolution",
        "Physiologie",
        "Molekularbiologie",
        "Entwicklungsbiologie",
        "Neurobiologie",
        "Immunbiologie",
        "Verhaltensbiologie",
        "Biodiversität",
        "Anatomie",
        "Pflanzenphysiologie",
        "Tierphysiologie",
        "Bioinformatik",
        "Paläontologie",
        "Biologische Systematik",
        "Histologie",
    ],
}


def level2_terms(domain: str) -> list[str]:
    """The PoTeC expert technical terms for one domain."""
    wf = load_word_features()
    exp = wf[(wf["is_expert_technical_term"] == 1) & (wf["text_domain"] == domain)]
    terms = {
        w.strip() for w in exp["word"] if isinstance(w, str) and len(w.strip()) > 2
    }
    return sorted(terms, key=len, reverse=True)


def _is_content_category(cat_title: str) -> bool:
    """Keep topical categories; drop admin/maintenance and substance-stub ones.

    wikipediaapi yields keys like ``Kategorie:Quantenmechanik``. A nested namespace
    (``Kategorie:Wikipedia:…``) or a ``DROP_CAT_TOKENS`` / ``_STOFF_RE`` hit marks a
    non-content grouping — maintenance, disambiguation, or a chemical hazard class
    whose members are infobox-only substance stubs (~40 words of prose). Biology's
    seed terms are molecules, so their articles sit in exactly those stub-farms.
    """
    body = cat_title.split(":", 1)[-1]
    if ":" in body:  # nested namespace, e.g. Kategorie:Wikipedia:Redundanz
        return False
    return not (any(tok in body for tok in DROP_CAT_TOKENS) or _STOFF_RE.search(body))


def _search_title(term: str, wiki) -> str | None:
    """Nearest German-Wikipedia article title for a term"""
    try:
        res = wiki.search(term, limit=1)
    except wikipediaapi.WikipediaException:
        return None
    return next(iter(res.pages), None)


def _is_article_title(title: str) -> bool:
    """Drop list / disambiguation pages (namespace 0, but not prose articles)."""
    return not title.startswith("Liste") and "(Begriffsklärung)" not in title


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
            alt = _search_title(term, wiki)
            page = wiki.page(alt) if alt else None
        if not page or not page.exists():
            continue
        for cat in page.categories:
            if _is_content_category(cat):
                cats[cat] += 1
        time.sleep(REQUEST_DELAY)
    return [c for c, _ in cats.most_common()]


def collect_articles(seed_cats, wiki, max_depth, pool_size) -> list:
    """BFS the seed categories' sub-category tree; return member article pages.

    Namespace 0 members are articles; namespace 14 members are sub-categories,
    followed until ``max_depth`` (``cmnamespace="0|14"`` filters the rest out at
    the API). Returns the ``WikipediaPage`` stubs themselves so ``fetch_texts``
    reuses them instead of re-resolving each title. Gathers up to ``pool_size`` —
    the word budget in ``fetch_texts`` is what actually sizes the corpus, so the
    pool is kept larger than needed. Interleaves members breadth-first so early
    articles spread across seed categories, not exhausted from the first one.
    """
    seen_cat: set[str] = set()
    seen_art: set[str] = set()
    articles: list = []
    frontier = [(c, 0) for c in seed_cats]
    bar = tqdm(total=pool_size, desc="pool titles", unit="art")
    while frontier and len(articles) < pool_size:
        cat, depth = frontier.pop(0)
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        members = wiki.categorymembers(wiki.page(cat), cmnamespace="0|14")
        for title, member in members.items():
            if member.ns == wikipediaapi.Namespace.MAIN:
                if title not in seen_art and _is_article_title(title):
                    seen_art.add(title)
                    articles.append(member)
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
    """Article prose: lead + body, minus reference sections and math.

    ``page.summary``/``sections[].text`` are HTML extracts. We rebuild from the
    sections dropping the reference tail (Einzelnachweise / Weblinks / Literatur —
    citation noise), then strip HTML to text with ``<math>`` blocks removed.
    """
    parts = [page.summary, *(_section_text(s) for s in page.sections)]
    text = _html_to_text("\n\n".join(p for p in parts if p))
    return _WS_RE.sub(" ", text).strip()


def _fetch_one(page) -> dict | None:
    """Resolve + clean one article and return a row dict"""
    for attempt in range(FETCH_RETRIES):
        try:
            if not page.exists():
                return None
            text = _clean_text(page)  # accessing content resolves any redirect
            break
        except wikipediaapi.WikipediaException:
            if attempt == FETCH_RETRIES - 1:
                return None  # give up on this article, keep the scrape alive
            time.sleep(RETRY_BACKOFF * (2**attempt))
    if len(text) < MIN_ARTICLE_CHARS or not _is_article_title(page.title):
        return None
    return {
        "title": page.title,
        "text": text,
        "source": "wikipedia",
        "num_words": _count_words(text),
    }


def fetch_texts(pages, max_words) -> list[dict]:
    """Article bodies until the word budget is reached (short/missing dropped).

    ``pages`` are the ``WikipediaPage`` stubs from ``collect_articles`` (reused,
    not re-resolved). Fetched ``FETCH_WORKERS`` at a time in a thread pool; results
    are consumed with ``as_completed`` so the bar advances per FINISHED article,
    not in submission order — one slow/large page no longer freezes progress while
    other workers are done. The word budget is checked as rows land and fetching
    stops (remaining futures cancelled) once it crosses ``max_words``. The corpus
    is sized by words (comparable across domains), not article count.

    Also reports ``kept``/``skipped`` so a stalled-looking run is diagnosable: a
    high skip count with words far below budget means the title pool ran dry
    (redirects / stubs / list pages), not the network.
    """
    rows: list[dict] = []
    total = 0
    kept = skipped = 0
    bar = tqdm(total=max_words, desc="fetch text", unit="word", unit_scale=True)
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, p): p for p in pages}
        try:
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)
                    kept += 1
                    total += row["num_words"]
                    bar.update(row["num_words"])
                bar.set_postfix(kept=kept, skip=skipped, refresh=False)
                if total >= max_words:
                    break
        finally:
            for fut in futures:
                fut.cancel()
    bar.close()
    print(f"    fetched: kept={kept} skipped={skipped} words={total:,}")
    return rows


def scrape_domain(
    domain: str,
    max_depth: int = MAX_DEPTH,
    max_words: int = MAX_WORDS,
    pool_size: int = ARTICLE_POOL,
    max_terms: int | None = None,
) -> Dataset:
    """Scrape one domain's Wikipedia corpus (to a word budget) into a ``Dataset``."""
    wiki = _wiki()
    expert = level2_terms(domain)
    terms = expert + CURRICULUM_TERMS[domain]
    print(
        f"[{domain}] {len(terms)} seed terms "
        f"({len(expert)} level-2 + {len(CURRICULUM_TERMS[domain])} curriculum)"
    )

    cats = seed_categories(terms, wiki, max_terms=max_terms)
    print(f"[{domain}] {len(cats)} seed content categories (top: {cats[:8]})")

    pages = collect_articles(cats, wiki, max_depth, pool_size)
    print(f"[{domain}] {len(pages)} article titles pooled (depth<= {max_depth})")

    rows = fetch_texts(pages, max_words)
    ds = Dataset.from_list([{**r, "domain": domain} for r in rows])
    total_words = sum(r["num_words"] for r in rows)
    print(f"[{domain}] kept {len(ds)} articles, {total_words:,} words")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape German Wikipedia domain corpora.")
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    test = args.test
    depth = 1 if test else MAX_DEPTH
    words = 30_000 if test else MAX_WORDS
    pool = 15 if test else ARTICLE_POOL
    terms = 4 if test else None
    if args.domain:
        doms = (args.domain,)
    else:
        doms = ("physics",) if test else DOMAINS

    for domain in doms:
        ds = scrape_domain(
            domain, max_depth=depth, max_words=words, pool_size=pool, max_terms=terms
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
