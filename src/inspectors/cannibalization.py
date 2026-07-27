from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Common stopwords removed before topic comparison
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "this", "that",
    "these", "those", "it", "its", "we", "you", "they", "he", "she",
    "not", "no", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "now", "how", "when", "where", "which", "who",
    "what", "why", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "out", "off", "up", "down",
    "then", "here", "there", "if", "as", "while", "until", "because",
    "since", "although", "though", "whether", "without",
    # Japanese stopwords
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
    "ある", "いる", "する", "から", "など", "まで", "として", "より",
    "この", "その", "あの", "これ", "それ", "あれ", "ここ", "そこ",
}

# Keywords that suggest the page is a category/taxonomy page
TAXONOMY_PATH_MARKERS = [
    "/category/", "/categories/", "/tag/", "/tags/", "/topic/", "/topics/",
    "/label/", "/labels/", "/type/", "/types/", "/collection/", "/collections/",
]

# Blog-like path markers
BLOG_PATH_MARKERS = [
    "/blog/", "/news/", "/article/", "/articles/", "/post/", "/posts/",
    "/insights/", "/resources/", "/guide/", "/guides/",
]


class CannibalizationDetector(BaseInspector):
    """Detect content cannibalization: multiple pages targeting the same keywords.

    Checks:
    - Title similarity (near-duplicate titles)
    - Topic overlap (title + H1 + keyword set)
    - URL granularity (category vs individual post competing)
    """

    inspector_name = "cannibalization"

    # Thresholds
    TITLE_SIMILARITY_THRESHOLD = 0.75   # titles ≥75% similar → high risk
    TITLE_SIMILARITY_WARN = 0.60        # titles ≥60% similar → moderate risk
    KEYWORD_OVERLAP_THRESHOLD = 0.60    # keyword set overlaps ≥60% → topic conflict

    def __init__(self) -> None:
        self._page_data: list[dict] = []
        self._already_reported: set[str] = set()
        self._soup_cache: dict[str, "BeautifulSoup"] = {}

    async def setup(self) -> None:
        self._soup_cache.clear()

    async def teardown(self) -> None:
        self._soup_cache.clear()

    def set_page_data(self, pages: list[dict]) -> None:
        """Pre-parse all page HTMLs once to avoid O(N²) BeautifulSoup parses."""
        self._page_data = pages
        self._soup_cache.clear()
        for p in pages:
            url = p.get("url", "")
            html = p.get("html_content") or ""
            if html:
                self._soup_cache[url] = BeautifulSoup(html, "html.parser")

    def set_page_data(self, pages: list[dict]) -> None:
        """Receive all crawled page metadata for cross-site comparison.

        Each dict should have:
            url: str, title: str | None, html_content: str | None
        """
        self._page_data = pages

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if len(self._page_data) < 2:
            return findings

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract this page's signals
        title = self._extract_title(soup)
        h1 = self._extract_h1(soup)
        body_text = self._extract_body_text(soup)
        keywords = self._extract_keywords(title, h1, body_text)

        if not title and not h1:
            return findings

        this_topic = self._build_topic_vector(title, h1, keywords)

        for other in self._page_data:
            other_url = other.get("url", "")
            if other_url == url:
                continue

            pair_key = self._pair_key(url, other_url)
            if pair_key in self._already_reported:
                continue

            other_soup = self._soup_cache.get(other_url)

            other_title = other.get("title") or (
                self._extract_title(other_soup) if other_soup else ""
            )
            other_h1 = self._extract_h1(other_soup) if other_soup else ""
            other_body = self._extract_body_text(other_soup) if other_soup else ""
            other_keywords = self._extract_keywords(other_title, other_h1, other_body)
            other_topic = self._build_topic_vector(other_title, other_h1, other_keywords)

            pair_findings: list[RawFinding] = []

            # Check 1: Title similarity
            title_sim = self._similarity(title, other_title) if title and other_title else 0.0
            if title_sim >= self.TITLE_SIMILARITY_THRESHOLD:
                pair_findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="cannibalization_title_duplicate",
                    description=(
                        f"Near-identical title with '{other_url}': "
                        f"'{title[:80]}' vs '{other_title[:80]}' "
                        f"({title_sim:.0%} similar). These pages compete for "
                        f"the same query intent."
                    ),
                    current_value=title,
                    suggested_value=(
                        f"Differentiate one page's title; consider merging content "
                        f"or using canonical to designate the primary version."
                    ),
                    raw_metadata={
                        "other_url": other_url,
                        "similarity": round(title_sim, 3),
                        "this_title": title,
                        "other_title": other_title,
                    },
                ))
            elif title_sim >= self.TITLE_SIMILARITY_WARN:
                pair_findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="cannibalization_title_similar",
                    description=(
                        f"Similar title with '{other_url}': "
                        f"'{title[:80]}' vs '{other_title[:80]}' "
                        f"({title_sim:.0%} similar). Risk of partial cannibalization."
                    ),
                    current_value=title,
                    suggested_value=(
                        "Ensure each page targets a distinct keyword intent; "
                        "consolidate if they cover the same topic."
                    ),
                    raw_metadata={
                        "other_url": other_url,
                        "similarity": round(title_sim, 3),
                    },
                ))

            # Check 2: Topic vector overlap (keyword + semantic overlap)
            topic_overlap = self._topic_overlap(this_topic, other_topic)
            if title_sim >= 0.40 and topic_overlap >= self.KEYWORD_OVERLAP_THRESHOLD:
                pair_findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="cannibalization_topic_overlap",
                    description=(
                        f"Significant topic overlap with '{other_url}': "
                        f"{topic_overlap:.0%} keyword overlap. "
                        f"Both pages target similar search intent — "
                        f"consider consolidating or differentiating."
                    ),
                    current_value=f"{topic_overlap:.0%} overlap",
                    suggested_value=(
                        "Merge into one comprehensive page (stronger ranking signal) "
                        "OR sharpen each page's focus to distinct subtopics."
                    ),
                    raw_metadata={
                        "other_url": other_url,
                        "topic_overlap": round(topic_overlap, 3),
                        "shared_keywords": sorted(
                            set(this_topic["keywords"][:10]) &
                            set(other_topic["keywords"][:10])
                        ),
                    },
                ))

            # Check 3: URL taxonomy cannibalization
            # Blog post URL slug matches a category/tag slug
            url_conflict = self._check_url_conflict(url, other_url, title, other_title)
            if url_conflict:
                pair_findings.append(url_conflict)

            # Mark this pair as handled
            self._already_reported.add(pair_key)

            findings.extend(pair_findings)

        return findings

    # ── Extraction helpers ──────────────────────────────────────────

    @staticmethod
    def _extract_title(soup: BeautifulSoup | None) -> str:
        if not soup:
            return ""
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_h1(soup: BeautifulSoup | None) -> str:
        if not soup:
            return ""
        tag = soup.find("h1")
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_body_text(soup: BeautifulSoup | None) -> str:
        if not soup:
            return ""
        body = soup.find("body")
        if not body:
            return ""
        for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return body.get_text(separator=" ", strip=True)

    def _extract_keywords(
        self, title: str, h1: str, body_text: str,
    ) -> list[str]:
        """Extract meaningful keywords from page signals, weighted by importance."""
        combined = f"{title} {title} {h1} {h1} {body_text[:2000]}"  # title/h1 weighted 2x
        words = re.findall(r"[a-zA-Z一-鿿]{2,}", combined.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        freq = Counter(filtered)
        return [w for w, _ in freq.most_common(15)]

    def _build_topic_vector(
        self, title: str, h1: str, keywords: list[str],
    ) -> dict:
        """Build a lightweight topic vector from page signals."""
        # Combine title + H1 into a short "topic fingerprint"
        topic_words = set(
            re.findall(r"[a-zA-Z一-鿿]{2,}", f"{title} {h1}".lower())
        )
        topic_words -= STOPWORDS
        return {
            "title": title,
            "h1": h1,
            "topic_words": topic_words,
            "keywords": keywords,
        }

    # ── Comparison helpers ───────────────────────────────────────────

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """SequenceMatcher-based similarity between two strings."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    @staticmethod
    def _topic_overlap(a: dict, b: dict) -> float:
        """Compute keyword + topic-word overlap ratio between two page vectors."""
        kw_a = set(a.get("keywords", [])[:10])
        kw_b = set(b.get("keywords", [])[:10])
        tw_a = a.get("topic_words", set())
        tw_b = b.get("topic_words", set())

        all_a = kw_a | tw_a
        all_b = kw_b | tw_b

        if not all_a or not all_b:
            return 0.0

        intersection = all_a & all_b
        union = all_a | all_b
        return len(intersection) / len(union) if union else 0.0

    def _check_url_conflict(
        self, url_a: str, url_b: str, title_a: str, title_b: str,
    ) -> RawFinding | None:
        """Check for URL-level cannibalization patterns.

        A common pattern: /blog/post-about-widgets vs /category/widgets
        Both target 'widgets' — the category page cannibalizes the blog post.
        """
        path_a = urlparse(url_a).path.lower().rstrip("/")
        path_b = urlparse(url_b).path.lower().rstrip("/")

        # Determine if one is a taxonomy page and the other is a detail page
        a_is_tax = any(m in path_a for m in TAXONOMY_PATH_MARKERS)
        b_is_tax = any(m in path_b for m in TAXONOMY_PATH_MARKERS)

        if a_is_tax == b_is_tax:
            return None  # same type, not a taxonomy-vs-detail conflict

        tax_url = url_a if a_is_tax else url_b
        detail_url = url_b if a_is_tax else url_a
        tax_path = path_a if a_is_tax else path_b
        detail_path = path_b if a_is_tax else path_a
        tax_title = title_a if a_is_tax else title_b
        detail_title = title_b if a_is_tax else title_a

        # Extract the key slug segment (last meaningful path segment)
        tax_slug = self._last_segment(tax_path)
        detail_slug = self._last_segment(detail_path)

        if not tax_slug or not detail_slug:
            return None

        # If the category slug matches part of the post slug, it's a risk
        if tax_slug in detail_slug or detail_slug in tax_slug:
            return RawFinding(
                url=url_a, inspector=self.inspector_name,
                category="cannibalization_taxonomy_vs_detail",
                description=(
                    f"URL taxonomy conflict: '{tax_url}' (category/tag) and "
                    f"'{detail_url}' (detail page) both target '{tax_slug}'. "
                    f"The taxonomy page may cannibalize the detail page."
                ),
                current_value=f"taxonomy='{tax_slug}' ↔ detail='{detail_slug}'",
                suggested_value=(
                    f"Noindex the taxonomy page for '{tax_slug}', or ensure the "
                    f"detail page has clearly distinct and more specific targeting "
                    f"(title, H1, content depth)."
                ),
                raw_metadata={
                    "taxonomy_url": tax_url,
                    "detail_url": detail_url,
                    "shared_slug": tax_slug,
                },
            )

        # Also check: do the titles suggest topic overlap between tax & detail?
        title_sim = self._similarity(tax_title, detail_title)
        if title_sim >= 0.50:
            return RawFinding(
                url=url_a, inspector=self.inspector_name,
                category="cannibalization_taxonomy_title_overlap",
                description=(
                    f"Taxonomy page '{tax_url}' and detail page '{detail_url}' "
                    f"have similar titles ({title_sim:.0%}). "
                    f"Search engines may rank the wrong page."
                ),
                current_value=f"tax='{tax_title[:80]}' vs detail='{detail_title[:80]}'",
                suggested_value=(
                    "Prefix taxonomy page titles with 'Category:' or 'All', "
                    "or use more distinct naming."
                ),
                raw_metadata={
                    "taxonomy_url": tax_url,
                    "detail_url": detail_url,
                    "title_similarity": round(title_sim, 3),
                },
            )

        return None

    @staticmethod
    def _last_segment(path: str) -> str:
        """Get the last meaningful path segment."""
        segments = [s for s in path.split("/") if s]
        # Skip known taxonomy markers
        skip = {"category", "categories", "tag", "tags", "topic", "topics",
                 "label", "labels", "type", "types", "blog", "news", "article",
                 "articles", "post", "posts", "collection", "collections",
                 "page", "p"}
        for seg in reversed(segments):
            if seg.lower() not in skip:
                return seg.lower()
        return segments[-1] if segments else ""

    # ── Utility ─────────────────────────────────────────────────────

    @staticmethod
    def _pair_key(url_a: str, url_b: str) -> str:
        """Deterministic key for an unordered pair of URLs."""
        pair = sorted([url_a, url_b])
        return hashlib.md5(f"{pair[0]}|||{pair[1]}".encode()).hexdigest()
