from __future__ import annotations

import logging
import re
from collections import Counter

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


class ContentGapDetector(BaseInspector):
    """Compare different language versions of the same page for content parity."""

    inspector_name = "content_gap"

    def __init__(self):
        self._page_pairs: list[dict] = []
        self._page_htmls: dict[str, str] = {}
        self._already_checked: set[str] = set()

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_page_pairs(self, pairs: list[dict]) -> None:
        """Set language page pairs for comparison.
        pairs format: [{"en": "/", "fr": "/fr/"}, {"en": "/products/", "fr": "/fr/produits/"}, ...]
        """
        self._page_pairs = pairs

    def set_page_htmls(self, html_map: dict[str, str]) -> None:
        """Set a mapping of URL → HTML content for all crawled pages."""
        self._page_htmls = html_map

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not self._page_pairs or not self._page_htmls:
            return findings

        for pair in self._page_pairs:
            lang_urls = list(pair.items())  # [(lang_code, path), ...]
            urls_in_pair = {v for _, v in lang_urls}

            if url not in urls_in_pair:
                continue

            pair_key = "↔".join(sorted(urls_in_pair))
            if pair_key in self._already_checked:
                return findings
            self._already_checked.add(pair_key)

            htmls_by_lang = {}
            for lang_code, lang_path in lang_urls:
                html = self._page_htmls.get(lang_path, "")
                if html:
                    htmls_by_lang[lang_code] = (lang_path, html)

            if len(htmls_by_lang) < 2:
                continue

            findings.extend(self._compare_pair(htmls_by_lang, url))
            break

        return findings

    def _compare_pair(self, htmls_by_lang: dict, current_url: str) -> list[RawFinding]:
        """Compare content across language versions. Uses the first language as baseline."""
        findings: list[RawFinding] = []
        lang_entries = list(htmls_by_lang.items())
        baseline_lang, (baseline_url, baseline_html) = lang_entries[0]
        baseline_soup = BeautifulSoup(baseline_html, "html.parser")

        for lang_code, (lang_url, lang_html) in lang_entries[1:]:
            if lang_url == current_url or baseline_url == current_url:
                findings.extend(self._compare_two(
                    baseline_lang, baseline_url, baseline_soup,
                    lang_code, lang_url, lang_html,
                ))

        return findings

    def _compare_two(self, lang_a: str, url_a: str, soup_a: BeautifulSoup,
                     lang_b: str, url_b: str, html_b: str) -> list[RawFinding]:
        findings: list[RawFinding] = []
        soup_b = BeautifulSoup(html_b, "html.parser")

        # 1. Word count ratio
        text_a = self._get_visible_text(soup_a)
        text_b = self._get_visible_text(soup_b)
        words_a = len(text_a.split())
        words_b = len(text_b.split())

        if words_a > 100 and words_b > 0:
            ratio = words_b / words_a if words_a > 0 else 1.0
            if ratio < 0.5:
                findings.append(RawFinding(
                    url=url_b, inspector=self.inspector_name,
                    category="content_gap_word_count",
                    description=f"{lang_b.upper()} page has significantly less content than "
                                f"{lang_a.upper()} ({words_b} vs {words_a} words, ratio {ratio:.1%})",
                    current_value=f"{lang_b.upper()}: {words_b} words, {lang_a.upper()}: {words_a} words",
                    suggested_value=f"Expand {lang_b.upper()} content to at least {int(words_a * 0.5)} words",
                ))

        # 2. Compare heading structure
        headings_a = self._get_heading_structure(soup_a)
        headings_b = self._get_heading_structure(soup_b)

        for h_text in headings_a:
            if not self._heading_exists_in(h_text, headings_b):
                findings.append(RawFinding(
                    url=url_b, inspector=self.inspector_name,
                    category="content_gap_section",
                    description=f"Section '{h_text[:60]}' exists in {lang_a.upper()} but "
                                f"missing in {lang_b.upper()}",
                    current_value=f"{lang_a.upper()}: {h_text[:100]}",
                    suggested_value=f"Add equivalent section to {lang_b.upper()} page",
                ))

        # 3. Compare link count
        links_a = self._get_significant_links(soup_a)
        links_b = self._get_significant_links(soup_b)

        if links_a and links_b:
            ratio = len(links_b) / len(links_a) if links_a else 1.0
            if ratio < 0.5:
                findings.append(RawFinding(
                    url=url_b, inspector=self.inspector_name,
                    category="content_gap_links",
                    description=f"{lang_b.upper()} page has significantly fewer internal links "
                                f"({len(links_b)} vs {lang_a.upper()} {len(links_a)}, ratio {ratio:.1%})",
                    current_value=f"{lang_b.upper()}: {len(links_b)} links, "
                                  f"{lang_a.upper()}: {len(links_a)} links",
                ))

        return findings

    @staticmethod
    def _get_visible_text(soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)

    @staticmethod
    def _get_heading_structure(soup: BeautifulSoup) -> list[str]:
        headings = []
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 2:
                headings.append(text.lower())
        return headings

    @staticmethod
    def _heading_exists_in(source_heading: str, target_headings: list[str]) -> bool:
        """Check if a heading has a rough equivalent in the target language headings.
        Uses substring matching and numeric/date pattern matching."""
        src_lower = source_heading.lower()
        for t_h in target_headings:
            t_lower = t_h.lower()
            if src_lower in t_lower or t_lower in src_lower:
                return True

        # Check for shared numeric patterns (e.g., "2024 market report" vs "2024 市场报告")
        src_digits = set(re.findall(r"\d+", src_lower))
        if src_digits:
            for t_h in target_headings:
                t_digits = set(re.findall(r"\d+", t_h.lower()))
                if src_digits & t_digits:
                    return True

        return False

    @staticmethod
    def _get_significant_links(soup: BeautifulSoup) -> list[str]:
        """Get internal navigation and CTA links (not footer boilerplate)."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text(strip=True)
            # Skip empty, hash-only, external, and common footer links
            if not text or href.startswith("#") or href.startswith("http"):
                continue
            if text.lower() in ("privacy policy", "terms", "©", "top", "back"):
                continue
            links.append(text.lower())
        return links
