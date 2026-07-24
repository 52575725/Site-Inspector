from __future__ import annotations

import difflib
import re
from collections import Counter

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "not", "no", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "just",
}


class KeywordsFixer(BaseFixer):
    """Auto-optimize keyword placement in title, H1, first paragraph, and URL."""

    fixer_name = "keywords_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "keyword_not_in_title",
        "keyword_not_in_h1",
        "keyword_not_in_first_paragraph",
        "keyword_not_in_url",
        "keyword_density_low",
    ]

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        category = issue.get("category", "")
        raw_metadata = issue.get("raw_metadata", {}) or {}
        keyword = raw_metadata.get("keyword", "")

        if not keyword:
            keyword = self._extract_keyword(soup, page_content)

        original = page_content

        if category == "keyword_not_in_title":
            page_content = self._fix_title(soup, page_content, keyword)
        elif category == "keyword_not_in_h1":
            page_content = self._fix_h1(soup, page_content, keyword)
        elif category == "keyword_not_in_first_paragraph":
            page_content = self._fix_first_paragraph(soup, page_content, keyword)
        elif category == "keyword_not_in_url":
            # Can't change URL in HTML file, but add a note in diff
            pass
        elif category == "keyword_density_low":
            page_content = self._boost_density(soup, page_content, keyword)

        if page_content == original and category != "keyword_not_in_url":
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path="", before_content=original, after_content="",
                error_message=f"Could not apply keyword fix for '{keyword}'",
            )

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            page_content.splitlines(keepends=True),
            fromfile="before", tofile="after",
        ))

        if category == "keyword_not_in_url":
            diff += (
                f"\n# URL 优化建议: 在 URL 路径中包含关键词 '{keyword}'\n"
                f"# 当前: {issue.get('url', '?')}\n"
                f"# 建议: 将 '{keyword}' 加入 URL slug（如 /{keyword}-guide/）"
            )

        return FixResult(
            success=True, issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name, fix_type=self.fix_type,
            file_path=self._url_to_filename(issue.get("url", "")),
            before_content=original, after_content=page_content, diff=diff,
        )

    # ── Title Fix ───────────────────────────────────────────────────

    def _fix_title(self, soup: BeautifulSoup, page_content: str, keyword: str) -> str:
        title_tag = soup.find("title")
        if not title_tag:
            head = soup.find("head")
            if not head:
                return page_content
            title_tag = soup.new_tag("title")
            head.insert(0, title_tag)

        current = title_tag.get_text(strip=True)
        if keyword.lower() in current.lower():
            return page_content

        # Insert keyword naturally: "Keyword - Existing Title" or "Existing Title | Keyword"
        if " - " in current:
            parts = current.split(" - ", 1)
            title_tag.string = f"{keyword} - {parts[1]}"
        elif " | " in current:
            parts = current.split(" | ", 1)
            title_tag.string = f"{parts[0]} | {keyword}"
        else:
            title_tag.string = f"{keyword} - {current}"

        return str(soup)

    # ── H1 Fix ──────────────────────────────────────────────────────

    def _fix_h1(self, soup: BeautifulSoup, page_content: str, keyword: str) -> str:
        h1 = soup.find("h1")
        # Never create a new H1 — that's HTagRestructurer's job
        if not h1:
            return page_content

        current = h1.get_text(strip=True)
        if keyword.lower() in current.lower():
            return page_content

        # Prepend keyword to existing H1 only
        h1.string = f"{keyword}: {current}"
        return str(soup)

    # ── First Paragraph Fix ─────────────────────────────────────────

    def _fix_first_paragraph(self, soup: BeautifulSoup, page_content: str, keyword: str) -> str:
        body = soup.find("body")
        if not body:
            return page_content

        # Find the first meaningful paragraph
        first_p = None
        for p in body.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 30:
                first_p = p
                break

        if not first_p:
            # Create one before the first h2 or at top of body
            first_h2 = body.find("h2")
            first_p = soup.new_tag("p")
            if first_h2:
                first_h2.insert_before(first_p)
            else:
                body.insert(0, first_p)

        current = first_p.get_text(strip=True)
        if keyword.lower() in current.lower():
            return page_content

        # Add keyword naturally to the opening sentence
        sentences = re.split(r"(?<=[.!?])\s+", current)
        if sentences:
            first_sentence = sentences[0]
            if len(first_sentence) > 50:
                # Insert keyword into first sentence
                first_sentence = re.sub(
                    r"(is|are|was|were|provides|offers|delivers)\s",
                    rf"\1 {keyword} ",
                    first_sentence,
                    count=1,
                )
                sentences[0] = first_sentence
            else:
                sentences.insert(0, f"When it comes to {keyword}, {sentences[0][0].lower() + sentences[0][1:]}")
            first_p.string = " ".join(sentences)

        return str(soup)

    # ── Density Boost ───────────────────────────────────────────────

    def _boost_density(self, soup: BeautifulSoup, page_content: str, keyword: str) -> str:
        body = soup.find("body")
        if not body:
            return page_content

        # Count current occurrences
        text = body.get_text(separator=" ", strip=True).lower()
        count = text.count(keyword.lower())
        words = len(text.split())
        density = (count / words) * 100 if words > 0 else 0

        if density >= 1.0:
            return page_content  # already good

        # Add one more occurrence in a natural position
        paragraphs = body.find_all("p")
        # Find a paragraph that doesn't already have the keyword
        for p in paragraphs:
            if keyword.lower() not in p.get_text(strip=True).lower():
                current_text = p.get_text(strip=True)
                if len(current_text) > 80:
                    # Append a sentence with the keyword
                    new_text = (
                        f"{current_text} This is why {keyword} remains "
                        f"such an important consideration for industry professionals."
                    )
                    p.string = new_text
                    break

        return str(soup)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_keyword(soup: BeautifulSoup, page_content: str) -> str:
        """Extract primary keyword from page content."""
        title = ""
        h1 = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        h1_tag = soup.find("h1")
        if h1_tag:
            h1 = h1_tag.get_text(strip=True)

        # Combine and extract meaningful words
        combined = f"{title} {h1}".lower()
        words = re.findall(r"[a-z]{4,}", combined)
        filtered = [w for w in words if w not in STOPWORDS]
        if filtered:
            counts = Counter(filtered)
            return counts.most_common(1)[0][0]
        return ""

    @staticmethod
    def _url_to_filename(url: str) -> str:
        from urllib.parse import urlparse
        path = urlparse(url).path.strip("/")
        if not path or path.endswith("/"):
            return (path or "index") + "index.html"
        if "." not in path.split("/")[-1]:
            return path + "/index.html"
        return path
