from __future__ import annotations

import difflib
import json
import re
from datetime import datetime

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class FreshnessFixer(BaseFixer):
    """Auto-update stale content: year references, relative time phrases, dates."""

    fixer_name = "freshness_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "freshness_outdated_year_refs",
        "freshness_relative_time_refs",
        "freshness_very_stale",
        "freshness_stale",
        "freshness_aging",
        "freshness_no_date_modified",
        "freshness_schema_date_modified_old",
    ]

    def __init__(self):
        self._today = datetime.utcnow()
        self._current_year = self._today.year

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        category = issue.get("category", "")
        raw_metadata = issue.get("raw_metadata", {}) or {}
        original = page_content

        if category == "freshness_outdated_year_refs":
            stale_years = raw_metadata.get("stale_years", [])
            page_content = self._update_year_refs(soup, page_content, stale_years)

        elif category == "freshness_relative_time_refs":
            page_content = self._fix_relative_refs(soup, page_content)

        elif category in ("freshness_very_stale", "freshness_stale", "freshness_aging",
                          "freshness_schema_date_modified_old"):
            page_content = self._update_schema_date(soup, page_content)
            page_content = self._add_updated_notice(soup, page_content)

        elif category == "freshness_no_date_modified":
            page_content = self._update_schema_date(soup, page_content)

        if page_content == original:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path="", before_content=original, after_content="",
                error_message=f"No changes needed for {category}",
            )

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            page_content.splitlines(keepends=True),
            fromfile="before", tofile="after",
        ))

        return FixResult(
            success=True, issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name, fix_type=self.fix_type,
            file_path=self._url_to_filename(issue.get("url", "")),
            before_content=original, after_content=page_content, diff=diff,
        )

    # ── Year Reference Updates ──────────────────────────────────────

    def _update_year_refs(
        self, soup: BeautifulSoup, page_content: str, stale_years: list[int],
    ) -> str:
        """Replace stale year references with current year throughout the page."""
        if not stale_years:
            return page_content

        body = soup.find("body")
        if not body:
            return page_content

        # Update visible text
        for element in body.find_all(string=True):
            new_text = element
            for year in stale_years:
                # Replace patterns like "best of 2023" → "best of 2026"
                # But NOT dates in URLs or structured data
                if str(year) in new_text and element.parent.name != "script":
                    new_text = new_text.replace(
                        str(year), str(self._current_year),
                    )
            element.replace_with(new_text)

        # Also update meta description if it has old years
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            content = meta_desc["content"]
            for year in stale_years:
                if str(year) in content:
                    content = content.replace(str(year), str(self._current_year))
            meta_desc["content"] = content

        return str(soup)

    # ── Relative Time Phrases ───────────────────────────────────────

    def _fix_relative_refs(self, soup: BeautifulSoup, page_content: str) -> str:
        """Replace relative time phrases with absolute dates."""
        replacements = {
            "last year": f"{self._current_year - 1}",
            "earlier this year": f"early {self._current_year}",
            "recently launched": f"launched in {self._current_year}",
            "just announced": f"announced in {self._current_year}",
            "newly released": f"released in {self._current_year}",
            "this year": f"{self._current_year}",
        }

        body = soup.find("body")
        if not body:
            return page_content

        for element in body.find_all(string=True):
            new_text = element
            for phrase, replacement in replacements.items():
                if phrase in new_text.lower() and element.parent.name != "script":
                    new_text = re.sub(
                        re.escape(phrase), replacement, new_text,
                        flags=re.IGNORECASE,
                    )
            if new_text != element:
                element.replace_with(new_text)

        return str(soup)

    # ── Schema Date Update ──────────────────────────────────────────

    def _update_schema_date(self, soup: BeautifulSoup, page_content: str) -> str:
        """Update dateModified in JSON-LD schema to today."""
        today_str = self._today.strftime("%Y-%m-%d")
        scripts = soup.find_all("script", type="application/ld+json")

        for script in scripts:
            try:
                data = json.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    if isinstance(block, dict):
                        if block.get("@type") in ("Article", "BlogPosting", "NewsArticle", "WebPage"):
                            block["dateModified"] = today_str
                script.string = json.dumps(data, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass

        return str(soup)

    # ── Updated Notice ──────────────────────────────────────────────

    def _add_updated_notice(self, soup: BeautifulSoup, page_content: str) -> str:
        """Add or update a visible 'Updated on [date]' notice."""
        today_str = self._today.strftime("%B %d, %Y")

        # Look for existing update notice
        update_el = soup.find(
            string=re.compile(r"(?:Updated|Last updated|Reviewed)(?:\s+on)?\s+", re.IGNORECASE),
        )
        if update_el:
            update_el.replace_with(f"Updated: {today_str}")
            return str(soup)

        # Add after H1 or first paragraph
        h1 = soup.find("h1")
        if h1:
            notice = soup.new_tag("p", attrs={"class": "article-updated"})
            notice.string = f"Updated: {today_str}"
            h1.insert_after(notice)

        return str(soup)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _url_to_filename(url: str) -> str:
        from urllib.parse import urlparse
        path = urlparse(url).path.strip("/")
        if not path or path.endswith("/"):
            return (path or "index") + "index.html"
        if "." not in path.split("/")[-1]:
            return path + "/index.html"
        return path
