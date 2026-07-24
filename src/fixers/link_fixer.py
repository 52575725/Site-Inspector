from __future__ import annotations

import difflib
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class LinkFixer(BaseFixer):
    """Auto-fix broken internal links and mixed content (HTTP→HTTPS upgrade)."""

    fixer_name = "link_fixer"
    fix_type = "semi_auto"
    supported_categories = ["http_404", "http_500", "mixed_content",
                           "redirect_chain", "link_timeout"]

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        category = issue.get("category", "")
        element = issue.get("element", "")
        url = issue.get("url", "")

        soup = BeautifulSoup(page_content, "html.parser")

        if category == "mixed_content":
            page_content = self._fix_mixed_content(soup, page_content)
        elif category in ("http_404", "http_500", "link_timeout"):
            page_content = self._fix_broken_link(soup, page_content, element, url)

        diff = difflib.unified_diff(
            (issue.get("before_content") or "").splitlines(True),
            page_content.splitlines(True),
            lineterm="",
        )

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=issue.get("before_content", ""),
            after_content=page_content,
            diff="\n".join(diff),
        )

    def _fix_mixed_content(self, soup: BeautifulSoup, html: str) -> str:
        """Upgrade HTTP resources to HTTPS."""
        changed = False
        for tag_name, attr in [("a", "href"), ("img", "src"),
                                ("link", "href"), ("script", "src"),
                                ("iframe", "src"), ("source", "src"),
                                ("video", "src"), ("audio", "src")]:
            for tag in soup.find_all(tag_name):
                val = tag.get(attr)
                if val and val.startswith("http://"):
                    tag[attr] = val.replace("http://", "https://", 1)
                    changed = True
        return str(soup) if changed else html

    def _fix_broken_link(self, soup: BeautifulSoup, html: str,
                         broken_url: str, page_url: str) -> str:
        """Attempt to fix a broken internal link by finding similar URLs or removing."""
        if not broken_url:
            return html

        # Try fixing .html extension inconsistencies
        base_domain = urlparse(page_url).netloc
        broken_domain = urlparse(broken_url).netloc

        if broken_domain != base_domain:
            return html  # Can't fix external links

        # Common fix: add/remove trailing slash or .html extension
        candidates = [
            broken_url.rstrip("/") + "/",
            broken_url.rstrip("/") + ".html",
            broken_url.replace(".html", ""),
            broken_url.rstrip("/"),
        ]

        # Try to find link element and replace with most likely candidate
        for tag_name, attr in [("a", "href"), ("img", "src")]:
            for tag in soup.find_all(tag_name):
                if tag.get(attr) == broken_url:
                    # Try the first candidate as the most likely fix
                    tag[attr] = candidates[0]
                    return str(soup)

        return html
