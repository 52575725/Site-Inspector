from __future__ import annotations

import difflib
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.ollama_client import OllamaClient
from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class HTagRestructurer(BaseFixer):
    """Semi-auto fix heading hierarchy problems."""

    fixer_name = "htag_restructurer"
    fix_type = "semi_auto"
    supported_categories = ["h_tag_skip", "multiple_h1", "missing_h1"]

    def __init__(self, ollama: Optional[OllamaClient] = None):
        self.ollama = ollama

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        category = issue.get("category", "")
        soup = BeautifulSoup(page_content, "html.parser")

        if category == "missing_h1":
            page_content = self._add_h1(soup, issue)
        elif category == "multiple_h1":
            page_content = self._fix_multiple_h1(soup)
        elif category == "h_tag_skip":
            page_content = self._fix_hierarchy_gap(soup)

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

    def _add_h1(self, soup: BeautifulSoup, issue: dict) -> str:
        """Add an H1 if missing, using the page title or first meaningful heading."""
        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else ""

        # Find main content area or body
        main = soup.find("main") or soup.find("body")
        if not main:
            return str(soup)

        h1 = soup.new_tag("h1")
        if title:
            h1.string = title
        elif issue.get("url"):
            path = issue["url"].rstrip("/").split("/")[-1].replace("-", " ").title()
            h1.string = path or "Main"
        else:
            h1.string = "Main"

        # Insert at beginning of main content
        main.insert(0, h1)
        return str(soup)

    def _fix_multiple_h1(self, soup: BeautifulSoup) -> str:
        """Convert all H1s after the first to H2."""
        h1s = soup.find_all("h1")
        if len(h1s) <= 1:
            return str(soup)
        for h1 in h1s[1:]:
            h1.name = "h2"
        return str(soup)

    def _fix_hierarchy_gap(self, soup: BeautifulSoup) -> str:
        """Fix heading level skips by promoting/demoting."""
        headings = list(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))
        if not headings:
            return str(soup)

        # Normalize: ensure no skips
        prev_level = int(headings[0].name[1])
        for tag in headings[1:]:
            curr_level = int(tag.name[1])
            if curr_level > prev_level + 1:
                tag.name = f"h{prev_level + 1}"
            prev_level = int(tag.name[1])

        return str(soup)
