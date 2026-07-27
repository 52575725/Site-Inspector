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
        """Add an H1 if missing, using the page title or first meaningful heading.

        IMPORTANT: Checks for existing H1 before inserting — a false-positive
        missing_h1 detection must not create a duplicate H1.
        """
        # Guard: if the page already has an H1, the inspector's missing_h1
        # finding is a false positive — do NOT add another one.
        existing_h1s = soup.find_all("h1")
        if existing_h1s:
            return str(soup)

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
        """Keep the most likely visible page heading and demote the rest."""
        h1s = soup.find_all("h1")
        if len(h1s) <= 1:
            return str(soup)

        preferred = next((
            h1 for h1 in h1s
            if h1.find_parent(["main", "article"])
            or h1.find_parent(class_=re.compile(r"hero|page-header|article-header", re.I))
        ), None)
        keeper = preferred or next((h1 for h1 in h1s if not self._is_hidden(h1)), h1s[0])

        for h1 in h1s:
            if h1 is not keeper:
                h1.name = "h2"
        return str(soup)

    @staticmethod
    def _is_hidden(tag) -> bool:
        current = tag
        while current is not None:
            if current.get("aria-hidden", "").lower() == "true":
                return True
            style = re.sub(r"\s+", "", current.get("style", "").lower())
            if any(token in style for token in (
                "display:none", "visibility:hidden", "opacity:0", "left:-9999",
            )):
                return True
            current = current.parent
        return False

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
