from __future__ import annotations

import difflib

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class CanonicalFixer(BaseFixer):
    """Auto-fix missing or incorrect canonical URLs."""

    fixer_name = "canonical_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "missing_canonical", "canonical_mismatch",
    ]

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        url = issue.get("url", "")
        head = soup.find("head")

        if not head:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path="", before_content=page_content, after_content="",
                error_message="No <head> element found",
            )

        # Check if canonical already exists
        existing = soup.find("link", rel="canonical")
        if existing and existing.get("href"):
            # Update existing
            existing["href"] = url
        else:
            # Add new canonical link
            tag = soup.new_tag("link", rel="canonical", href=url)
            head.insert(0, tag)

        after_content = str(soup)
        diff = "".join(difflib.unified_diff(
            page_content.splitlines(keepends=True),
            after_content.splitlines(keepends=True),
            fromfile="before", tofile="after",
        ))

        return FixResult(
            success=True, issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name, fix_type=self.fix_type,
            file_path=self._url_to_filename(url),
            before_content=page_content,
            after_content=after_content,
            diff=diff,
        )

    @staticmethod
    def _url_to_filename(url: str) -> str:
        from urllib.parse import urlparse
        path = urlparse(url).path.strip("/")
        if not path or path.endswith("/"):
            return (path or "") + "index.html"
        if "." not in path.split("/")[-1]:
            return path + "/index.html"
        return path
