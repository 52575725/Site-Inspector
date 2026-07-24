from __future__ import annotations

import difflib
import logging

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)


class RobotsTxtFixer(BaseFixer):
    """Fully-auto fix robots.txt issues: create missing file, add sitemap
    reference, fix disallow-all, reduce crawl-delay.
    """

    fixer_name = "robots_txt_fixer"
    fix_type = "fully_auto"
    supported_categories = [
        "robots_txt_missing",
        "robots_txt_empty",
        "robots_txt_no_sitemap",
        "robots_txt_disallow_all",
        "robots_txt_high_crawl_delay",
    ]

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        category = issue.get("category", "")
        file_path = issue.get("file_path", "") or "robots.txt"
        issue_id = issue.get("id", 0)
        url = issue.get("url", "")

        # Current content (empty for missing)
        current = page_content or ""

        # Build robots.txt content
        new_content = self._build_robots_txt(category, current, url)

        if new_content == current:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=current, after_content=current,
                error_message="No changes needed for robots.txt",
            )

        diff = "\n".join(difflib.unified_diff(
            current.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
            lineterm="",
        ))

        logger.info(f"RobotsTxtFixer: applied '{category}' fix")

        return FixResult(
            success=True,
            issue_id=issue_id,
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=file_path,
            before_content=current,
            after_content=new_content,
            diff=diff,
        )

    @staticmethod
    def _build_robots_txt(category: str, current: str, url: str) -> str:
        """Build or fix robots.txt content."""
        from urllib.parse import urlparse

        # Determine sitemap URL from the issue URL
        parsed = urlparse(url) if url else None
        sitemap_url = (
            f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
            if parsed and parsed.netloc
            else "https://example.com/sitemap.xml"
        )

        lines = current.split("\n") if current else []

        if category in ("robots_txt_missing", "robots_txt_empty"):
            # Create new robots.txt from scratch
            return (
                f"User-agent: *\n"
                f"Allow: /\n\n"
                f"Sitemap: {sitemap_url}\n"
            )

        if category == "robots_txt_no_sitemap":
            # Add sitemap directive
            if not any(l.lower().startswith("sitemap:") for l in lines):
                if lines and lines[-1].strip():
                    lines.append("")
                lines.append(f"Sitemap: {sitemap_url}")
            return "\n".join(lines) + ("\n" if lines else "")

        if category == "robots_txt_disallow_all":
            # Replace Disallow: / with Allow: /
            fixed = []
            for line in lines:
                stripped = line.strip()
                if stripped.lower() == "disallow: /":
                    fixed.append("Allow: /")
                else:
                    fixed.append(line)
            return "\n".join(fixed) + ("\n" if fixed else "")

        if category == "robots_txt_high_crawl_delay":
            # Reduce crawl-delay
            fixed = []
            for line in lines:
                stripped = line.strip()
                if stripped.lower().startswith("crawl-delay:"):
                    fixed.append("Crawl-delay: 5")
                else:
                    fixed.append(line)
            return "\n".join(fixed) + ("\n" if fixed else "")

        return current
