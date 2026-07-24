from __future__ import annotations

import difflib
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class OgImageFixer(BaseFixer):
    """Fix missing og:image and Twitter card tags."""

    fixer_name = "og_image_fixer"
    fix_type = "fully_auto"
    supported_categories = ["missing_og_image", "missing_twitter_cards"]

    def __init__(self, default_image: str = "",
                 default_width: str = "1200",
                 default_height: str = "630"):
        self.default_image = default_image or ""
        self.default_width = default_width
        self.default_height = default_height

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        head = soup.find("head")
        if not head:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message="No <head> tag found",
            )

        category = issue.get("category", "")
        url = issue.get("url", "")

        if category == "missing_og_image":
            self._fix_og_image(soup, head, url)
        elif category == "missing_twitter_cards":
            self._fix_twitter_cards(soup, head, url)

        new_content = str(soup)
        diff = difflib.unified_diff(
            page_content.splitlines(True),
            new_content.splitlines(True),
            lineterm="",
        )

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=page_content,
            after_content=new_content,
            diff="\n".join(diff),
        )

    def _fix_og_image(self, soup: BeautifulSoup, head, url: str) -> None:
        """Insert missing og:image tags."""
        image_url = self._find_page_image(soup, url)

        existing_og = {
            meta.get("property", "")
            for meta in soup.find_all("meta", attrs={"property": True})
        }

        tags = [
            ("og:image", image_url),
            ("og:image:width", self.default_width),
            ("og:image:height", self.default_height),
        ]
        for prop, content in tags:
            if prop not in existing_og:
                new_tag = soup.new_tag("meta", property=prop, content=content)
                head.append(new_tag)

    def _fix_twitter_cards(self, soup: BeautifulSoup, head, url: str) -> None:
        """Insert missing Twitter card tags, derived from OG tags."""
        existing_twitter = {
            meta.get("name", "")
            for meta in soup.find_all("meta", attrs={"name": True})
        }

        og_title = soup.find("meta", property="og:title")
        og_desc = soup.find("meta", property="og:description")
        og_image = soup.find("meta", property="og:image")

        title = og_title.get("content", "") if og_title else ""
        desc = og_desc.get("content", "") if og_desc else ""
        image = og_image.get("content", "") if og_image else self.default_image

        tags = [
            ("twitter:card", "summary_large_image"),
            ("twitter:title", title),
            ("twitter:description", desc),
            ("twitter:image", image),
        ]
        for meta_name, content in tags:
            if meta_name not in existing_twitter and content:
                new_tag = soup.new_tag("meta", attrs={"name": meta_name, "content": content})
                head.append(new_tag)

    def _find_page_image(self, soup: BeautifulSoup, url: str) -> str:
        """Try to find a suitable image on the page. Falls back to default."""
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if not src:
                continue
            width = img.get("width", "")
            try:
                if width and int(width) >= 200:
                    return urljoin(url, src)
            except ValueError:
                pass

        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src and not src.startswith("data:"):
                return urljoin(url, src)

        return self.default_image
