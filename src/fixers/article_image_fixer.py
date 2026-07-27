"""Auto-fix: find and insert relevant images into blog articles.

Searches free image APIs based on article content, downloads
images locally, and inserts them at natural positions in the article body.
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.integrations.image_search import (
    download_image,
    extract_keywords_from_html,
    search_images,
)
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)


class ArticleImageFixer(BaseFixer):
    """Fix articles with no images by searching and inserting relevant images.

    Searches free image APIs (Unsplash → Pexels → Pixabay) using article
    title and headings as queries. Downloads images to the local images/
    directory and inserts <img> tags at SEO-friendly positions:
      - First image after H1 (hero image)
      - Remaining images after H2 sections

    Fix type is "semi_auto" because image selection should be reviewed
    by a human before publishing.
    """

    fixer_name = "article_image_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "article_no_images",
        "blog_no_images",
        "thin_content_images",
    ]

    # Maximum images to insert per article
    MAX_IMAGES = 3
    # Minimum word count to consider inserting images
    MIN_WORDS = 200

    def __init__(
        self,
        unsplash_key: str = "",
        pexels_key: str = "",
        pixabay_key: str = "",
    ):
        self.unsplash_key = unsplash_key
        self.pexels_key = pexels_key
        self.pixabay_key = pixabay_key

    async def generate_fix(
        self,
        issue: dict,
        source: BaseSource,
        page_content: str,
    ) -> FixResult:
        url = issue.get("url", "")
        file_path = issue.get("file_path", "")
        issue_id = issue.get("id", 0)

        if not page_content:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content="",
                after_content="",
                error_message="No page content provided",
            )

        soup = BeautifulSoup(page_content, "html.parser")

        # Count existing images
        existing_images = soup.find_all("img")
        if len(existing_images) >= self.MAX_IMAGES:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message=f"Article already has {len(existing_images)} images",
            )

        # Extract keywords for image search
        queries = extract_keywords_from_html(page_content, max_queries=self.MAX_IMAGES)
        if not queries:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="Could not extract keywords from article",
            )

        # Search for images
        all_results = []
        for query in queries:
            results = search_images(
                query,
                count=1,
                unsplash_key=self.unsplash_key,
                pexels_key=self.pexels_key,
                pixabay_key=self.pixabay_key,
            )
            all_results.extend(results)

        if not all_results:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="No images found for article keywords",
            )

        # Download images to local directory
        work_dir = self._get_work_dir(source)
        images_dir = Path(work_dir) / "images" if work_dir else Path("images")
        images_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        for idx, result in enumerate(all_results[:self.MAX_IMAGES]):
            filename = self._generate_filename(result, idx, url)
            local_path = download_image(result.url, images_dir, filename)
            if local_path:
                result_dict = {
                    "local_path": f"/images/{Path(local_path).name}",
                    "alt_text": result.alt_text,
                    "width": result.width or 800,
                    "height": result.height or 600,
                    "source": result.source,
                    "photographer": result.photographer,
                }
                downloaded.append(result_dict)

        if not downloaded:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="Failed to download any images",
            )

        # Insert images into article
        modified_count = self._insert_images(soup, downloaded)

        if modified_count == 0:
            return FixResult(
                success=False,
                issue_id=issue_id,
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="Could not find suitable insertion points in article",
            )

        new_content = str(soup)
        diff = "\n".join(difflib.unified_diff(
            page_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))

        logger.info(
            f"ArticleImageFixer: inserted {modified_count} images into {url} "
            f"(queries: {', '.join(queries[:3])})"
        )

        return FixResult(
            success=True,
            issue_id=issue_id,
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=file_path,
            before_content=page_content,
            after_content=new_content,
            diff=diff[:5000],
        )

    def _insert_images(self, soup: BeautifulSoup, images: list[dict]) -> int:
        """Insert images at natural positions in the article body.

        Strategy:
        - First image: after H1 (hero/featured image)
        - Remaining images: after H2 sections (illustrate each section)
        """
        inserted = 0

        # Find content area
        body = soup.find("body")
        if not body:
            return 0

        main = (
            body.find("main")
            or body.find("article")
            or body.find("div", class_=re.compile(r"content|article|blog|post", re.I))
            or body
        )

        # Image 1: After H1 (hero image)
        if images:
            h1 = main.find("h1")
            if h1:
                img_html = self._build_img_tag(images[0], is_hero=True)
                img_soup = BeautifulSoup(img_html, "html.parser")
                h1.insert_after(img_soup)
                inserted += 1
            else:
                # No H1 — insert at top of main
                img_html = self._build_img_tag(images[0], is_hero=True)
                img_soup = BeautifulSoup(img_html, "html.parser")
                main.insert(0, img_soup)
                inserted += 1

        # Remaining images: After H2 sections
        if len(images) > 1:
            h2_tags = main.find_all("h2")
            for idx, h2 in enumerate(h2_tags):
                img_idx = inserted
                if img_idx >= len(images):
                    break
                img_html = self._build_img_tag(images[img_idx], is_hero=False)
                img_soup = BeautifulSoup(img_html, "html.parser")
                h2.insert_after(img_soup)
                inserted += 1

        return inserted

    @staticmethod
    def _build_img_tag(image: dict, is_hero: bool = False) -> str:
        """Build an <img> tag with proper SEO attributes."""
        src = image["local_path"]
        alt = image.get("alt_text", "Article image")
        width = image.get("width", 800)
        height = image.get("height", 600)
        source = image.get("source", "")
        photographer = image.get("photographer", "")

        style = "width:100%;max-width:800px;height:auto;margin:24px 0;border-radius:8px;display:block;"

        attribution = ""
        if source and photographer:
            attribution = f" <!-- Photo by {photographer} on {source.title()} -->"

        return (
            f'<img src="{src}" alt="{alt}" width="{width}" height="{height}" '
            f'loading="lazy" decoding="async" style="{style}"'
            f'{attribution}>'
        )

    @staticmethod
    def _generate_filename(result, idx: int, url: str) -> str:
        """Generate a descriptive filename for a downloaded image."""
        import hashlib

        alt_slug = result.alt_text.lower().replace(" ", "-")[:40]
        alt_slug = re.sub(r"[^a-z0-9-]", "", alt_slug)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        source_abbr = result.source[:4]
        return f"article-{url_hash}-{source_abbr}-{idx + 1}-{alt_slug}.jpg"

    @staticmethod
    def _get_work_dir(source: BaseSource | None) -> str | None:
        """Extract the working directory from a source for file I/O."""
        if source is None:
            return None
        return (
            getattr(source, "root", None)
            or getattr(source, "_work_dir", None)
        )
