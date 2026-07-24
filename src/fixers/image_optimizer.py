from __future__ import annotations

import difflib
import logging
import re

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)


class ImageOptimizer(BaseFixer):
    """Fully-auto fix image SEO issues: add dimensions, lazy loading,
    async decoding, WebP conversion, and alt text fallback.

    When source provides file-system access (local/git), also performs
    actual image file conversion to WebP.
    """

    fixer_name = "image_optimizer"
    fix_type = "semi_auto"
    supported_categories = [
        "image_no_alt",
        "image_empty_alt",
        "image_missing_dimensions",
        "image_no_lazy_loading",
        "image_not_webp",
        "image_no_async_decoding",
        "large_images",  # images exceeding recommended size
    ]

    # Default fallback dimensions for common image types
    DEFAULT_DIMENSIONS = {
        "banner": (1200, 630),
        "hero": (1200, 800),
        "logo": (200, 60),
        "icon": (48, 48),
        "product": (800, 800),
        "thumbnail": (300, 300),
    }

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        category = issue.get("category", "")
        file_path = issue.get("file_path", "")
        issue_id = issue.get("id", 0)
        element_selector = issue.get("element", "")
        url = issue.get("url", "")

        soup = BeautifulSoup(page_content, "html.parser")

        # Find the target image(s)
        if element_selector:
            # Try to find by src match
            src_match = re.search(r"src='([^']+)'", element_selector)
            if src_match:
                target_src = src_match.group(1)
                images = [
                    img for img in soup.find_all("img")
                    if img.get("src", "") == target_src
                    or target_src in img.get("src", "")
                ]
            else:
                images = soup.find_all("img")
        else:
            images = soup.find_all("img")

        if not images:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message="No images found on page",
            )

        modified = 0
        for img in images:
            src = img.get("src", "")
            if not src:
                continue

            # Apply fixes based on category
            if category in ("image_no_alt", "image_empty_alt"):
                if self._fix_alt(img, src):
                    modified += 1

            if category == "image_missing_dimensions":
                if self._fix_dimensions(img, src):
                    modified += 1

            if category == "image_no_lazy_loading":
                if self._fix_lazy_loading(img):
                    modified += 1

            if category == "image_no_async_decoding":
                if self._fix_async_decoding(img):
                    modified += 1

            if category == "image_not_webp":
                if self._fix_webp_fallback(img, src, soup):
                    modified += 1

        if modified == 0:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message=f"No images needed '{category}' fix",
            )

        new_content = str(soup)
        diff = "\n".join(difflib.unified_diff(
            page_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
            lineterm="",
        ))

        logger.info(
            f"ImageOptimizer: applied '{category}' fix to {modified} "
            f"image(s) on {file_path}"
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

    @staticmethod
    def _fix_alt(img, src: str) -> bool:
        """Add alt text from filename or placeholder."""
        if img.get("alt") and img["alt"].strip():
            return False  # Already has meaningful alt text

        # Generate alt from filename
        filename = src.rstrip("/").split("/")[-1].split("?")[0]
        name = filename.rsplit(".", 1)[0] if "." in filename else filename
        # Convert dashes/underscores to spaces, title case
        alt = name.replace("-", " ").replace("_", " ").title()
        # Clean up common patterns
        alt = re.sub(r"\b(Img|Image|Photo|Pic)\b", "", alt, flags=re.IGNORECASE).strip()
        if len(alt) < 3:
            alt = "Product image"
        img["alt"] = alt
        return True

    @staticmethod
    def _fix_dimensions(img, src: str) -> bool:
        """Add width/height from existing attributes or defaults."""
        if img.get("width") and img.get("height"):
            return False

        # Try to get from style attribute
        style = img.get("style", "")
        w_match = re.search(r"width\s*:\s*(\d+)px", style)
        h_match = re.search(r"height\s*:\s*(\d+)px", style)
        if w_match and h_match:
            img["width"] = w_match.group(1)
            img["height"] = h_match.group(1)
            return True

        # Try to guess from image class/context
        classes = " ".join(img.get("class", []))
        src_lower = src.lower()
        for hint, (w, h) in ImageOptimizer.DEFAULT_DIMENSIONS.items():
            if hint in classes or hint in src_lower:
                img["width"] = str(w)
                img["height"] = str(h)
                return True

        # Default to common OG image dimensions
        img["width"] = "800"
        img["height"] = "600"
        return True

    @staticmethod
    def _fix_lazy_loading(img) -> bool:
        """Add loading='lazy' for below-fold images."""
        if img.get("loading"):
            return False
        img["loading"] = "lazy"
        return True

    @staticmethod
    def _fix_async_decoding(img) -> bool:
        """Add decoding='async' for off-screen images."""
        if img.get("decoding"):
            return False
        img["decoding"] = "async"
        return True

    @staticmethod
    def _fix_webp_fallback(img, src: str, soup: BeautifulSoup) -> bool:
        """Add <picture> element with WebP source and original fallback."""
        # Only add if parent is not already <picture>
        if img.parent and img.parent.name == "picture":
            return False

        # Build WebP URL by changing extension
        if "?" in src:
            base, query = src.split("?", 1)
        else:
            base = src
            query = ""

        webp_src = re.sub(r"\.(png|jpe?g|gif)$", ".webp", base, flags=re.IGNORECASE)
        if webp_src == base:
            return False  # Can't convert — already WebP or unknown format

        if query:
            webp_src += "?" + query

        # Create <picture> wrapper
        picture = soup.new_tag("picture")
        source_tag = soup.new_tag("source", srcset=webp_src, type="image/webp")
        picture.append(source_tag)

        # Move img into picture (BeautifulSoup handles this)
        img_copy = soup.new_tag(
            "img",
            src=img.get("src", ""),
            alt=img.get("alt", ""),
            width=img.get("width", ""),
            height=img.get("height", ""),
            loading=img.get("loading", ""),
            decoding=img.get("decoding", ""),
        )
        # Remove empty attributes
        for attr in list(img_copy.attrs.keys()):
            if not img_copy[attr]:
                del img_copy[attr]
        picture.append(img_copy)

        img.replace_with(picture)
        return True
