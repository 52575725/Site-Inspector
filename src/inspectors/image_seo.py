from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Image extensions that support modern formats
MODERN_FORMATS = {".webp", ".avif"}
LEGACY_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


class ImageSEOInspector(BaseInspector):
    """Detects image SEO issues: missing dimensions, no lazy loading,
    legacy formats (no WebP), missing alt text, and large inline images.
    """

    inspector_name = "image_seo"

    # Images above the fold (first image, images in header) should NOT be lazy-loaded
    ABOVE_FOLD_POSITION = 3  # First 3 images on page are considered above-fold

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        images = soup.find_all("img")

        if not images:
            return findings

        for idx, img in enumerate(images):
            src = img.get("src", "")
            if not src:
                continue

            is_above_fold = idx < self.ABOVE_FOLD_POSITION

            # Check 1: Missing alt text
            alt = img.get("alt")
            if alt is None:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="image_no_alt",
                    description=f"Image missing alt attribute: src='{src[:100]}'",
                    element=f"img[src='{src[:80]}']",
                    current_value="(missing)",
                    suggested_value="Add descriptive alt text for accessibility and SEO",
                ))
            elif alt.strip() == "" and not img.get("role") == "presentation":
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="image_empty_alt",
                    description=f"Image has empty alt text: src='{src[:100]}'",
                    element=f"img[src='{src[:80]}']",
                    current_value="(empty)",
                    suggested_value="Add descriptive alt text or role='presentation'",
                ))

            # Check 2: Missing width/height attributes (CLS prevention)
            width = img.get("width")
            height = img.get("height")
            if not width or not height:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="image_missing_dimensions",
                    description=(
                        f"Image missing explicit dimensions: src='{src[:100]}'. "
                        f"This causes Cumulative Layout Shift (CLS)."
                    ),
                    element=f"img[src='{src[:80]}']",
                    current_value=f"width={width}, height={height}",
                    suggested_value="Add width/height attributes matching intrinsic image size",
                ))

            # Check 3: No lazy loading (below-fold images only)
            if not is_above_fold:
                loading = img.get("loading", "")
                if loading != "lazy":
                    findings.append(RawFinding(
                        url=url,
                        inspector=self.inspector_name,
                        category="image_no_lazy_loading",
                        description=(
                            f"Below-fold image missing loading='lazy': src='{src[:100]}'. "
                            f"Lazy loading reduces initial page load time."
                        ),
                        element=f"img[src='{src[:80]}']",
                        current_value=f"loading={loading or '(not set)'}",
                        suggested_value="loading='lazy'",
                    ))

            # Check 4: Legacy image format (not WebP/AVIF)
            parsed = urlparse(src)
            path_lower = parsed.path.lower()
            ext = "." + path_lower.split(".")[-1] if "." in path_lower else ""
            if ext in LEGACY_FORMATS:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="image_not_webp",
                    description=(
                        f"Image uses legacy format '{ext}': src='{src[:100]}'. "
                        f"WebP/AVIF formats reduce file size by 25-50% vs JPEG/PNG."
                    ),
                    element=f"img[src='{src[:80]}']",
                    current_value=ext,
                    suggested_value="Convert to WebP and add <picture> with fallback",
                ))

            # Check 5: Missing decoding="async" for off-screen images
            if not is_above_fold and not img.get("decoding"):
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="image_no_async_decoding",
                    description=(
                        f"Off-screen image missing decoding='async': src='{src[:100]}'"
                    ),
                    element=f"img[src='{src[:80]}']",
                    current_value="(not set)",
                    suggested_value="decoding='async'",
                ))

        return findings
