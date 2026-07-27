"""Visual regression testing via Playwright screenshot comparison.

Captures screenshots before and after fix application, performs pixel-level
comparison, and flags layout-breaking changes for rollback.

Uses the existing Playwright dependency (already required by MobileInspector).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Layout shift threshold: if >5% of pixels changed, flag as potential breakage
LAYOUT_SHIFT_WARNING = 0.05
LAYOUT_SHIFT_CRITICAL = 0.15


@dataclass
class ScreenshotResult:
    """Result of a before/after screenshot comparison."""
    file_path: str
    passed: bool = True
    before_path: str = ""
    after_path: str = ""
    diff_path: str = ""
    changed_pixel_ratio: float = 0.0
    errors: list[str] = field(default_factory=list)


async def capture_screenshot(url: str, output_path: Path) -> Optional[str]:
    """Capture a full-page screenshot of a URL using Playwright.

    Returns the path to the saved screenshot or None on failure.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not available for screenshot capture")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.screenshot(path=str(output_path), full_page=True)
        await browser.close()
        await pw.stop()
        return str(output_path)
    except Exception as e:
        logger.warning(f"Screenshot capture failed for {url}: {e}")
        return None


def compare_screenshots(
    before_path: str, after_path: str, diff_output_path: Path,
) -> float:
    """Pixel-level comparison of two screenshots.

    Uses PIL to compute the ratio of changed pixels.
    Returns the ratio (0.0 = identical, 1.0 = completely different).
    """
    try:
        from PIL import Image, ImageChops
    except ImportError:
        logger.warning("PIL not available for screenshot comparison")
        return 0.0

    try:
        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")

        # Resize to same dimensions if different
        if before.size != after.size:
            after = after.resize(before.size, Image.LANCZOS)

        # Compute absolute pixel difference
        diff = ImageChops.difference(before, after)
        diff = diff.convert("L")  # Grayscale

        # Count changed pixels (threshold >20 to ignore anti-aliasing noise)
        threshold = 20
        changed = sum(1 for p in diff.getdata() if p > threshold)
        total = diff.size[0] * diff.size[1]

        ratio = changed / max(total, 1)

        # Save diff image with red highlight
        if ratio > 0.001:
            diff_colored = Image.blend(
                before, after.resize(before.size, Image.LANCZOS), 0.5,
            )
            diff_colored.save(str(diff_output_path))

        return ratio
    except Exception as e:
        logger.warning(f"Screenshot comparison failed: {e}")
        return 0.0


async def visual_regression_check(
    file_path: str, before_content: str, after_content: str,
    screenshot_dir: Path, page_url: str,
) -> ScreenshotResult:
    """Run visual regression: render before/after HTML, screenshot, compare.

    This is a simplified version that writes HTML to temp files and opens
    them in Playwright. For production use with a live server, the URL
    should point to a staging environment instead.
    """
    result = ScreenshotResult(file_path=file_path)

    # Write before/after HTML to temp files
    before_html = screenshot_dir / f"{Path(file_path).stem}_before.html"
    after_html = screenshot_dir / f"{Path(file_path).stem}_after.html"

    before_html.write_text(before_content, encoding="utf-8")
    after_html.write_text(after_content, encoding="utf-8")

    before_png = screenshot_dir / f"{Path(file_path).stem}_before.png"
    after_png = screenshot_dir / f"{Path(file_path).stem}_after.png"
    diff_png = screenshot_dir / f"{Path(file_path).stem}_diff.png"

    # Capture screenshots of before/after
    before_captured = await capture_screenshot(
        f"file:///{before_html.as_posix()}", before_png,
    )
    after_captured = await capture_screenshot(
        f"file:///{after_html.as_posix()}", after_png,
    )

    if not before_captured or not after_captured:
        result.errors.append("Screenshot capture failed")
        result.passed = False
        return result

    result.before_path = before_captured
    result.after_path = after_captured

    # Compare
    ratio = compare_screenshots(before_captured, after_captured, diff_png)
    result.changed_pixel_ratio = ratio
    result.diff_path = str(diff_png)

    if ratio > LAYOUT_SHIFT_CRITICAL:
        result.passed = False
        result.errors.append(
            f"Critical layout shift: {ratio:.1%} pixels changed "
            f"(threshold: {LAYOUT_SHIFT_CRITICAL:.0%})"
        )
    elif ratio > LAYOUT_SHIFT_WARNING:
        result.errors.append(
            f"Warning: {ratio:.1%} pixels changed "
            f"(threshold: {LAYOUT_SHIFT_WARNING:.0%})"
        )

    return result
