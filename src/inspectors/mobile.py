from __future__ import annotations

import logging
from typing import Optional

from playwright.async_api import async_playwright

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

MOBILE_VIEWPORTS = {
    "small_phone": {"width": 375, "height": 812},
    "tablet": {"width": 768, "height": 1024},
    "desktop": {"width": 1024, "height": 768},
}


class MobileInspector(BaseInspector):
    """Inspect mobile adaptation issues across multiple viewports."""

    inspector_name = "mobile"

    def __init__(self):
        self._playwright = None
        self._browser = None

    async def setup(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)

    async def teardown(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not self._browser:
            logger.warning("Browser not started, skipping mobile inspection")
            return findings

        page = await self._browser.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            logger.warning(f"Failed to load {url} for mobile check: {e}")
            await page.close()
            return findings

        # Check viewport meta tag
        viewport_meta = await page.evaluate("""() => {
            const meta = document.querySelector('meta[name="viewport"]');
            return meta ? meta.content : null;
        }""")
        if not viewport_meta:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_viewport_meta",
                description="Page has no viewport meta tag (required for mobile)",
            ))

        # Check at each viewport
        for vp_name, vp_size in MOBILE_VIEWPORTS.items():
            await page.set_viewport_size(vp_size)
            await page.wait_for_timeout(500)

            vp_findings = await self._check_viewport(page, url, vp_name, vp_size)
            findings.extend(vp_findings)

        await page.close()
        return findings

    async def _check_viewport(self, page, url: str, vp_name: str,
                              vp_size: dict) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Check horizontal overflow
        has_horizontal_scroll = await page.evaluate("""() => {
            return document.documentElement.scrollWidth > document.documentElement.clientWidth + 5;
        }""")
        if has_horizontal_scroll:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="horizontal_scroll",
                description=f"Horizontal scrollbar detected at {vp_name} ({vp_size['width']}px)",
                raw_metadata={"viewport": vp_name, **vp_size},
            ))

        # Check font size (minimum 16px for body text to prevent iOS zoom)
        small_fonts = await page.evaluate("""() => {
            const small = [];
            document.querySelectorAll('p, li, span, a, div').forEach(el => {
                const style = window.getComputedStyle(el);
                const fontSize = parseFloat(style.fontSize);
                if (fontSize > 0 && fontSize < 16 && el.textContent.trim().length > 10) {
                    small.push({tag: el.tagName, size: fontSize, text: el.textContent.trim().substring(0, 30)});
                }
            });
            return small.slice(0, 3);
        }""")
        if small_fonts:
            details = "; ".join(f"{f['tag']} ({f['size']}px): '{f['text']}'" for f in small_fonts)
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="small_font_size",
                description=f"Text elements with font-size < 16px at {vp_name}: {details}",
                raw_metadata={"viewport": vp_name, "instances": small_fonts},
            ))

        # Check touch target sizes (< 48px is too small per WCAG)
        if vp_size["width"] < 1024:
            small_targets = await page.evaluate("""() => {
                const too_small = [];
                document.querySelectorAll('a, button, [role="button"], input[type="submit"]').forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0 &&
                        (rect.width < 48 || rect.height < 48)) {
                        too_small.push({
                            tag: el.tagName,
                            w: Math.round(rect.width),
                            h: Math.round(rect.height),
                            text: (el.textContent || '').trim().substring(0, 20)
                        });
                    }
                });
                return too_small.slice(0, 5);
            }""")
            if small_targets:
                details = "; ".join(
                    f"{t['tag']} ({t['w']}x{t['h']}px): '{t['text']}'"
                    for t in small_targets
                )
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="small_touch_targets",
                    description=f"Touch targets smaller than 48px at {vp_name}: {details}",
                    raw_metadata={"viewport": vp_name, "instances": small_targets},
                ))

        return findings
