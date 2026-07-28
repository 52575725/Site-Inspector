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
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception as e:
                logger.warning(f"Failed to load {url} for mobile check: {e}")
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
        finally:
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

        # Flag only strong mobile readability signals. Body text has no universal
        # 16px WCAG minimum; 16px matters specifically for iOS form-control zoom.
        small_fonts = []
        if vp_size["width"] < 1024:
            small_fonts = await page.evaluate("""() => {
            const small = [];
            const selectorFor = el => {
                if (el.id) return `#${el.id}`;
                const parts = [];
                while (el && el.nodeType === 1 && parts.length < 6) {
                    let part = el.tagName.toLowerCase();
                    const siblings = el.parentElement
                        ? [...el.parentElement.children].filter(s => s.tagName === el.tagName)
                        : [];
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(el) + 1})`;
                    parts.unshift(part);
                    el = el.parentElement;
                }
                return parts.join(' > ');
            };
            document.querySelectorAll('p, li, input, select, textarea').forEach(el => {
                const style = window.getComputedStyle(el);
                const fontSize = parseFloat(style.fontSize);
                const isControl = ['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName);
                const hasReadableText = isControl || el.textContent.trim().length > 10;
                if (hasReadableText && ((isControl && fontSize < 16) || (!isControl && fontSize < 12))) {
                    small.push({
                        tag: el.tagName, size: fontSize,
                        text: el.textContent.trim().substring(0, 30),
                        selector: selectorFor(el), html: el.outerHTML.substring(0, 300)
                    });
                }
            });
            return small.slice(0, 3);
            }""")
        if small_fonts:
            details = "; ".join(f"{f['tag']} ({f['size']}px): '{f['text']}'" for f in small_fonts)
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="small_font_size",
                description=f"Potentially unreadable mobile text at {vp_name}: {details}",
                element=small_fonts[0]["selector"],
                element_html=small_fonts[0]["html"],
                raw_metadata={"viewport": vp_name, "instances": small_fonts},
                scope="element",
                confidence=0.8,
            ))

        # WCAG 2.2 AA target-size minimum is 24x24 CSS px, with exceptions.
        if vp_size["width"] < 1024:
            small_targets = await page.evaluate("""() => {
                const too_small = [];
                const selectorFor = el => {
                    if (el.id) return `#${el.id}`;
                    const parts = [];
                    while (el && el.nodeType === 1 && parts.length < 6) {
                        let part = el.tagName.toLowerCase();
                        const siblings = el.parentElement
                            ? [...el.parentElement.children].filter(s => s.tagName === el.tagName)
                            : [];
                        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(el) + 1})`;
                        parts.unshift(part);
                        el = el.parentElement;
                    }
                    return parts.join(' > ');
                };
                document.querySelectorAll('a, button, [role="button"], input[type="submit"]').forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0 &&
                        (rect.width < 24 || rect.height < 24)) {
                        too_small.push({
                            tag: el.tagName,
                            w: Math.round(rect.width),
                            h: Math.round(rect.height),
                            text: (el.textContent || '').trim().substring(0, 20),
                            selector: selectorFor(el),
                            html: el.outerHTML.substring(0, 300)
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
                    description=f"Potential touch targets smaller than 24px at {vp_name}: {details}",
                    element=small_targets[0]["selector"],
                    element_html=small_targets[0]["html"],
                    raw_metadata={"viewport": vp_name, "instances": small_targets},
                    scope="element",
                    confidence=0.7,
                ))

        return findings
