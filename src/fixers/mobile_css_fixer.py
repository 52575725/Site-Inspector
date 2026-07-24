from __future__ import annotations

import difflib
import logging
import re

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)


class MobileCssFixer(BaseFixer):
    """Semi-auto fix mobile CSS issues (horizontal scroll, touch targets, font sizes).

    Injects inline <style> blocks with mobile-friendly CSS rules directly into
    the <head> of the page. Uses string manipulation instead of BeautifulSoup
    to avoid HTML reformatting that causes fix failures.
    """

    fixer_name = "mobile_css_fixer"
    fix_type = "semi_auto"
    supported_categories = ["horizontal_scroll", "small_font_size", "small_touch_targets"]

    CSS_TEMPLATES = {
        "small_font_size": """
/* [Site Inspector] Ensure minimum 16px font size for mobile readability */
@media (max-width: 768px) {
    body, p, li, span, a, div, td, th, label, input, textarea, select, button {
        font-size: 16px !important;
    }
}""",
        "small_touch_targets": """
/* [Site Inspector] Ensure minimum 48px touch targets for mobile */
@media (pointer: coarse) {
    a, button, [role="button"], input[type="submit"], input[type="button"],
    .btn, .button, [onclick] {
        min-width: 48px;
        min-height: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
}""",
        "horizontal_scroll": """
/* [Site Inspector] Prevent horizontal overflow on all viewports */
html, body {
    overflow-x: hidden;
    max-width: 100vw;
}
img, table, pre, iframe, video, canvas, svg {
    max-width: 100%;
    height: auto;
}""",
    }

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        category = issue.get("category", "")
        file_path = issue.get("file_path", "")
        issue_id = issue.get("id", 0)

        if category not in self.CSS_TEMPLATES:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message=f"Unknown category: {category}",
            )

        css_rule = self.CSS_TEMPLATES[category]

        # Check if this specific fix is already present
        marker = "/* [Site Inspector]"
        if marker in page_content and category in page_content:
            # Check if the exact rule for this category is already there
            # Simple heuristic: look for the category-specific comment
            cat_marker = f"/* [Site Inspector] {self._get_rule_description(category)}"
            if cat_marker.lower() in page_content.lower():
                return FixResult(
                    success=False, issue_id=issue_id,
                    fixer_name=self.fixer_name, fix_type=self.fix_type,
                    file_path=file_path,
                    before_content=page_content, after_content=page_content,
                    error_message=f"CSS fix for '{category}' already applied",
                )

        # Inject CSS using string manipulation (more reliable than BS4 parsing)
        after_content = self._inject_css(page_content, css_rule)

        if after_content == page_content:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message="Failed to inject CSS — no suitable insertion point found",
            )

        diff = "\n".join(difflib.unified_diff(
            page_content.splitlines(keepends=True),
            after_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
            lineterm="",
        ))

        logger.info(
            f"MobileCssFixer: applied '{category}' fix to {file_path}"
        )

        return FixResult(
            success=True,
            issue_id=issue_id,
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=file_path,
            before_content=page_content,
            after_content=after_content,
            diff=diff[:5000],
        )

    @staticmethod
    def _get_rule_description(category: str) -> str:
        return {
            "small_font_size": "Ensure minimum 16px font size",
            "small_touch_targets": "Ensure minimum 48px touch targets",
            "horizontal_scroll": "Prevent horizontal overflow",
        }.get(category, category)

    @staticmethod
    def _inject_css(html: str, css_rule: str) -> str:
        """Inject CSS rule into the page's <style> tag or <head>.

        Tries in order:
        1. Append to existing <style> tag in <head>
        2. Create new <style> in <head> (before </head>)
        3. Insert <style> after <head> opening tag
        4. Prepend to document if no <head> exists
        """
        # Strategy 1: Append to existing <style> tag
        style_match = re.search(
            r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE,
        )
        if style_match:
            insert_pos = style_match.end() - len("</style>")
            return html[:insert_pos] + "\n" + css_rule + "\n" + html[insert_pos:]

        # Strategy 2: Insert before </head>
        head_close = re.search(r"</head>", html, re.IGNORECASE)
        if head_close:
            style_block = f"\n<style>\n{css_rule}\n</style>\n"
            return html[:head_close.start()] + style_block + html[head_close.start():]

        # Strategy 3: After <head> opening
        head_open = re.search(r"<head[^>]*>", html, re.IGNORECASE)
        if head_open:
            style_block = f"\n<style>\n{css_rule}\n</style>\n"
            return html[:head_open.end()] + style_block + html[head_open.end():]

        # Strategy 4: After <html> or at document start
        html_open = re.search(r"<html[^>]*>", html, re.IGNORECASE)
        if html_open:
            style_block = f"<head>\n<style>\n{css_rule}\n</style>\n</head>\n"
            return html[:html_open.end()] + "\n" + style_block + html[html_open.end():]

        # Last resort: prepend
        return f"<style>\n{css_rule}\n</style>\n" + html
