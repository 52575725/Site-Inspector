from __future__ import annotations

import difflib
import logging
import re

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)

# Platform-specific config
PLATFORM_META = {
    "google": "google-site-verification",
    "baidu": "baidu-site-verification",
    "bing": "msvalidate.01",
    "yandex": "yandex-verification",
}


class VerificationFixer(BaseFixer):
    """Adds search-engine verification meta tags to the homepage <head>.

    The verification CODE must be provided by the user (obtained from
    each platform's webmaster console). This fixer inserts it as a
    <meta> tag in <head>.

    Fix type is "semi_auto" because the user needs to supply the actual
    verification code from each platform.
    """

    fixer_name = "verification_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "platform_missing_google_verify",
        "platform_missing_baidu_verify",
        "platform_missing_bing_verify",
        "platform_missing_yandex_verify",
    ]

    # User-provided verification codes (set via env or config)
    # Format: {"google": "xxx", "baidu": "xxx", "bing": "xxx"}

    def __init__(self, verification_codes: dict[str, str] | None = None):
        self.codes = verification_codes or {}

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        category = issue.get("category", "")
        file_path = issue.get("file_path", "")
        issue_id = issue.get("id", 0)
        metadata = issue.get("raw_metadata", {}) or {}

        # Determine which platform
        platform = metadata.get("platform", "").lower()
        if not platform:
            # Fallback from category name
            for p in ["google", "baidu", "bing", "yandex"]:
                if p in category:
                    platform = p
                    break

        if not platform or platform not in PLATFORM_META:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message=f"Cannot determine platform from category: {category}",
            )

        meta_name = PLATFORM_META[platform]
        code = self.codes.get(platform, "")

        if not code:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message=(
                    f"No verification code for {platform}. "
                    f"Get your code from the platform webmaster console "
                    f"and set it in config/targets.yaml under "
                    f"verification.{platform}."
                ),
            )

        # Insert <meta name="xxx" content="code" /> in <head>
        meta_tag = f'<meta name="{meta_name}" content="{code}" />'

        # Check if already present
        if meta_tag in page_content:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message=f"{platform} verification tag already present",
            )

        # Insert before </head>
        head_close = re.search(r"</head>", page_content, re.IGNORECASE)
        if not head_close:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message="No </head> tag found — cannot insert verification meta",
            )

        after_content = (
            page_content[:head_close.start()]
            + f"\n{meta_tag}\n"
            + page_content[head_close.start():]
        )

        diff = "\n".join(difflib.unified_diff(
            page_content.splitlines(keepends=True),
            after_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
            lineterm="",
        ))

        logger.info(f"VerificationFixer: added {platform} verification tag")

        return FixResult(
            success=True,
            issue_id=issue_id,
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=file_path,
            before_content=page_content,
            after_content=after_content,
            diff=diff,
        )
