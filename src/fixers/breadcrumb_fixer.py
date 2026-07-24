from __future__ import annotations

import difflib
import json
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class BreadcrumbFixer(BaseFixer):
    """Generate BreadcrumbList JSON-LD for pages that have other schemas but miss it."""

    fixer_name = "breadcrumb_fixer"
    fix_type = "fully_auto"
    supported_categories = ["schema_missing_type", "missing_breadcrumb"]

    def __init__(self, language_paths: dict | None = None,
                 slug_translations: dict | None = None):
        """Args:
            language_paths: e.g. {"en": "/", "ja": "/jp/"}. Default: single-language.
            slug_translations: Optional per-language slug→name mappings.
                e.g. {"ja": {"products": "製品", "about": "会社概要"}}
        """
        self._lang_paths = language_paths or {"en": "/"}
        self._slug_translations = slug_translations or {}

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        url = issue.get("url", "")
        category = issue.get("category", "")
        suggested = issue.get("suggested_value", "")

        # Only act if the missing type is BreadcrumbList
        if category == "schema_missing_type" and "BreadcrumbList" not in suggested:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message=f"Missing type is not BreadcrumbList: {suggested}",
            )

        ld_json = self._generate_breadcrumb(url)
        if not ld_json:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message="Could not generate breadcrumb from URL",
            )

        script_tag = soup.new_tag("script", type="application/ld+json")
        script_tag.string = json.dumps(ld_json, ensure_ascii=False, indent=2)

        head = soup.find("head")
        if head:
            head.append(script_tag)
        else:
            soup.insert(0, script_tag)

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

    def _detect_lang(self, url: str) -> tuple[str, str]:
        """Detect language code and path prefix from URL. Returns (lang_code, prefix)."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        for lang, lang_path in self._lang_paths.items():
            prefix = lang_path.strip("/")
            if prefix and prefix != "/" and f"/{prefix}/" in path:
                return lang, prefix
        primary = list(self._lang_paths.keys())[0]
        return primary, ""

    def _generate_breadcrumb(self, url: str) -> dict | None:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return None

        lang_code, lang_prefix = self._detect_lang(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        # Build home URL
        if lang_prefix:
            home_url = f"{domain}/{lang_prefix}/"
        else:
            home_url = f"{domain}/"

        parts = path.split("/")
        # Skip language prefix segments
        url_parts = [p for p in parts if p != lang_prefix]

        items = []
        position = 1
        accumulated = lang_prefix if lang_prefix else ""

        # Home
        home_name = self._get_translation("home", lang_code) or "Home"
        items.append({
            "@type": "ListItem",
            "position": position,
            "name": home_name,
            "item": home_url,
        })
        position += 1

        for part in url_parts:
            accumulated += f"/{part}"
            name = self._path_to_name(part, lang_code)
            items.append({
                "@type": "ListItem",
                "position": position,
                "name": name,
                "item": f"{domain}{accumulated}/",
            })
            position += 1

        if len(items) <= 1:
            return None

        return {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": items,
        }

    def _path_to_name(self, part: str, lang_code: str) -> str:
        """Convert a URL slug to a human-readable name. Uses configured translations
        when available, otherwise falls back to generic title-case conversion."""
        lang_translations = self._slug_translations.get(lang_code, {})
        if part in lang_translations:
            return lang_translations[part]
        return part.replace("-", " ").title()

    def _get_translation(self, slug: str, lang_code: str) -> str | None:
        lang_translations = self._slug_translations.get(lang_code, {})
        return lang_translations.get(slug)
