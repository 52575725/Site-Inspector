from __future__ import annotations

import difflib
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class HreflangFixer(BaseFixer):
    """Auto-fix hreflang alternate links for bilingual sites."""

    fixer_name = "hreflang_fixer"
    fix_type = "semi_auto"
    supported_categories = ["missing_hreflang", "incomplete_hreflang"]

    def __init__(self, languages: dict | None = None):
        self.languages = languages or {"en": "/", "ja": "/jp/"}

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        url = issue.get("url", "")
        current_lang = self._detect_language(url)

        head = soup.find("head")
        if not head:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message="No <head> element found",
            )

        # Remove only hreflang tags matching our configured languages,
        # preserving manually-authored or third-party hreflang tags.
        configured_langs = set(self.languages.keys())
        for existing in soup.find_all("link", rel="alternate"):
            hl = existing.get("hreflang", "")
            if hl and hl != "x-default" and hl in configured_langs:
                existing.decompose()

        # Add hreflang tags for all configured languages
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path

        for lang, lang_path in self.languages.items():
            lang_url = self._build_lang_url(base, path, lang, lang_path, current_lang)
            tag = soup.new_tag("link", rel="alternate", hreflang=lang, href=lang_url)
            head.append(tag)

        # x-default (use the primary language version)
        primary_lang = list(self.languages.keys())[0]
        primary_path = list(self.languages.values())[0]
        xdefault_url = self._build_lang_url(base, path, primary_lang, primary_path,
                                            current_lang)
        xdefault_tag = soup.new_tag("link", rel="alternate", hreflang="x-default",
                                    href=xdefault_url)
        head.append(xdefault_tag)

        new_content = str(soup)
        diff = difflib.unified_diff(
            page_content.splitlines(True), new_content.splitlines(True), lineterm="",
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

    def _detect_language(self, url: str) -> str:
        """Detect page language from URL path using configured language paths."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        for lang, lang_path in self.languages.items():
            if lang_path != "/" and lang_path.rstrip("/") in path:
                return lang
        return list(self.languages.keys())[0]

    def _build_lang_url(self, base: str, path: str, lang: str,
                         lang_path: str, current_lang: str) -> str:
        """Build the URL for a specific language version."""
        if lang == current_lang:
            return f"{base}{path}"

        primary_lang = list(self.languages.keys())[0]
        current_prefix = self.languages.get(current_lang, "/").rstrip("/")
        target_prefix = lang_path.rstrip("/")

        if current_lang == primary_lang:
            # From primary (/) to alternate (/jp/): prepend target prefix
            if path in ("/", ""):
                return f"{base}{target_prefix}/"
            return f"{base}{target_prefix}{path}"
        else:
            # From alternate to primary: remove current prefix
            path = path or "/"
            prefix_with_slash = current_prefix + "/"
            if path.startswith(prefix_with_slash):
                stripped = path[len(prefix_with_slash):]
                return f"{base}/{stripped}"
            # Prefix stripping failed — keep the original path instead of
            # dropping everything to root, which would create broken hreflangs.
            logger.warning(
                f"hreflang: path '{path}' doesn't start with expected prefix "
                f"'{prefix_with_slash}', keeping original path"
            )
            return f"{base}{path}"
