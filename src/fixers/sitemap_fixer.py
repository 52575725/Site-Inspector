from __future__ import annotations

import difflib
from datetime import datetime
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class SitemapFixer(BaseFixer):
    """Auto-regenerate sitemap.xml: remove dead URLs, add missing pages."""

    fixer_name = "sitemap_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "sitemap_dead_url", "sitemap_missing_url",
        "sitemap_missing_hreflang", "sitemap_missing",
    ]

    def __init__(self, language_paths: dict | None = None):
        self._lang_paths = language_paths or {"en": "/"}

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        url = issue.get("url", "")
        category = issue.get("category", "")

        if category == "sitemap_missing":
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message="Sitemap file not found; cannot auto-create from scratch",
            )

        before = page_content

        if category == "sitemap_dead_url":
            page_content = self._remove_dead_url(page_content, url)

        if category == "sitemap_missing_url":
            page_content = self._add_missing_url(page_content, url)

        if category == "sitemap_missing_hreflang":
            page_content = self._add_hreflang_to_entry(page_content, url)

        diff = difflib.unified_diff(
            before.splitlines(True),
            page_content.splitlines(True),
            lineterm="",
        )

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=before,
            after_content=page_content,
            diff="\n".join(diff),
        )

    def _remove_dead_url(self, xml_content: str, dead_url: str) -> str:
        """Remove a <url> entry for a dead URL."""
        soup = BeautifulSoup(xml_content, "xml")
        for url_elem in soup.find_all("url"):
            loc = url_elem.find("loc")
            if loc and loc.text.strip().rstrip("/") == dead_url.rstrip("/"):
                url_elem.decompose()
                break
        return str(soup)

    def _add_missing_url(self, xml_content: str, new_url: str) -> str:
        """Add a <url> entry for a missing page."""
        soup = BeautifulSoup(xml_content, "xml")
        urlset = soup.find("urlset")
        if not urlset:
            return xml_content

        # Check if already exists
        for url_elem in soup.find_all("url"):
            loc = url_elem.find("loc")
            if loc and loc.text.strip().rstrip("/") == new_url.rstrip("/"):
                return xml_content

        today = datetime.utcnow().strftime("%Y-%m-%d")
        priority = self._guess_priority(new_url)
        changefreq = self._guess_changefreq(new_url)

        new_elem = soup.new_tag("url")
        loc_tag = soup.new_tag("loc")
        loc_tag.string = new_url
        new_elem.append(loc_tag)

        lm_tag = soup.new_tag("lastmod")
        lm_tag.string = today
        new_elem.append(lm_tag)

        cf_tag = soup.new_tag("changefreq")
        cf_tag.string = changefreq
        new_elem.append(cf_tag)

        prio_tag = soup.new_tag("priority")
        prio_tag.string = str(priority)
        new_elem.append(prio_tag)

        # Add hreflang alternates for multilingual pages
        counterpart = self._get_counterpart_url(new_url)
        if counterpart:
            lang = self._lang_code_for_url(new_url)
            alt_lang = self._lang_code_for_url(counterpart)
            link1 = soup.new_tag("xhtml:link", rel="alternate",
                                 hreflang=lang, href=new_url)
            link2 = soup.new_tag("xhtml:link", rel="alternate",
                                 hreflang=alt_lang, href=counterpart)
            new_elem.append(link1)
            new_elem.append(link2)

        urlset.append(new_elem)
        return str(soup)

    def _add_hreflang_to_entry(self, xml_content: str, url: str) -> str:
        """Add xhtml:link hreflang alternates to an existing sitemap entry."""
        soup = BeautifulSoup(xml_content, "xml")
        for url_elem in soup.find_all("url"):
            loc = url_elem.find("loc")
            if not loc or loc.text.strip().rstrip("/") != url.rstrip("/"):
                continue

            existing = url_elem.find_all("xhtml:link")
            if existing:
                continue

            lang = self._lang_code_for_url(url)
            counterpart = self._get_counterpart_url(url)
            alt_lang = self._lang_code_for_url(counterpart) if counterpart else list(self._lang_paths.keys())[0]

            link1 = soup.new_tag("xhtml:link", rel="alternate",
                                 hreflang=lang, href=url)
            link2 = soup.new_tag("xhtml:link", rel="alternate",
                                 hreflang=alt_lang, href=counterpart)
            url_elem.append(link1)
            url_elem.append(link2)
            break

        return str(soup)

    def _guess_priority(self, url: str) -> float:
        path = urlparse(url).path.rstrip("/")
        if path in ("", "/"):
            return 1.0
        alt_prefixes = [v.strip("/") for v in self._lang_paths.values() if v != "/"]
        for prefix in alt_prefixes:
            if path == f"/{prefix}":
                return 0.9
        if "/products/" in path:
            return 0.8
        if "/blog/" in path and path.count("/") > 2:
            return 0.7
        if "/about/" in path or "/contact/" in path:
            return 0.7
        if "/blog" in path:
            return 0.5
        return 0.5

    def _guess_changefreq(self, url: str) -> str:
        path = urlparse(url).path.rstrip("/")
        if path in ("", "/"):
            return "weekly"
        alt_prefixes = [v.strip("/") for v in self._lang_paths.values() if v != "/"]
        for prefix in alt_prefixes:
            if path == f"/{prefix}":
                return "weekly"
        if "/blog" in path:
            return "weekly"
        return "monthly"

    def _get_counterpart_url(self, url: str) -> str | None:
        """Get the counterpart URL for a different language version."""
        parsed = urlparse(url)
        path = parsed.path
        alt_langs = {k: v.strip("/") for k, v in self._lang_paths.items() if v != "/"}

        for lang, lang_path in alt_langs.items():
            prefix = f"/{lang_path}"
            if path.startswith(f"{prefix}/") or path == prefix:
                # Strip alt language prefix from path
                new_path = path[len(prefix):] if path != prefix else "/"
                if not new_path:
                    new_path = "/"
                return f"{parsed.scheme}://{parsed.netloc}{new_path}"

        # URL is in primary language — render first alt language counterpart
        if alt_langs:
            first_alt = list(alt_langs.values())[0]
            if path in ("/", ""):
                return f"{parsed.scheme}://{parsed.netloc}/{first_alt}/"
            return f"{parsed.scheme}://{parsed.netloc}/{first_alt}{path}"
        return None

    def _lang_code_for_url(self, url: str) -> str:
        """Determine ISO language code for a URL based on configured language paths."""
        primary = list(self._lang_paths.keys())[0]
        for lang, path in self._lang_paths.items():
            prefix = path.strip("/")
            if prefix and f"/{prefix}/" in url:
                return lang
        return primary
