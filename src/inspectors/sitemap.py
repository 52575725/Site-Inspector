from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx
from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


class SitemapInspector(BaseInspector):
    """Validate sitemap.xml: dead URLs, missing pages, stale lastmod, missing hreflang."""

    inspector_name = "sitemap"

    def __init__(self):
        self._crawled_urls: list[str] = []
        self._sitemap_url: str = ""
        self._base_url: str = ""
        self._already_checked: bool = False
        self._language_paths: dict[str, str] = {}

    def set_language_paths(self, paths: dict[str, str]) -> None:
        self._language_paths = paths

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_crawled_urls(self, urls: list[str]) -> None:
        self._crawled_urls = urls

    def set_sitemap_url(self, sitemap_url: str) -> None:
        self._sitemap_url = sitemap_url

    def set_base_url(self, base_url: str) -> None:
        self._base_url = base_url

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        if self._already_checked:
            return []
        self._already_checked = True

        if not self._sitemap_url:
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="sitemap_missing",
                description="No sitemap URL configured for this target",
            )]

        findings: list[RawFinding] = []

        try:
            sitemap_entries = await self._fetch_sitemap(self._sitemap_url)
        except Exception as e:
            return [RawFinding(
                url=self._sitemap_url, inspector=self.inspector_name,
                category="sitemap_missing",
                description=f"Cannot fetch or parse sitemap: {str(e)[:200]}",
            )]

        if not sitemap_entries:
            return [RawFinding(
                url=self._sitemap_url, inspector=self.inspector_name,
                category="sitemap_missing",
                description="Sitemap is empty or could not be parsed",
            )]

        sitemap_urls = set(sitemap_entries.keys())
        crawled_set = self._normalize_urls(self._crawled_urls)

        # 1. Dead URLs: in sitemap but not in crawled pages
        dead_urls = sitemap_urls - crawled_set
        for dead_url in sorted(dead_urls):
            findings.append(RawFinding(
                url=dead_url, inspector=self.inspector_name,
                category="sitemap_dead_url",
                description=f"URL in sitemap but not found on site: {dead_url}",
                current_value=dead_url,
            ))

        # 2. Missing URLs: in crawled pages but not in sitemap
        missing_urls = crawled_set - sitemap_urls
        for missing_url in sorted(missing_urls):
            findings.append(RawFinding(
                url=missing_url, inspector=self.inspector_name,
                category="sitemap_missing_url",
                description=f"Page exists but is not listed in sitemap: {missing_url}",
                current_value=missing_url,
            ))

        # 3. Stale lastmod (older than 30 days)
        threshold = (datetime.utcnow() - timedelta(days=30)).date()
        for sitemap_url_str, entry in sitemap_entries.items():
            lastmod = entry.get("lastmod")
            if lastmod:
                try:
                    lm_date = datetime.strptime(lastmod, "%Y-%m-%d").date()
                    if lm_date < threshold:
                        findings.append(RawFinding(
                            url=sitemap_url_str, inspector=self.inspector_name,
                            category="sitemap_stale_lastmod",
                            description=f"Sitemap lastmod is stale ({lastmod}, >30 days ago)",
                            current_value=lastmod,
                        ))
                except ValueError:
                    pass

        # 4. Missing hreflang alternates in sitemap (for multilingual pages)
        lang_paths = self._language_paths
        if lang_paths and len(lang_paths) > 1:
            primary_path = list(lang_paths.values())[0].strip("/")
            alt_langs = {k: v.strip("/") for k, v in lang_paths.items()
                        if v.strip("/") != primary_path}

            for sitemap_url_str, entry in sitemap_entries.items():
                counterpart = None
                for alt_lang, alt_path in alt_langs.items():
                    if f"/{alt_path}/" in sitemap_url_str:
                        counterpart = sitemap_url_str.replace(f"/{alt_path}/", "/")
                        counterpart = counterpart.replace("//", "/")
                        break

                if counterpart and counterpart in sitemap_entries:
                    hreflangs = entry.get("hreflangs", [])
                    if not hreflangs:
                        findings.append(RawFinding(
                            url=sitemap_url_str, inspector=self.inspector_name,
                            category="sitemap_missing_hreflang",
                            description=f"Sitemap entry missing xhtml:link hreflang alternates: {sitemap_url_str}",
                            current_value=sitemap_url_str,
                        ))

        return findings

    async def _fetch_sitemap(self, sitemap_url: str) -> dict[str, dict]:
        """Fetch and parse sitemap.xml. Returns {url: {lastmod, changefreq, hreflangs}}."""
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(sitemap_url)
                resp.raise_for_status()
                content = resp.text
        except Exception as e:
            logger.warning(f"Failed to fetch sitemap {sitemap_url}: {e}")
            raise

        soup = BeautifulSoup(content, "xml")

        # Check for sitemap index
        sitemaps = soup.find_all("sitemap")
        if sitemaps:
            entries = {}
            for sm in sitemaps:
                loc = sm.find("loc")
                if loc and loc.text:
                    try:
                        sub_entries = await self._fetch_sitemap(loc.text.strip())
                        entries.update(sub_entries)
                    except Exception:
                        pass
            return entries

        entries: dict[str, dict] = {}
        for url_elem in soup.find_all("url"):
            loc = url_elem.find("loc")
            if not loc or not loc.text:
                continue

            url_str = loc.text.strip()
            entry: dict = {}

            lm = url_elem.find("lastmod")
            if lm and lm.text:
                entry["lastmod"] = lm.text.strip()

            cf = url_elem.find("changefreq")
            if cf and cf.text:
                entry["changefreq"] = cf.text.strip()

            # Extract hreflang alternates
            hreflangs = []
            for link in url_elem.find_all("xhtml:link"):
                hreflangs.append({
                    "hreflang": link.get("hreflang", ""),
                    "href": link.get("href", ""),
                })
            entry["hreflangs"] = hreflangs

            entries[url_str] = entry

        return entries

    @staticmethod
    def _normalize_urls(urls: list[str]) -> set[str]:
        """Normalize URLs to one identity instead of inventing slash variants."""
        result = set()
        for u in urls:
            normalized = u.rstrip("/") or u
            result.add(normalized)
        return result
