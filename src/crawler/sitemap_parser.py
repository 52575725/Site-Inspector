from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass
class DiscoveredPage:
    url: str
    language: str = "en"
    priority: float = 0.5
    lastmod: str | None = None
    changefreq: str | None = None


class SitemapParser:
    """Parse sitemap.xml to discover all pages on the target site."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def parse(self, sitemap_url: str) -> list[DiscoveredPage]:
        """Fetch and parse a sitemap. Follows sitemap index files recursively."""
        try:
            resp = await self.client.get(sitemap_url, follow_redirects=True)
            resp.raise_for_status()
            content = resp.text
        except Exception:
            return []

        soup = BeautifulSoup(content, "xml")

        # Check for sitemap index
        sitemaps = soup.find_all("sitemap")
        if sitemaps:
            pages = []
            for sm in sitemaps:
                loc = sm.find("loc")
                if loc and loc.text:
                    sub_pages = await self.parse(loc.text.strip())
                    pages.extend(sub_pages)
            return pages

        # Parse URL entries
        pages = []
        base_url = self._get_base_url(sitemap_url)

        for url_elem in soup.find_all("url"):
            loc = url_elem.find("loc")
            if not loc or not loc.text:
                continue
            url = loc.text.strip()

            priority = 0.5
            prio_elem = url_elem.find("priority")
            if prio_elem and prio_elem.text:
                try:
                    priority = float(prio_elem.text)
                except ValueError:
                    pass

            lastmod = None
            lm_elem = url_elem.find("lastmod")
            if lm_elem and lm_elem.text:
                lastmod = lm_elem.text.strip()

            changefreq = None
            cf_elem = url_elem.find("changefreq")
            if cf_elem and cf_elem.text:
                changefreq = cf_elem.text.strip()

            language = self._detect_language(url, base_url)

            pages.append(DiscoveredPage(
                url=url,
                language=language,
                priority=priority,
                lastmod=lastmod,
                changefreq=changefreq,
            ))

        return pages

    @staticmethod
    def _detect_language(url: str, base_url: str) -> str:
        """Detect page language from URL path."""
        parsed = urlparse(url)
        path = parsed.path.lower()

        if "/jp/" in path or path.startswith("/jp") or path.endswith("/jp"):
            return "jp"
        # Check for other common language codes
        for lang in ["zh", "ko", "es", "de", "fr"]:
            if f"/{lang}/" in path or path.startswith(f"/{lang}") or path.endswith(f"/{lang}"):
                return lang
        return "en"

    @staticmethod
    def _get_base_url(sitemap_url: str) -> str:
        parsed = urlparse(sitemap_url)
        return f"{parsed.scheme}://{parsed.netloc}"


class RobotsTxtParser:
    """Parse robots.txt for sitemap references and crawl rules."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def parse(self, base_url: str) -> dict:
        """Parse robots.txt and return {sitemaps, crawl_delay, disallowed}."""
        robots_url = urljoin(base_url, "/robots.txt")
        result = {"sitemaps": [], "crawl_delay": None, "disallowed": []}

        try:
            resp = await self.client.get(robots_url, follow_redirects=True)
            resp.raise_for_status()
            content = resp.text
        except Exception:
            return result

        for line in content.splitlines():
            line = line.strip().lower()

            if line.startswith("sitemap:"):
                url = line.split(":", 1)[1].strip()
                result["sitemaps"].append(url)

            elif line.startswith("crawl-delay:"):
                try:
                    result["crawl_delay"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                result["disallowed"].append(path)

        return result

    async def get_sitemap_urls(self, base_url: str) -> list[str]:
        """Get sitemap URLs from robots.txt."""
        info = await self.parse(base_url)
        return info["sitemaps"]
