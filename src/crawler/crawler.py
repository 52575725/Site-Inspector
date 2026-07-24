from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from config.settings import Settings
from src.crawler.rate_limiter import TokenBucketRateLimiter, backoff_delay
from src.crawler.sitemap_parser import DiscoveredPage, RobotsTxtParser, SitemapParser

logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    url: str
    language: str
    title: str | None
    http_status: int
    html_content: str
    load_time_ms: int
    html_size_bytes: int
    headers: dict


class Crawler:
    """Polite full-site crawler with rate limiting."""

    def __init__(self, settings: Settings, base_url: str | None = None):
        self.settings = settings
        self.base_url = base_url or settings.target_base_url
        self.max_concurrent = settings.crawl_max_concurrent
        self.rate_limiter = TokenBucketRateLimiter(
            rate=settings.crawl_rate, burst=settings.crawl_max_concurrent
        )
        self.semaphore = asyncio.Semaphore(settings.crawl_max_concurrent)
        self.client = httpx.AsyncClient(
            timeout=settings.crawl_timeout,
            headers={"User-Agent": settings.crawl_user_agent},
            follow_redirects=False,
        )

    async def discover_pages(self) -> list[DiscoveredPage]:
        """Discover all pages via sitemap.xml and robots.txt."""
        pages: list[DiscoveredPage] = []

        # Try sitemap directly first
        sitemap_url = f"{self.base_url}/sitemap.xml"
        parser = SitemapParser(self.client)
        pages = await parser.parse(sitemap_url)

        # Fallback to robots.txt if sitemap was empty
        if not pages:
            robots = RobotsTxtParser(self.client)
            sitemap_urls = await robots.get_sitemap_urls(self.base_url)
            for url in sitemap_urls:
                sub_pages = await parser.parse(url)
                pages.extend(sub_pages)

        # Fallback: crawl homepage and extract same-domain links
        if not pages:
            pages = await self._discover_from_homepage()

        # Filter excluded patterns
        base_host = urlparse(self.base_url).hostname
        base_host_alt = "www." + base_host if not base_host.startswith("www.") else base_host[4:]

        def _same_host(url_host: str) -> bool:
            return url_host == base_host or url_host == base_host_alt

        pages = [
            page for page in pages
            if urlparse(page.url).scheme in {"http", "https"}
            and _same_host(urlparse(page.url).hostname)
        ]
        if len(pages) > self.settings.crawl_max_pages:
            logger.warning(
                f"Discovered {len(pages)} pages; limiting scan to "
                f"{self.settings.crawl_max_pages}"
            )
            pages = pages[:self.settings.crawl_max_pages]
        logger.info(f"Discovered {len(pages)} pages")
        return pages

    async def _discover_from_homepage(self) -> list[DiscoveredPage]:
        """Extract same-domain links from the homepage."""
        from urllib.parse import urljoin, urlparse
        import re

        logger.info(f"No sitemap found, crawling homepage for links: {self.base_url}")
        try:
            resp = await self._get_same_origin(self.base_url)
            if resp.status_code >= 400:
                logger.warning(f"Homepage returned {resp.status_code}")
                return []
            if "text/html" not in resp.headers.get("content-type", ""):
                return [DiscoveredPage(url=self.base_url, language="en")]
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as e:
            logger.warning(f"Failed to fetch homepage: {e}")
            return []

        base_host = urlparse(self.base_url).hostname
        base_host_alt = "www." + base_host if not base_host.startswith("www.") else base_host[4:]

        def _same_host(h: str) -> bool:
            return h == base_host or h == base_host_alt

        seen: set[str] = {self.base_url}
        pages = [DiscoveredPage(url=self.base_url, language="en")]

        hrefs = re.findall(r'href=["\']([^"\']+?)["\']', resp.text, re.IGNORECASE)
        for href in hrefs:
            full = urljoin(self.base_url, href.split("#")[0])
            parsed = urlparse(full)
            if not _same_host(parsed.hostname):
                continue
            if parsed.scheme not in ("http", "https"):
                continue
            if full in seen:
                continue
            seen.add(full)
            pages.append(DiscoveredPage(url=full, language="en"))

        logger.info(f"Extracted {len(pages)} links from homepage")
        return pages

    async def crawl_page(self, page: DiscoveredPage) -> Optional[CrawledPage]:
        """Crawl a single page with rate limiting and retries."""
        async with self.semaphore:
            await self.rate_limiter.acquire()

            for attempt in range(self.settings.crawl_max_retries):
                try:
                    start = asyncio.get_event_loop().time()
                    resp = await self._get_same_origin(page.url)
                    elapsed_ms = int((asyncio.get_event_loop().time() - start) * 1000)

                    if resp.status_code == 429:
                        delay = backoff_delay(attempt)
                        logger.warning(f"429 from {page.url}, waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue

                    title = None
                    if "text/html" in resp.headers.get("content-type", ""):
                        import re
                        match = re.search(r"<title>(.*?)</title>", resp.text, re.IGNORECASE)
                        if match:
                            title = match.group(1).strip()

                    return CrawledPage(
                        url=page.url,
                        language=page.language,
                        title=title,
                        http_status=resp.status_code,
                        html_content=resp.text,
                        load_time_ms=elapsed_ms,
                        html_size_bytes=len(resp.content),
                        headers=dict(resp.headers),
                    )
                except httpx.TimeoutException:
                    logger.warning(f"Timeout for {page.url}, attempt {attempt + 1}")
                    if attempt < self.settings.crawl_max_retries - 1:
                        await asyncio.sleep(backoff_delay(attempt))
                except Exception as e:
                    logger.error(f"Error crawling {page.url}: {e}")
                    break

            return CrawledPage(
                url=page.url,
                language=page.language,
                title=None,
                http_status=0,
                html_content="",
                load_time_ms=0,
                html_size_bytes=0,
                headers={},
            )

    async def crawl_all(self, pages: list[DiscoveredPage]) -> list[CrawledPage]:
        """Crawl all discovered pages concurrently with rate limiting."""
        tasks = [self.crawl_page(p) for p in pages]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def _get_same_origin(self, url: str) -> httpx.Response:
        """Follow redirects only while they remain on the configured host."""
        base_host = urlparse(self.base_url).hostname
        base_host_alt = "www." + base_host if not base_host.startswith("www.") else base_host[4:]

        def _same_host(h: str) -> bool:
            return h == base_host or h == base_host_alt

        current = url
        for _ in range(6):
            parsed = urlparse(current)
            if parsed.scheme not in {"http", "https"} or not _same_host(parsed.hostname):
                raise ValueError(f"Cross-origin crawl URL rejected: {current}")
            response = await self.client.get(current)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                return response
            current = urljoin(current, location)
        raise httpx.TooManyRedirects("Too many redirects", request=response.request)

    async def close(self) -> None:
        await self.client.aclose()
