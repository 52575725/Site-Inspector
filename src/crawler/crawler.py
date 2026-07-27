from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
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
    """Polite site crawler with rate limiting and optional recursive BFS discovery.

    Supports two fetch modes:
    - httpx (default): fast, works for server-rendered pages
    - Playwright (opt-in): renders JavaScript, needed for SPA / CSR sites.
      Enable per-target in targets.yaml: crawl.use_browser: true
    """

    def __init__(self, settings: Settings, base_url: str | None = None,
                 use_browser: bool = False):
        self.settings = settings
        self.base_url = base_url or settings.target_base_url
        self.max_concurrent = settings.crawl_max_concurrent
        self.max_depth = getattr(settings, "crawl_max_depth", 2)
        self.use_browser = use_browser
        self.rate_limiter = TokenBucketRateLimiter(
            rate=settings.crawl_rate, burst=settings.crawl_max_concurrent
        )
        self.semaphore = asyncio.Semaphore(settings.crawl_max_concurrent)
        self.client = httpx.AsyncClient(
            timeout=settings.crawl_timeout,
            headers={"User-Agent": settings.crawl_user_agent},
            follow_redirects=False,
        )
        self._browser = None
        self._playwright = None

    # ── Discovery ──────────────────────────────────────────────────────

    async def discover_pages(self) -> list[DiscoveredPage]:
        """Discover pages via sitemap → robots.txt → homepage → recursive BFS.

        When crawl_max_depth > 1, supplements sitemap discovery with recursive
        link-following so pages omitted from the sitemap are not missed.
        """
        pages: list[DiscoveredPage] = []
        seen_urls: set[str] = set()

        # 1. Try sitemap first
        sitemap_url = f"{self.base_url}/sitemap.xml"
        parser = SitemapParser(self.client)
        sitemap_pages = await parser.parse(sitemap_url)

        if not sitemap_pages:
            robots = RobotsTxtParser(self.client)
            sitemap_urls = await robots.get_sitemap_urls(self.base_url)
            for url in sitemap_urls:
                sub_pages = await parser.parse(url)
                sitemap_pages.extend(sub_pages)

        for p in sitemap_pages:
            if p.url not in seen_urls:
                seen_urls.add(p.url)
                pages.append(p)

        logger.info(f"Sitemap discovered {len(pages)} pages")

        # 2. Recursive BFS crawl to discover pages not in sitemap
        if self.max_depth > 0:
            bfs_pages = await self._discover_deep(list(seen_urls))
            for url in bfs_pages:
                if url not in seen_urls:
                    seen_urls.add(url)
                    pages.append(DiscoveredPage(url=url, language="en"))
            logger.info(f"BFS discovered {len(bfs_pages)} additional pages "
                        f"({len(pages)} total after merge)")

        # 3. Fallback: homepage link extraction if nothing found
        if not pages:
            pages = await self._discover_from_homepage()
            for p in pages:
                seen_urls.add(p.url)

        # 4. Filter to same-host, apply max_pages cap
        pages = self._filter_same_host(pages)
        if len(pages) > self.settings.crawl_max_pages:
            logger.warning(
                f"Discovered {len(pages)} pages; limiting scan to "
                f"{self.settings.crawl_max_pages}"
            )
            pages = pages[: self.settings.crawl_max_pages]
        logger.info(f"Final page list: {len(pages)} pages")
        return pages

    async def _discover_deep(self, seed_urls: list[str]) -> list[str]:
        """BFS recursive discovery: follow internal links up to max_depth.

        Seeds from sitemap/homepage. Each discovered page is fetched and its
        internal <a href> links are added to the queue for the next depth level.
        """
        discovered: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        visited_html: set[str] = set()  # pages whose HTML we've already fetched

        base_host = urlparse(self.base_url).hostname
        base_host_alt = (
            "www." + base_host if not base_host.startswith("www.") else base_host[4:]
        )

        def _same_host(h: str) -> bool:
            return h == base_host or h == base_host_alt

        # Seed with 0 depth so they get fetched and their links discovered
        for url in seed_urls:
            queue.append((url, 0))

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _fetch_and_extract(url: str, depth: int) -> list[str]:
            """Fetch one page and extract its internal links."""
            async with sem:
                try:
                    resp = await self._get_same_origin(url)
                    if resp.status_code >= 400:
                        return []
                    ct = resp.headers.get("content-type", "")
                    if "text/html" not in ct:
                        return []
                    links = self._extract_internal_links(resp.text, url, _same_host)
                    return links
                except Exception:
                    return []

        pending_tasks: dict[str, asyncio.Task] = {}

        while queue:
            # Collect all URLs at the current front depth level
            batch: list[tuple[str, int]] = []
            if queue:
                batch.append(queue.popleft())
            current_depth = batch[0][1] if batch else -1

            # Don't exceed max_depth
            if current_depth > self.max_depth:
                continue

            # Launch concurrent fetches for this depth level
            tasks_map: dict[str, tuple[str, int]] = {}
            for url, d in batch:
                if url in visited_html:
                    continue
                visited_html.add(url)
                tasks_map[url] = (url, d)

            if not tasks_map:
                continue

            results = await asyncio.gather(
                *[_fetch_and_extract(url, d) for url, d in tasks_map.values()],
                return_exceptions=True,
            )

            for (url, depth), result in zip(tasks_map.values(), results):
                if isinstance(result, Exception):
                    continue
                if not isinstance(result, list):
                    continue
                discovered.add(url)
                for link in result:
                    if link not in discovered and link not in visited_html:
                        discovered.add(link)
                        queue.append((link, depth + 1))

            # Respect rate limits between depth levels
            await asyncio.sleep(0.5)

            # Safety valve
            total = len(discovered) + len(visited_html)
            if total >= self.settings.crawl_max_pages * 2:
                logger.warning(
                    f"BFS discovery hit safety limit ({total} URLs), stopping"
                )
                break

        return list(discovered - set(seed_urls))

    @staticmethod
    def _extract_internal_links(
        html: str, base_url: str, same_host_check
    ) -> list[str]:
        """Extract same-domain <a href> links from HTML."""
        links: list[str] = []
        hrefs = re.findall(r'<a\s[^>]*?href=["\']([^"\']+?)["\']', html, re.IGNORECASE)
        seen: set[str] = set()

        for href in hrefs:
            # Skip anchors, javascript, mailto, tel
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full = urljoin(base_url, href.split("#")[0].split("?")[0])
            parsed = urlparse(full)
            if not same_host_check(parsed.hostname):
                continue
            if parsed.scheme not in ("http", "https"):
                continue
            # Skip binary / asset paths
            path_lower = parsed.path.lower()
            if any(
                path_lower.endswith(ext)
                for ext in (
                    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
                    ".pdf", ".zip", ".css", ".js", ".ico", ".woff",
                    ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".mov",
                )
            ):
                continue
            if full not in seen:
                seen.add(full)
                links.append(full)

        return links

    # ── Homepage fallback ──────────────────────────────────────────────

    async def _discover_from_homepage(self) -> list[DiscoveredPage]:
        """Extract same-domain links from the homepage."""
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
        base_host_alt = (
            "www." + base_host if not base_host.startswith("www.") else base_host[4:]
        )

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

    # ── Page crawling ──────────────────────────────────────────────────

    async def crawl_page(self, page: DiscoveredPage) -> CrawledPage | None:
        """Crawl a single page with rate limiting and retries."""
        async with self.semaphore:
            await self.rate_limiter.acquire()

            for attempt in range(self.settings.crawl_max_retries):
                try:
                    start = asyncio.get_event_loop().time()

                    if self.use_browser:
                        html, status, resp_headers = await self._fetch_rendered(page.url)
                    else:
                        resp = await self._get_same_origin(page.url)
                        html = resp.text
                        status = resp.status_code
                        resp_headers = dict(resp.headers)

                    elapsed_ms = int(
                        (asyncio.get_event_loop().time() - start) * 1000
                    )

                    if status == 429:
                        delay = backoff_delay(attempt)
                        logger.warning(f"429 from {page.url}, waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue

                    title = None
                    if html and "text/html" in (
                        resp_headers.get("content-type", "")
                        if not self.use_browser else "text/html"
                    ):
                        match = re.search(
                            r"<title>(.*?)</title>", html, re.IGNORECASE
                        )
                        if match:
                            title = match.group(1).strip()

                    return CrawledPage(
                        url=page.url,
                        language=page.language,
                        title=title,
                        http_status=status,
                        html_content=html,
                        load_time_ms=elapsed_ms,
                        html_size_bytes=len(html.encode()) if html else 0,
                        headers=resp_headers,
                    )
                except httpx.TimeoutException:
                    logger.warning(
                        f"Timeout for {page.url}, attempt {attempt + 1}"
                    )
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

    async def crawl_all(
        self, pages: list[DiscoveredPage]
    ) -> list[CrawledPage]:
        """Crawl all discovered pages concurrently with rate limiting."""
        tasks = [self.crawl_page(p) for p in pages]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    # ── Internal helpers ──────────────────────────────────────────────

    async def _get_same_origin(self, url: str) -> httpx.Response:
        """Follow redirects only while they remain on the configured host."""
        base_host = urlparse(self.base_url).hostname
        base_host_alt = (
            "www." + base_host if not base_host.startswith("www.") else base_host[4:]
        )

        def _same_host(h: str) -> bool:
            return h == base_host or h == base_host_alt

        current = url
        for _ in range(6):
            parsed = urlparse(current)
            if parsed.scheme not in {"http", "https"} or not _same_host(
                parsed.hostname
            ):
                raise ValueError(f"Cross-origin crawl URL rejected: {current}")
            response = await self.client.get(current)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                return response
            current = urljoin(current, location)
        raise httpx.TooManyRedirects(
            "Too many redirects", request=response.request
        )

    def _filter_same_host(
        self, pages: list[DiscoveredPage]
    ) -> list[DiscoveredPage]:
        base_host = urlparse(self.base_url).hostname
        base_host_alt = (
            "www." + base_host if not base_host.startswith("www.") else base_host[4:]
        )

        def _same_host(url_host: str) -> bool:
            return url_host == base_host or url_host == base_host_alt

        return [
            page
            for page in pages
            if urlparse(page.url).scheme in {"http", "https"}
            and _same_host(urlparse(page.url).hostname)
        ]

    async def _fetch_rendered(self, url: str) -> tuple[str, int, dict]:
        """Fetch fully-rendered HTML using Playwright (for JS-heavy sites).

        Returns (html, http_status, headers_dict).
        Gracefully falls back to httpx if Playwright is not installed.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.debug("Playwright not installed, falling back to httpx")
            resp = await self._get_same_origin(url)
            return resp.text, resp.status_code, dict(resp.headers)

        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )

        page = await self._browser.new_page()
        try:
            resp = await page.goto(
                url,
                wait_until="networkidle",
                timeout=self.settings.crawl_timeout * 1000,
            )
            status = resp.status if resp else 200
            html = await page.content()
            # Extract headers from the response
            headers = {}
            if resp:
                for key, value in resp.headers.items():
                    headers[key.lower()] = value
            return html, status, headers
        finally:
            await page.close()

    async def close(self) -> None:
        await self.client.aclose()
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
