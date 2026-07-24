from __future__ import annotations

import asyncio
import hashlib
import logging
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


class BrokenLinksInspector(BaseInspector):
    """Inspect all links on a page for 404/500 errors, redirect chains, and mixed content."""

    inspector_name = "broken_links"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=15, follow_redirects=False)

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        await self.client.aclose()

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        base_url_domain = urlparse(url).netloc

        # Extract all links
        links_to_check: list[tuple[str, str]] = []

        for tag in soup.find_all("a", href=True):
            links_to_check.append(("a_href", tag["href"]))
        for tag in soup.find_all("img", src=True):
            links_to_check.append(("img_src", tag["src"]))
        for tag in soup.find_all("link", href=True):
            links_to_check.append(("link_href", tag["href"]))
        for tag in soup.find_all("script", src=True):
            links_to_check.append(("script_src", tag["src"]))

        # De-duplicate by absolute URL
        seen_urls: dict[str, tuple[str, str]] = {}
        for tag_type, href in links_to_check:
            full_url = urljoin(url, href)
            url_hash = hashlib.md5(full_url.encode()).hexdigest()
            if url_hash not in seen_urls:
                seen_urls[url_hash] = (tag_type, full_url)

        # Check mixed content
        if url.startswith("https://"):
            for _, full_url in seen_urls.values():
                if full_url.startswith("http://"):
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="mixed_content",
                        description=f"HTTPS page loads HTTP resource: {full_url}",
                        element=full_url,
                    ))

        # Check internal links (same domain) for 404/500
        internal_links = [
            (tag_type, full_url)
            for _, (tag_type, full_url) in seen_urls.items()
            if urlparse(full_url).netloc == base_url_domain
        ]

        # Limit concurrent checks to 5
        semaphore = asyncio.Semaphore(5)

        async def check_link(tag_type: str, link_url: str) -> RawFinding | None:
            async with semaphore:
                try:
                    resp = await self.client.get(link_url)
                    status = resp.status_code

                    # Check redirect chain
                    if resp.history and len(resp.history) > 3:
                        chain = " → ".join(str(h.status_code) for h in resp.history)
                        return RawFinding(
                            url=url, inspector=self.inspector_name,
                            category="redirect_chain",
                            description=f"Long redirect chain ({len(resp.history)} hops): {chain} → {status}",
                            element=link_url,
                            current_value=chain,
                        )

                    if status >= 400:
                        return RawFinding(
                            url=url, inspector=self.inspector_name,
                            category=f"http_{status}",
                            description=f"{tag_type} returns {status}: {link_url}",
                            element=link_url,
                            current_value=str(status),
                        )
                except httpx.TimeoutException:
                    return RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="link_timeout",
                        description=f"Link timed out: {link_url}",
                        element=link_url,
                    )
                except Exception as e:
                    logger.debug(f"Error checking link {link_url}: {e}")
                return None

        tasks = [check_link(tt, lu) for tt, lu in internal_links]
        results = await asyncio.gather(*tasks)
        findings.extend(r for r in results if r is not None)

        return findings
