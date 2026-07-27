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
    """Inspect all links on a page for 404/500 errors, redirect chains,
    mixed content, and broken external links.

    Internal links are checked with full GET requests.  External links
    are checked with lightweight HEAD requests (with GET fallback) to
    verify they're reachable without downloading full pages.
    """

    inspector_name = "broken_links"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=15, follow_redirects=False)
        self._semaphore = None

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

        # External links (different domain) — checked with HEAD for efficiency
        external_links = [
            (tag_type, full_url)
            for _, (tag_type, full_url) in seen_urls.items()
            if urlparse(full_url).netloc != base_url_domain
        ]

        # Limit concurrent checks
        semaphore = asyncio.Semaphore(5)

        async def check_internal_link(tag_type: str, link_url: str) -> RawFinding | None:
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

        async def check_external_link(tag_type: str, link_url: str) -> RawFinding | None:
            """Check external link with HEAD request, fall back to GET."""
            async with semaphore:
                try:
                    # Try HEAD first
                    try:
                        resp = await self.client.head(link_url)
                    except Exception:
                        # Fall back to GET with stream to avoid downloading body
                        resp = await self.client.get(link_url)

                    status = resp.status_code

                    if status >= 500:
                        return RawFinding(
                            url=url, inspector=self.inspector_name,
                            category="external_link_broken",
                            description=(
                                f"External {tag_type} returns {status}: {link_url}. "
                                f"Consider removing or updating this link."
                            ),
                            element=link_url,
                            current_value=str(status),
                        )
                    if status >= 400 and status < 500:
                        return RawFinding(
                            url=url, inspector=self.inspector_name,
                            category="external_link_warning",
                            description=(
                                f"External {tag_type} returns {status}: {link_url}. "
                                f"Verify the link is still valid."
                            ),
                            element=link_url,
                            current_value=str(status),
                        )
                except httpx.TimeoutException:
                    return RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="external_link_timeout",
                        description=f"External link timed out: {link_url}",
                        element=link_url,
                    )
                except Exception:
                    pass  # External links can fail for many reasons
                return None

        tasks = [check_internal_link(tt, lu) for tt, lu in internal_links]
        # Only check up to 20 external links per page to stay efficient
        ext_sample = external_links[:20]
        tasks += [check_external_link(tt, lu) for tt, lu in ext_sample]

        results = await asyncio.gather(*tasks)
        findings.extend(r for r in results if r is not None)

        return findings
