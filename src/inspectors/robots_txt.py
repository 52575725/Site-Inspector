from __future__ import annotations

import logging
from urllib.parse import urljoin

import httpx

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


class RobotsTxtInspector(BaseInspector):
    """Inspects robots.txt for SEO issues.

    Runs once per scan (not per page) using _already_checked flag.
    """

    inspector_name = "robots_txt"

    def __init__(self):
        super().__init__()
        self._already_checked = False

    async def setup(self) -> None:
        self._already_checked = False

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if self._already_checked:
            return findings
        self._already_checked = True

        # Build robots.txt URL from the first page URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(robots_url, follow_redirects=True)
        except Exception as e:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_missing",
                description=f"Could not fetch robots.txt: {str(e)[:200]}",
                suggested_value="Create a robots.txt file with Sitemap reference",
            ))
            return findings

        if resp.status_code == 404:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_missing",
                description="robots.txt returns 404 — search engines may crawl without guidance",
                suggested_value="Create a robots.txt file",
            ))
            return findings

        if resp.status_code != 200:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_missing",
                description=f"robots.txt returned HTTP {resp.status_code}",
                suggested_value="Ensure robots.txt returns 200 OK",
            ))
            return findings

        content = resp.text.strip()
        if not content:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_empty",
                description="robots.txt exists but is empty",
                suggested_value="Add User-agent rules and Sitemap reference",
            ))
            return findings

        # Parse directives
        has_sitemap = False
        has_disallow_all = False
        has_crawl_delay = False
        crawl_delay_value = 0

        for line in content.split("\n"):
            line = line.strip()

            if line.lower().startswith("sitemap:"):
                has_sitemap = True

            if line.lower() == "disallow: /":
                has_disallow_all = True

            if line.lower().startswith("crawl-delay:"):
                has_crawl_delay = True
                try:
                    crawl_delay_value = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        # Check: missing sitemap reference
        if not has_sitemap:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_no_sitemap",
                description="robots.txt does not reference the XML sitemap",
                suggested_value=f"Add: Sitemap: {parsed.scheme}://{parsed.netloc}/sitemap.xml",
            ))

        # Check: disallow all
        if has_disallow_all:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_disallow_all",
                description="robots.txt has 'Disallow: /' — blocking all crawlers",
                suggested_value="Remove 'Disallow: /' to allow search engine indexing",
            ))

        # Check: high crawl delay
        if has_crawl_delay and crawl_delay_value > 10:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_high_crawl_delay",
                description=f"Crawl-delay is {crawl_delay_value}s — may limit indexing speed",
                current_value=str(crawl_delay_value),
                suggested_value="Reduce crawl-delay to 5-10 seconds or remove entirely",
            ))

        return findings
