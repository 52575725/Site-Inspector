from __future__ import annotations

import logging

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Known verification meta tag patterns for major platforms
PLATFORM_CHECKS = [
    {
        "platform": "Google Search Console",
        "category": "platform_missing_google_verify",
        "name_attr": "google-site-verification",
        "desc": "Missing Google Search Console verification tag",
        "guide": "Get your verification code from https://search.google.com/search-console",
    },
    {
        "platform": "Baidu Ziyuan",
        "category": "platform_missing_baidu_verify",
        "name_attr": "baidu-site-verification",
        "desc": "Missing Baidu site verification tag",
        "guide": "Get your verification code from https://ziyuan.baidu.com",
    },
    {
        "platform": "Bing Webmaster",
        "category": "platform_missing_bing_verify",
        "name_attr": "msvalidate.01",
        "desc": "Missing Bing Webmaster Tools verification tag",
        "guide": "Get your verification code from https://www.bing.com/webmasters",
    },
    {
        "platform": "Yandex Webmaster",
        "category": "platform_missing_yandex_verify",
        "name_attr": "yandex-verification",
        "desc": "Missing Yandex Webmaster verification tag",
        "guide": "Get your verification code from https://webmaster.yandex.com",
    },
]


class PlatformSEOInspector(BaseInspector):
    """Checks homepage for search-engine verification meta tags.

    Only the homepage is checked — verification tags belong in <head>.
    """

    inspector_name = "platform_seo"

    def __init__(self):
        super().__init__()
        self._homepage_checked = False

    async def setup(self) -> None:
        self._homepage_checked = False

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Only check the root/homepage (verification tags go in <head> there)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path and path not in ("/index.html", "/index.htm", "/index.php"):
            return findings

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        head = soup.find("head")
        if not head:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="platform_missing_head",
                description="Page has no <head> — cannot add verification meta tags",
                suggested_value="Add a proper <head> section to the homepage",
            ))
            return findings

        for check in PLATFORM_CHECKS:
            existing = head.find(
                "meta", attrs={"name": check["name_attr"]},
            )
            if not existing:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category=check["category"],
                    description=f"{check['desc']}. {check['guide']}",
                    suggested_value=f"<meta name=\"{check['name_attr']}\" content=\"YOUR_CODE\" />",
                    raw_metadata={
                        "platform": check["platform"],
                        "meta_name": check["name_attr"],
                    },
                ))

        return findings
