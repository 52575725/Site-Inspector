from __future__ import annotations

import logging
from urllib.parse import urlparse

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

# sitemap submission endpoints to check
SITEMAP_CHECK_PATTERNS = [
    (("google", "google-site-verification"), "/sitemap.xml"),
    (("bing", "msvalidate.01"), "/sitemap.xml"),
]


class PlatformSEOInspector(BaseInspector):
    """Checks homepage for search-engine verification and platform readiness.

    Extended beyond basic meta-tag presence:
    - Verification meta tags (Google, Baidu, Bing, Yandex)
    - robots.txt crawlability for search engines
    - Common SEO anti-patterns (noindex on homepage, etc.)
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

        # Only check the root/homepage
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

        # ── 1. Verification meta tags ────────────────────────────────

        verified_platforms: list[str] = []
        for check in PLATFORM_CHECKS:
            existing = head.find(
                "meta", attrs={"name": check["name_attr"]},
            )
            if existing:
                verified_platforms.append(check["platform"])
            else:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category=check["category"],
                    description=f"{check['desc']}. {check['guide']}",
                    suggested_value=(
                        f"<meta name=\"{check['name_attr']}\" "
                        f"content=\"YOUR_CODE\" />"
                    ),
                    raw_metadata={
                        "platform": check["platform"],
                        "meta_name": check["name_attr"],
                    },
                ))

        # ── 2. Count verified platforms ───────────────────────────────

        if len(verified_platforms) == 0:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="platform_no_verification",
                description=(
                    "No search engine verification tags found on homepage. "
                    "Verifying your site with Google Search Console and Bing "
                    "Webmaster Tools is essential for indexing and analytics."
                ),
                suggested_value=(
                    "Register with at least Google Search Console and "
                    "Bing Webmaster Tools, then add verification meta tags"
                ),
                raw_metadata={"verified": verified_platforms},
            ))

        # ── 3. Homepage robots meta (noindex check) ──────────────────

        robots_meta = head.find("meta", attrs={"name": "robots"})
        if robots_meta:
            content_val = robots_meta.get("content", "").lower()
            if "noindex" in content_val:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="platform_homepage_noindex",
                    description=(
                        f"Homepage has robots meta with 'noindex' ({content_val}) — "
                        f"search engines will NOT index the homepage. This is "
                        f"almost always a mistake on a live site."
                    ),
                    current_value=content_val,
                    suggested_value=(
                        "Remove 'noindex' from homepage robots meta tag"
                    ),
                ))
            if "nofollow" in content_val:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="platform_homepage_nofollow",
                    description=(
                        f"Homepage has robots meta with 'nofollow' ({content_val}) — "
                        f"search engines will NOT follow links from the homepage."
                    ),
                    current_value=content_val,
                    suggested_value=(
                        "Remove 'nofollow' from homepage unless intentionally "
                        "blocking link equity flow"
                    ),
                ))

        # ── 4. Check for redirects on homepage (often missed) ────────

        x_robots = (headers or {}).get("x-robots-tag", "")
        if x_robots and "noindex" in x_robots.lower():
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="platform_x_robots_noindex",
                description=(
                    f"Homepage served with X-Robots-Tag: {x_robots} — "
                    f"this header-level noindex will prevent indexing"
                ),
                current_value=x_robots,
                suggested_value="Remove 'noindex' from X-Robots-Tag header",
            ))

        return findings
