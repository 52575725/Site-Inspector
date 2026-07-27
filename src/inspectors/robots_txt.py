from __future__ import annotations

import logging

import httpx

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Paths that must be crawlable for SEO (CSS/JS needed for rendering)
CRITICAL_ASSET_PATTERNS = [
    "/css/", "/js/", "/scripts/", "/assets/", "/static/",
    "/wp-content/themes/", "/wp-content/plugins/",
]
# Common CMS/media paths that are safe to disallow
SAFE_DISALLOW_PATTERNS = [
    "/wp-admin/", "/wp-login.php", "/xmlrpc.php",
    "/api/", "/search/", "/cdn-cgi/",
]


class RobotsTxtInspector(BaseInspector):
    """Inspects robots.txt for SEO issues.

    Runs once per scan (not per page) using _already_checked flag.
    Checks: existence, sitemap reference, disallow-all, crawl-delay,
    critical-resource blocking, and sitemap-vs-disallow conflicts.
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

        from urllib.parse import urlparse

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        base_origin = f"{parsed.scheme}://{parsed.netloc}"

        # ── Fetch robots.txt ──────────────────────────────────────────

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

        # ── Parse directives ──────────────────────────────────────────

        has_sitemap = False
        has_disallow_all = False
        crawl_delay_value: float | None = None
        disallowed_paths: list[str] = []

        for line in content.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if line_lower.startswith("sitemap:"):
                has_sitemap = True

            if line_lower == "disallow: /":
                has_disallow_all = True

            if line_lower.startswith("disallow:") and not line_lower == "disallow: /":
                path = line_stripped.split(":", 1)[1].strip()
                if path:
                    disallowed_paths.append(path)

            if line_lower.startswith("crawl-delay:"):
                try:
                    crawl_delay_value = float(line_stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass

        # ── Basic checks ──────────────────────────────────────────────

        if not has_sitemap:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_no_sitemap",
                description="robots.txt does not reference the XML sitemap",
                suggested_value=f"Add: Sitemap: {base_origin}/sitemap.xml",
            ))

        if has_disallow_all:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_disallow_all",
                description="robots.txt has 'Disallow: /' — blocking all crawlers",
                suggested_value="Remove 'Disallow: /' to allow search engine indexing",
            ))

        if crawl_delay_value is not None and crawl_delay_value > 10:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_high_crawl_delay",
                description=f"Crawl-delay is {crawl_delay_value}s — may limit indexing speed",
                current_value=str(crawl_delay_value),
                suggested_value="Reduce crawl-delay to 5-10 seconds or remove entirely",
            ))

        # ── NEW: Check for blocked critical resources ─────────────────

        blocked_critical: list[str] = []
        for pattern in CRITICAL_ASSET_PATTERNS:
            for path in disallowed_paths:
                if path.startswith(pattern) or (
                    "*" in path and pattern.startswith(path.rstrip("*"))
                ):
                    blocked_critical.append(f"{path} (blocks {pattern})")

        if blocked_critical:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_blocks_critical_resources",
                description=(
                    f"robots.txt disallows paths that may block CSS/JS rendering: "
                    f"{', '.join(blocked_critical[:5])}. "
                    f"Search engines need access to CSS/JS for proper rendering."
                ),
                current_value=f"{len(blocked_critical)} critical paths blocked",
                suggested_value=(
                    "Remove disallow rules for static asset directories. "
                    "Use more specific rules if needed."
                ),
                raw_metadata={"blocked_critical": blocked_critical},
            ))

        # ── NEW: Check for disallowed paths that might conflict with sitemap ──
        # (This is a heuristic — actual sitemap cross-reference requires the
        # sitemap from SitemapInspector. This check flags broad/aggressive disallows.)

        aggressive_disallows = [
            p for p in disallowed_paths
            if p.endswith("*") and len(p.rstrip("*").rstrip("/")) > 3
            and not any(p.startswith(s) for s in SAFE_DISALLOW_PATTERNS)
        ]
        if len(aggressive_disallows) >= 5:
            findings.append(RawFinding(
                url=robots_url,
                inspector=self.inspector_name,
                category="robots_txt_aggressive_disallows",
                description=(
                    f"robots.txt has {len(aggressive_disallows)} broad disallow "
                    f"rules — some may unintentionally block indexable pages. "
                    f"Examples: {', '.join(aggressive_disallows[:3])}"
                ),
                current_value=f"{len(aggressive_disallows)} broad rules",
                suggested_value=(
                    "Audit disallow rules against your sitemap. Use exact "
                    "paths instead of wildcards where possible."
                ),
                raw_metadata={"aggressive_disallows": aggressive_disallows[:10]},
            ))

        return findings
