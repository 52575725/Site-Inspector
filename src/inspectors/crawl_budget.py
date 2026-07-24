from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional, Set
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Marketing/tracking parameters — pure crawl waste
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "twclid",
    "ref", "referrer", "source",
    "replytocom",  # WordPress comment reply
    "share",  # social share variants
}

# Session / state params that create unique URLs per visitor
SESSION_PARAMS = {
    "sessionid", "phpsessid", "jsessionid", "aspsessionid", "cfid", "cftoken",
    "sid", "sess", "token",
}

# Facet / sort / filter / pagination params that expand crawl space combinatorially
FACET_PARAMS = {
    "sort", "order", "dir", "direction", "sortby", "orderby",
    "page", "p", "paged", "pg",
    "filter", "f", "filters", "facets",
    "view", "display", "layout", "mode",
    "print", "printer_friendly",
}

# Faceted navigation URL patterns
FACET_PATTERNS = [
    r"[?&](?:filter|facet|color|size|price|brand|category|tag|type|style|material|rating|sort|order)[=_]",
    r"/filter/", r"/facet/",
    r"[?&](?:min_price|max_price|price_range)",
]

# Pagination patterns (should use rel=prev/next, not just param-based)
PAGINATION_PARAMS = {"page", "p", "paged", "pg", "start", "offset", "from"}


class CrawlBudgetInspector(BaseInspector):
    """Analyze crawl budget efficiency: parameter waste, thin pages,
    orphan detection, faceted navigation, and pagination issues."""

    inspector_name = "crawl_budget"

    # Thresholds
    MIN_UNIQUE_CONTENT_WORDS = 100   # Below = thin, wasting budget if indexable
    PARAM_WASTE_RATIO_WARN = 0.15    # >15% of URLs have tracking params = issue

    def __init__(self) -> None:
        self._crawled_urls: list[str] = []
        self._page_htmls: dict[str, str] = {}
        self._page_titles: dict[str, str] = {}
        self._incoming_links: dict[str, set[str]] = {}
        self._already_reported_sitewide = False

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_crawled_urls(self, urls: list[str]) -> None:
        self._crawled_urls = urls

    def set_page_data(self, pages: list[dict]) -> None:
        """Receive page metadata for content analysis.

        Each dict: {"url": str, "title": str|None, "html_content": str|None}
        """
        for p in pages:
            url = p.get("url", "")
            self._page_htmls[url] = p.get("html_content") or ""
            self._page_titles[url] = p.get("title") or ""

    def set_incoming_links(self, links_map: dict[str, set[str]]) -> None:
        """Receive incoming internal link mapping for orphan detection."""
        self._incoming_links = links_map

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not self._crawled_urls:
            return findings

        # ── Per-page checks ──────────────────────────────────
        findings.extend(self._check_url_parameters(url))
        findings.extend(self._check_faceted_navigation(url))
        findings.extend(self._check_pagination(url, html_content))
        findings.extend(self._check_thin_page(url, html_content))
        findings.extend(self._check_orphan(url))

        # ── Site-wide checks (report once from the first/homepage URL) ──
        if not self._already_reported_sitewide:
            # Only report site-wide stats from the root URL
            path = urlparse(url).path.rstrip("/")
            if path in ("", "/", "/index.html", "/index.php"):
                findings.extend(self._site_wide_report())
                self._already_reported_sitewide = True

        return findings

    # ── Per-Page: URL Parameters ────────────────────────────────────

    def _check_url_parameters(self, url: str) -> list[RawFinding]:
        findings: list[RawFinding] = []
        parsed = urlparse(url)
        query = parsed.query

        if not query:
            return findings

        params = parse_qs(query, keep_blank_values=True)
        param_count = len(params)

        # Check for tracking parameters
        tracking = [k for k in params if k.lower() in TRACKING_PARAMS]
        session = [k for k in params
                   if k.lower() in SESSION_PARAMS or "session" in k.lower() or "sesid" in k.lower()]
        facet = [k for k in params if k.lower() in FACET_PARAMS]

        if session:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_session_params",
                description=(
                    f"URL contains session ID parameters ({', '.join(session)}). "
                    f"Search engines may index session-specific URLs, creating "
                    f"duplicate content and wasting crawl budget."
                ),
                current_value=url[:200],
                suggested_value=(
                    "Use cookies for session tracking; add "
                    "'Disallow: /*sessionid*' to robots.txt; or canonicalize "
                    "all session URLs to the clean version."
                ),
                raw_metadata={
                    "session_params": session,
                    "param_count": param_count,
                },
            ))

        if tracking:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_tracking_params",
                description=(
                    f"URL contains marketing/tracking parameters "
                    f"({', '.join(tracking)}). "
                    f"These create duplicate URLs that waste crawl budget."
                ),
                current_value=url[:200],
                suggested_value=(
                    "Canonicalize tracking URLs to the clean version; "
                    "or configure the parameter handling in Google Search Console "
                    "(Legacy tools → URL Parameters)."
                ),
                raw_metadata={
                    "tracking_params": tracking,
                },
            ))

        if facet:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_facet_params",
                description=(
                    f"URL contains facet/sort/filter parameters "
                    f"({', '.join(facet[:4])}). "
                    f"Combinatorial facet URLs can explode crawl space."
                ),
                current_value=url[:200],
                suggested_value=(
                    "Noindex faceted result pages; canonicalize to the clean "
                    "category URL; or block non-essential facet params in robots.txt."
                ),
                raw_metadata={
                    "facet_params": facet,
                },
            ))

        # General: too many parameters
        if param_count >= 5:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_many_params",
                description=(
                    f"URL has {param_count} query parameters — high risk of "
                    f"infinite crawl space (sorting, filtering combinations)."
                ),
                current_value=f"{param_count} params: {query[:120]}",
                suggested_value=(
                    "Minimize URL parameters; use 'noindex' on filtered/sorted "
                    "pages; or add parameter handling rules in robots.txt."
                ),
                raw_metadata={"param_count": param_count},
            ))

        return findings

    # ── Per-Page: Faceted Navigation ────────────────────────────────

    def _check_faceted_navigation(self, url: str) -> list[RawFinding]:
        findings: list[RawFinding] = []

        is_faceted = any(
            re.search(p, url, re.IGNORECASE) for p in FACET_PATTERNS
        )

        if is_faceted:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_faceted_url",
                description=(
                    f"URL appears to be a faceted navigation / filter page. "
                    f"Faceted URLs can generate thousands of near-duplicate "
                    f"pages that consume crawl budget."
                ),
                current_value=url[:200],
                suggested_value=(
                    "Add 'noindex, nofollow' to faceted result pages; "
                    "use canonical tags pointing to the main category page; "
                    "or block facet parameters in robots.txt."
                ),
            ))

        return findings

    # ── Per-Page: Pagination ────────────────────────────────────────

    def _check_pagination(
        self, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Detect if this is a paginated page
        is_paginated = any(p in query_params for p in PAGINATION_PARAMS)
        if not is_paginated:
            return findings

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")

        # Check for rel=prev/next
        has_prev = bool(soup.find("link", rel="prev"))
        has_next = bool(soup.find("link", rel="next"))

        if not has_prev and not has_next:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_pagination_no_rel",
                description=(
                    "Paginated page lacks rel=prev/next link tags. Search engines "
                    "may treat paginated pages as standalone (thin) pages or "
                    "consume excess crawl budget on deep pagination."
                ),
                current_value="(missing rel=prev/next)",
                suggested_value=(
                    "Add <link rel='prev'> and <link rel='next'> tags to "
                    "consolidate paginated series for search engines."
                ),
            ))

        # Check for canonical self-reference on page >1
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href", "") == url and is_paginated:
            # Page 2+ canonicalizing to itself = treating paginated pages as unique
            page_num = None
            for pname in PAGINATION_PARAMS:
                if pname in query_params:
                    try:
                        page_num = int(query_params[pname][0])
                    except (ValueError, IndexError):
                        pass

            if page_num and page_num > 1:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="crawl_budget_pagination_self_canonical",
                    description=(
                        f"Paginated page ({page_num}) has canonical pointing to "
                        f"itself instead of the 'View All' page or page 1. "
                        f"Each paginated page competes as a unique URL."
                    ),
                    current_value=canonical["href"],
                    suggested_value=(
                        "Consider a 'View All' page with all items and "
                        "canonicalize paginated URLs to it; or use "
                        "rel=prev/next + self-canonical with unique titles."
                    ),
                ))

        return findings

    # ── Per-Page: Thin Content ──────────────────────────────────────

    def _check_thin_page(
        self, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.find("body")
        if not body:
            return findings

        for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = body.get_text(separator=" ", strip=True)
        words = [w for w in text.split() if len(w) > 2]
        word_count = len(words)

        # Only flag for indexable-looking pages (not search/cart/etc)
        path = urlparse(url).path.lower()
        skip_paths = {"/cart", "/checkout", "/search", "/login", "/register",
                      "/account", "/wp-admin", "/admin"}
        is_utility_page = any(path.startswith(p) for p in skip_paths)

        if not is_utility_page and word_count < self.MIN_UNIQUE_CONTENT_WORDS:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_thin_page",
                description=(
                    f"Thin content page ({word_count} words). Low-value pages "
                    f"waste crawl budget that could be spent on important content."
                ),
                current_value=f"{word_count} words",
                suggested_value=(
                    "Either add substantial unique content (300+ words), "
                    "noindex the page, or consolidate with a related page "
                    "using a 301 redirect."
                ),
                raw_metadata={"word_count": word_count},
            ))

        return findings

    # ── Per-Page: Orphan Detection ──────────────────────────────────

    def _check_orphan(self, url: str) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not self._incoming_links:
            return findings

        normalized = url.rstrip("/")
        incoming = self._incoming_links.get(normalized, set())

        # Skip homepage and sitemap-only pages
        parsed = urlparse(url)
        if parsed.path.rstrip("/") in ("", "/"):
            return findings

        if not incoming:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="crawl_budget_orphan_page",
                description=(
                    "Orphan page: no internal links point to this URL. "
                    "Search engines may not discover or prioritize crawling "
                    "orphan pages."
                ),
                current_value="0 incoming internal links",
                suggested_value=(
                    "Add internal links from relevant pages; include in "
                    "sitemap.xml if content is valuable; or remove/noindex "
                    "if the page is not needed."
                ),
            ))

        return findings

    # ── Site-Wide Report ────────────────────────────────────────────

    def _site_wide_report(self) -> list[RawFinding]:
        """Aggregate site-wide crawl budget statistics and report once."""
        findings: list[RawFinding] = []
        total = len(self._crawled_urls)
        if total < 2:
            return findings

        # Parameter waste stats
        param_urls = 0
        tracking_urls = 0
        facet_urls = 0
        param_counter: Counter = Counter()

        for url_str in self._crawled_urls:
            parsed = urlparse(url_str)
            if parsed.query:
                param_urls += 1
                params = parse_qs(parsed.query, keep_blank_values=True)
                for p in params:
                    param_counter[p.lower()] += 1
                if any(k.lower() in TRACKING_PARAMS for k in params):
                    tracking_urls += 1

            if any(re.search(p, url_str, re.IGNORECASE) for p in FACET_PATTERNS):
                facet_urls += 1

        # Report: high param ratio
        if total >= 10:
            param_ratio = param_urls / total
            if param_ratio >= self.PARAM_WASTE_RATIO_WARN:
                findings.append(RawFinding(
                    url=self._crawled_urls[0],
                    inspector=self.inspector_name,
                    category="crawl_budget_high_param_ratio",
                    description=(
                        f"{param_urls}/{total} URLs ({param_ratio:.0%}) contain "
                        f"query parameters. High parameter proliferation wastes "
                        f"crawl budget on near-duplicate URLs."
                    ),
                    current_value=f"{param_ratio:.0%}",
                    suggested_value=(
                        "Audit URL parameters in Google Search Console; "
                        "canonicalize or noindex parameterized variants; "
                        "block tracking/sorting params in robots.txt."
                    ),
                    raw_metadata={
                        "param_urls": param_urls,
                        "total_urls": total,
                        "param_ratio": round(param_ratio, 3),
                    },
                ))

        # Report: tracking param prevalence
        if tracking_urls > 0:
            findings.append(RawFinding(
                url=self._crawled_urls[0],
                inspector=self.inspector_name,
                category="crawl_budget_tracking_param_scale",
                description=(
                    f"{tracking_urls}/{total} URLs contain tracking/UTM parameters. "
                    f"Each creates a duplicate URL that may be indexed separately."
                ),
                current_value=f"{tracking_urls} URLs with tracking params",
                suggested_value=(
                    "Canonicalize all URLs with tracking params to the clean version. "
                    "In Google Search Console → URL Parameters, set UTM params "
                    "to 'Representative URL'."
                ),
                raw_metadata={
                    "tracking_urls": tracking_urls,
                    "total_urls": total,
                },
            ))

        # Report: faceted navigation scale
        if facet_urls >= 3:
            findings.append(RawFinding(
                url=self._crawled_urls[0],
                inspector=self.inspector_name,
                category="crawl_budget_faceted_scale",
                description=(
                    f"{facet_urls}/{total} URLs appear to be faceted navigation "
                    f"pages. These can multiply into thousands of crawlable URLs."
                ),
                current_value=f"{facet_urls} faceted URLs",
                suggested_value=(
                    "Noindex faceted/filtered result pages; use canonical tags "
                    "pointing to the clean category URL; block facet params "
                    "in robots.txt for all but the most important combinations."
                ),
                raw_metadata={
                    "faceted_urls": facet_urls,
                    "total_urls": total,
                },
            ))

        # Report: top repeated parameters
        top_params = param_counter.most_common(3)
        if top_params and top_params[0][1] >= 3:
            param_list = ", ".join(f"{p} ({c} URLs)" for p, c in top_params)
            findings.append(RawFinding(
                url=self._crawled_urls[0],
                inspector=self.inspector_name,
                category="crawl_budget_top_params",
                description=(
                    f"Most repeated URL parameters across the site: {param_list}. "
                    f"These may indicate uncontrolled crawl space expansion."
                ),
                current_value=param_list,
                suggested_value=(
                    "Review each parameter's purpose; configure in GSC "
                    "parameter handling; block non-essential params in robots.txt."
                ),
                raw_metadata={
                    "top_params": dict(top_params),
                },
            ))

        return findings
