from __future__ import annotations

import logging
from typing import Optional, Set

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# === Security Headers ===

SECURITY_HEADERS: dict[str, str] = {
    "strict-transport-security": (
        "HSTS (HTTP Strict-Transport-Security) — enforces HTTPS; "
        "missing means the site is vulnerable to SSL-stripping attacks. "
        "Recommended: 'max-age=31536000; includeSubDomains; preload'"
    ),
    "content-security-policy": (
        "CSP (Content-Security-Policy) — mitigates XSS, data injection, and clickjacking. "
        "A strong CSP is one of the best defenses against cross-site scripting."
    ),
    "x-frame-options": (
        "X-Frame-Options — prevents clickjacking by controlling iframe embedding. "
        "Use 'DENY' or 'SAMEORIGIN'. Nested in CSP frame-ancestors when CSP is present."
    ),
    "x-content-type-options": (
        "X-Content-Type-Options — stops MIME-type sniffing. "
        "Always set to 'nosniff'."
    ),
    "referrer-policy": (
        "Referrer-Policy — controls how much referrer information is sent with requests. "
        "Recommended: 'strict-origin-when-cross-origin'."
    ),
    "permissions-policy": (
        "Permissions-Policy — restricts browser feature APIs "
        "(camera, microphone, geolocation, etc.). Reduces attack surface."
    ),
}

# Headers whose absence is informational (lower severity)
INFO_SECURITY_HEADERS: dict[str, str] = {
    "cross-origin-opener-policy": (
        "Cross-Origin-Opener-Policy (COOP) — isolates browsing context. "
        "Recommended for pages handling sensitive data."
    ),
    "cross-origin-resource-policy": (
        "Cross-Origin-Resource-Policy (CORP) — controls which sites can load this resource."
    ),
    "cross-origin-embedder-policy": (
        "Cross-Origin-Embedder-Policy (COEP) — required for SharedArrayBuffer and "
        "other high-security features."
    ),
}

# === Caching Headers ===

CACHE_HEADERS: dict[str, str] = {
    "cache-control": (
        "Cache-Control — the primary directive for browser and CDN caching. "
        "Missing means browsers use heuristic caching, which leads to stale content."
    ),
}

# === Information Disclosure Headers ===

LEAKY_HEADERS = {
    "server": "Server header reveals web-server name/version — useful fingerprint for attackers.",
    "x-powered-by": "X-Powered-By reveals the application framework/version (e.g. Express, PHP).",
    "x-aspnet-version": "X-AspNet-Version leaks ASP.NET framework version.",
    "x-generator": "X-Generator reveals the CMS or static-site generator in use.",
    "x-drupal-cache": "X-Drupal-Cache reveals Drupal caching internals.",
    "x-drupal-dynamic-cache": "X-Drupal-Dynamic-Cache reveals Drupal dynamic page cache state.",
}


class HeadersInspector(BaseInspector):
    """Inspect HTTP response headers for security, caching, compression, and privacy."""

    inspector_name = "headers"

    # Server-level categories: reported ONCE per scan, not per page
    SERVER_LEVEL_CATEGORIES: set[str] = {
        "missing_strict_transport_security", "hsts_max_age_too_short",
        "missing_content_security_policy", "missing_x_frame_options",
        "missing_x_frame_options_with_csp", "missing_x_content_type_options",
        "missing_referrer_policy", "missing_permissions_policy",
        "missing_cross_origin_opener_policy", "missing_cross_origin_resource_policy",
        "missing_cross_origin_embedder_policy", "info_leak_server",
        "info_leak_x_powered_by", "info_leak_x_aspnet_version",
        "info_leak_x_generator", "info_leak_x_drupal_cache",
        "info_leak_x_drupal_dynamic_cache", "missing_compression",
        "headers_no_response_headers",
    }

    def __init__(self) -> None:
        self._checked_urls: Set[str] = set()
        self._reported_server_categories: set[str] = set()

    async def setup(self) -> None:
        self._reported_server_categories.clear()
        self._checked_urls.clear()

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if headers is None or not headers:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="headers_no_response_headers",
                description="No HTTP response headers available for inspection "
                            "(the page may have failed to load).",
            ))
            return findings

        # Normalise to lowercase for httpx compatibility
        lower_headers = {k.lower(): v for k, v in headers.items()}

        findings.extend(self._check_security_headers(url, lower_headers))
        findings.extend(self._check_cache_headers(url, lower_headers))
        findings.extend(self._check_compression(url, lower_headers))
        findings.extend(self._check_info_leak(url, lower_headers))
        findings.extend(self._check_x_robots_tag(url, lower_headers, html_content))
        findings.extend(self._check_content_type_charset(url, lower_headers))

        return findings

    # ── Security Headers ────────────────────────────────────────────

    def _check_security_headers(
        self, url: str, headers: dict[str, str],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        present = set()

        # Mandatory checks
        for hdr_name, description in SECURITY_HEADERS.items():
            if hdr_name in headers:
                present.add(hdr_name)
            else:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"missing_{hdr_name.replace('-', '_')}",
                    description=f"Missing HTTP header '{hdr_name}': {description}",
                    current_value="(absent)",
                ))

        # Complementary checks — only report if CSP is also missing
        # (CSP frame-ancestors replaces X-Frame-Options)
        if "x-frame-options" not in present and "content-security-policy" in present:
            # Remove the X-Frame-Options finding — it was already added above but
            # we now know CSP exists; however, we only *weaken* the message.
            # Re-find and replace:
            for f in findings:
                if f.category == "missing_x_frame_options":
                    f.description += (
                        " (Note: CSP is present — frame-ancestors directive can "
                        "replace this header, but X-Frame-Options is still "
                        "recommended for older browser compatibility.)"
                    )
                    f.category = "missing_x_frame_options_with_csp"

        # Optional security headers (lower severity)
        for hdr_name, description in INFO_SECURITY_HEADERS.items():
            if hdr_name not in headers:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"missing_{hdr_name.replace('-', '_')}",
                    description=f"Missing HTTP header '{hdr_name}': {description}",
                    current_value="(absent)",
                ))

        # HSTS specific: warn if max-age is too short
        hsts = headers.get("strict-transport-security", "")
        if hsts:
            import re
            m = re.search(r"max-age=(\d+)", hsts, re.IGNORECASE)
            if m:
                max_age = int(m.group(1))
                if max_age < 31536000:
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="hsts_max_age_too_short",
                        description=f"HSTS max-age is {max_age}s ({max_age // 86400}d) — "
                                    f"Google recommends at least 1 year (31536000s) for "
                                    f"HSTS preload eligibility.",
                        current_value=f"max-age={max_age}",
                        suggested_value="max-age=31536000",
                    ))

        return findings

    # ── Caching Headers ─────────────────────────────────────────────

    def _check_cache_headers(
        self, url: str, headers: dict[str, str],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        cache_control = headers.get("cache-control", "")
        has_cache_control = bool(cache_control)

        if not has_cache_control:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_cache_control",
                description=f"Missing 'cache-control' header: {CACHE_HEADERS['cache-control']}",
                current_value="(absent)",
            ))
        else:
            # Validate Cache-Control directives
            directives = {d.strip() for d in cache_control.lower().split(",")}
            # Check for busted patterns
            if "no-cache" in directives and "max-age" in directives:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="cache_control_conflict",
                    description="Cache-Control has both 'no-cache' and 'max-age' — "
                                "these directives conflict; browsers may behave unpredictably.",
                    current_value=cache_control,
                    suggested_value="Use 'no-cache' alone or set a specific max-age.",
                ))
            if "no-store" in directives and "max-age" in directives:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="cache_control_conflict",
                    description="Cache-Control has both 'no-store' and 'max-age' — "
                                "'no-store' overrides 'max-age', making the latter irrelevant.",
                    current_value=cache_control,
                    suggested_value="Remove 'max-age' or replace 'no-store' with 'no-cache'.",
                ))

        # ETag
        if "etag" not in headers:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_etag",
                description="Missing 'ETag' header — browsers cannot perform conditional "
                            "requests (If-None-Match), leading to unnecessary re-downloads.",
                current_value="(absent)",
            ))

        # Vary
        if "vary" not in headers:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_vary",
                description="Missing 'Vary' header — intermediate caches may serve "
                            "the wrong variant (e.g. uncompressed to a gzip-capable client). "
                            "At minimum, set 'Vary: Accept-Encoding'.",
                current_value="(absent)",
            ))
        else:
            vary = headers["vary"].lower()
            if "accept-encoding" not in vary:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="vary_missing_accept_encoding",
                    description="'Vary' header does not include 'Accept-Encoding' — "
                                "CDNs may serve compressed content to clients that don't "
                                "support it (or vice versa).",
                    current_value=headers["vary"],
                    suggested_value=f"{headers['vary']}, Accept-Encoding",
                ))

        return findings

    # ── Compression ─────────────────────────────────────────────────

    def _check_compression(
        self, url: str, headers: dict[str, str],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        content_encoding = headers.get("content-encoding", "").lower()

        # Only check text/html pages (images etc. are pre-compressed)
        content_type = headers.get("content-type", "").lower()
        is_html = "text/html" in content_type or not content_type

        if not content_encoding and is_html:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_compression",
                description="Response is not compressed (no 'Content-Encoding' header). "
                            "Enable gzip or brotli compression to reduce transfer size.",
                current_value="(absent)",
                suggested_value="gzip or br",
            ))

        # Check transfer size hint (if content-length is available)
        content_length_str = headers.get("content-length")
        if content_length_str and is_html and not content_encoding:
            try:
                cl = int(content_length_str)
                if cl > 50_000:
                    for f in findings:
                        if f.category == "missing_compression":
                            f.description += (
                                f" Uncompressed page size is {cl // 1024} KiB — "
                                f"gzip typically reduces HTML by 70–80%."
                            )
            except (ValueError, TypeError):
                pass

        return findings

    # ── Information Disclosure ──────────────────────────────────────

    def _check_info_leak(
        self, url: str, headers: dict[str, str],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        for hdr_name, description in LEAKY_HEADERS.items():
            if hdr_name in headers:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"info_leak_{hdr_name.replace('-', '_')}",
                    description=f"Information disclosure: {description} "
                                f"Current value: '{headers[hdr_name][:100]}'",
                    current_value=headers[hdr_name][:100],
                    suggested_value="Remove or suppress this header via server config.",
                ))

        return findings

    # ── X-Robots-Tag ────────────────────────────────────────────────

    def _check_x_robots_tag(
        self, url: str, headers: dict[str, str], html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        x_robots = headers.get("x-robots-tag", "")
        if not x_robots:
            return findings

        # Check if x-robots-tag is blocking indexing
        xr_lower = x_robots.lower()
        blocking_directives = ["noindex", "nofollow", "none"]

        for directive in blocking_directives:
            if directive in xr_lower:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="x_robots_tag_blocks_indexing",
                    description=f"X-Robots-Tag header contains '{directive}' — "
                                f"this page will NOT be indexed by search engines. "
                                f"Verify this is intentional.",
                    current_value=x_robots,
                ))
                break

        # Detect conflicts with meta robots
        if html_content:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            meta_robots = soup.find("meta", attrs={"name": "robots"})
            if meta_robots and meta_robots.get("content"):
                meta_content = meta_robots["content"].lower()
                # Conflict: header says noindex but meta says index
                if "noindex" in xr_lower and "noindex" not in meta_content:
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="x_robots_tag_meta_conflict",
                        description="X-Robots-Tag header and meta robots tag may conflict. "
                                    f"Header: '{x_robots}', Meta: '{meta_robots['content']}'",
                        current_value=f"header={x_robots}; meta={meta_robots['content']}",
                        suggested_value="Align both directives.",
                    ))

        return findings

    # ── Content-Type charset ────────────────────────────────────────

    def _check_content_type_charset(
        self, url: str, headers: dict[str, str],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        content_type = headers.get("content-type", "")

        if not content_type:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_content_type",
                description="Response has no 'Content-Type' header — browsers must "
                            "sniff the content type, which is a security risk.",
                current_value="(absent)",
            ))
            return findings

        ct_lower = content_type.lower()
        if "text/html" in ct_lower and "charset" not in ct_lower:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="content_type_missing_charset",
                description=f"Content-Type is '{content_type}' but charset is not declared. "
                            "Always specify charset (e.g. 'text/html; charset=utf-8') to "
                            "prevent encoding-related XSS.",
                current_value=content_type,
                suggested_value=f"{content_type.split(';')[0].strip()}; charset=utf-8",
            ))

        return findings
