from __future__ import annotations

import difflib
import re

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class HeadersFixer(BaseFixer):
    """Generate server configuration and meta fallbacks for HTTP response headers.

    Covers: HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy,
    Permissions-Policy, Cache-Control, ETag, Vary, Content-Encoding (gzip/brotli),
    info leak suppression, Content-Type charset, and X-Robots-Tag.
    """

    fixer_name = "headers_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        # Security
        "missing_strict_transport_security",
        "missing_content_security_policy",
        "missing_x_frame_options",
        "missing_x_frame_options_with_csp",
        "missing_x_content_type_options",
        "missing_referrer_policy",
        "missing_permissions_policy",
        "missing_cross_origin_opener_policy",
        "missing_cross_origin_resource_policy",
        "missing_cross_origin_embedder_policy",
        "hsts_max_age_too_short",
        # Caching
        "missing_cache_control",
        "cache_control_conflict",
        "missing_etag",
        "missing_vary",
        "vary_missing_accept_encoding",
        # Compression
        "missing_compression",
        # Info leak
        "info_leak_server",
        "info_leak_x_powered_by",
        "info_leak_x_aspnet_version",
        "info_leak_x_generator",
        "info_leak_x_drupal_cache",
        "info_leak_x_drupal_dynamic_cache",
        # Other
        "missing_content_type",
        "content_type_missing_charset",
        "x_robots_tag_blocks_indexing",
        "x_robots_tag_meta_conflict",
        "headers_no_response_headers",
    ]

    def __init__(self):
        pass

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        category = issue.get("category", "")
        url = issue.get("url", "")

        config_snippet = self._build_config(category, page_content, url)
        if not config_snippet:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path="server-config.conf",
                before_content="", after_content="",
                error_message=f"No config template for '{category}'",
            )

        # For HTML-embeddable headers (CSP, charset), modify the HTML directly
        if category in ("missing_content_security_policy", "content_type_missing_charset"):
            after_html, file_path = self._patch_html(page_content, category, config_snippet, url)
        else:
            after_html = page_content
            file_path = self._config_file_path(category, url)

        diff_text = self._make_diff(page_content, after_html, config_snippet)

        return FixResult(
            success=True, issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name, fix_type=self.fix_type,
            file_path=file_path,
            before_content=page_content,
            after_content=after_html,
            diff=diff_text,
        )

    # ── Config Builder ──────────────────────────────────────────────

    def _build_config(self, category: str, page_content: str, url: str) -> str:
        """Build the recommended server config snippet for a given category."""
        soup = None

        # Security headers
        if category == "missing_strict_transport_security":
            return (
                "# HSTS — force HTTPS for all subdomains (preload-ready)\n"
                'add_header Strict-Transport-Security "max-age=31536000; '
                'includeSubDomains; preload" always;'
            )
        if category == "hsts_max_age_too_short":
            return (
                "# HSTS — increase max-age to 1 year for preload eligibility\n"
                'add_header Strict-Transport-Security "max-age=31536000; '
                'includeSubDomains; preload" always;'
            )
        if category == "missing_content_security_policy":
            return self._build_csp(page_content)
        if category in ("missing_x_frame_options", "missing_x_frame_options_with_csp"):
            return (
                "# X-Frame-Options — prevent clickjacking\n"
                'add_header X-Frame-Options "DENY" always;'
            )
        if category == "missing_x_content_type_options":
            return (
                "# X-Content-Type-Options — stop MIME sniffing\n"
                'add_header X-Content-Type-Options "nosniff" always;'
            )
        if category == "missing_referrer_policy":
            return (
                "# Referrer-Policy — control referrer information\n"
                'add_header Referrer-Policy "strict-origin-when-cross-origin" always;'
            )
        if category == "missing_permissions_policy":
            return (
                "# Permissions-Policy — restrict browser feature APIs\n"
                'add_header Permissions-Policy "camera=(), microphone=(), '
                'geolocation=(), interest-cohort=()" always;'
            )
        if category == "missing_cross_origin_opener_policy":
            return (
                "# Cross-Origin-Opener-Policy — isolate browsing context\n"
                'add_header Cross-Origin-Opener-Policy "same-origin" always;'
            )
        if category == "missing_cross_origin_resource_policy":
            return (
                "# Cross-Origin-Resource-Policy — control resource loading\n"
                'add_header Cross-Origin-Resource-Policy "same-origin" always;'
            )
        if category == "missing_cross_origin_embedder_policy":
            return (
                "# Cross-Origin-Embedder-Policy — enable cross-origin isolation\n"
                'add_header Cross-Origin-Embedder-Policy "require-corp" always;'
            )

        # Caching
        if category == "missing_cache_control":
            return (
                "# Cache-Control — set appropriate browser/CDN caching\n"
                'add_header Cache-Control "public, max-age=3600" always;'
            )
        if category == "cache_control_conflict":
            return (
                "# Cache-Control — fixed: removed conflicting directives\n"
                'add_header Cache-Control "public, max-age=3600" always;'
            )
        if category == "missing_etag":
            return (
                "# ETag — enable conditional requests (nginx does this by default)\n"
                "# No action needed if using nginx; for Apache, ensure:\n"
                "#   FileETag MTime Size"
            )
        if category == "missing_vary":
            return (
                "# Vary — signal content negotiation to intermediate caches\n"
                'add_header Vary "Accept-Encoding" always;'
            )
        if category == "vary_missing_accept_encoding":
            return (
                "# Vary — add Accept-Encoding to existing Vary header\n"
                'add_header Vary "Accept-Encoding" always;'
            )

        # Compression
        if category == "missing_compression":
            return (
                "# Enable gzip compression (nginx)\n"
                "gzip on;\n"
                "gzip_types text/html text/css application/javascript "
                "application/json image/svg+xml;\n"
                "gzip_min_length 1000;\n"
                "gzip_comp_level 6;\n\n"
                "# Or for Apache (.htaccess):\n"
                "#   AddOutputFilterByType DEFLATE text/html text/css "
                "application/javascript"
            )

        # Info leak
        if category == "info_leak_server":
            return (
                "# Suppress Server header (nginx)\n"
                "server_tokens off;\n"
                'more_clear_headers Server;  # requires ngx_headers_more module\n\n'
                "# Or set a minimal value:\n"
                '# add_header Server "" always;'
            )
        if category == "info_leak_x_powered_by":
            return (
                "# Suppress X-Powered-By header\n"
                "# For Express: app.disable('x-powered-by');\n"
                "# For PHP: expose_php = Off (in php.ini)\n"
                "# For nginx proxy: proxy_hide_header X-Powered-By;"
            )
        if category == "info_leak_x_aspnet_version":
            return (
                "# Suppress ASP.NET version header (web.config)\n"
                "<system.web>\n"
                '  <httpRuntime enableVersionHeader="false" />\n'
                "</system.web>"
            )
        if category == "info_leak_x_generator":
            return (
                "# Suppress X-Generator header\n"
                "# WordPress: add to functions.php:\n"
                "#   remove_action('wp_head', 'wp_generator');\n"
                "# Drupal: uncheck 'Display generator' in admin settings"
            )
        if category in ("info_leak_x_drupal_cache", "info_leak_x_drupal_dynamic_cache"):
            return (
                "# Suppress Drupal cache headers\n"
                "# In settings.php:\n"
                "#   $conf['omit_vary_cookie'] = TRUE;"
            )

        # Content-Type
        if category == "missing_content_type":
            return (
                "# Content-Type with charset\n"
                'add_header Content-Type "text/html; charset=utf-8" always;'
            )
        if category == "content_type_missing_charset":
            return (
                "# Declare charset in Content-Type header (nginx)\n"
                'charset utf-8;\n'
                'charset_types text/html text/css application/javascript;'
            )

        # X-Robots-Tag
        if category == "x_robots_tag_blocks_indexing":
            return (
                "# WARNING: X-Robots-Tag is blocking search indexing!\n"
                "# If intentional, leave as-is.\n"
                "# If NOT intentional, remove this from your server config:\n"
                "#   add_header X-Robots-Tag \"noindex, nofollow\";\n"
                "# Or change to:\n"
                '#   add_header X-Robots-Tag "index, follow";'
            )
        if category == "x_robots_tag_meta_conflict":
            return (
                "# CONFLICT: X-Robots-Tag header conflicts with meta robots tag.\n"
                "# Choose ONE approach and remove the other:\n"
                "#   Header: X-Robots-Tag value\n"
                "#   Meta: <meta name=\"robots\" content=\"...\">\n"
                "# Recommendation: use meta tags for per-page control, "
                "headers for site-wide rules."
            )
        if category == "headers_no_response_headers":
            return (
                "# Server returned no response headers — the page may be down.\n"
                "# Check server status, CDN configuration, and DNS resolution."
            )

        return ""

    def _build_csp(self, page_content: str) -> str:
        """Build a sensible default CSP based on page content analysis."""
        domains: set[str] = set()

        # Scan HTML for external resource domains
        for match in re.finditer(
            r'(?:src|href)=["\'](?:https?:)?//([^/"\'\s]+)',
            page_content, re.IGNORECASE,
        ):
            host = match.group(1)
            if host not in ("www.w3.org", "schema.org"):
                domains.add(host)

        csp_directives = [
            "default-src 'self'",
        ]
        if domains:
            csp_directives.append(f"script-src 'self' {' '.join(sorted(domains)[:5])}")
            csp_directives.append(f"style-src 'self' 'unsafe-inline' {' '.join(sorted(domains)[:5])}")
            csp_directives.append(f"img-src 'self' data: {' '.join(sorted(domains)[:5])}")
            csp_directives.append(f"connect-src 'self' {' '.join(sorted(domains)[:3])}")
            csp_directives.append(f"font-src 'self' {' '.join(sorted(domains)[:3])}")
        else:
            csp_directives += [
                "script-src 'self'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data:",
                "connect-src 'self'",
                "font-src 'self'",
            ]

        csp_directives += [
            "frame-ancestors 'self'",
            "base-uri 'self'",
            "form-action 'self'",
        ]

        csp_value = "; ".join(csp_directives) + ";"

        return (
            f"# Content-Security-Policy — auto-generated from page analysis\n"
            f'add_header Content-Security-Policy "{csp_value}" always;'
        )

    # ── HTML Patching (for embeddable headers) ─────────────────────

    def _patch_html(
        self, page_content: str, category: str, config_snippet: str, url: str,
    ) -> tuple[str, str]:
        """For headers that can be added as <meta> tags, patch the HTML."""
        soup = BeautifulSoup(page_content, "html.parser")
        head = soup.find("head")
        file_path = self._config_file_path(category, url)

        if category == "missing_content_security_policy":
            # Add CSP as meta tag (weaker than header but better than nothing)
            if head:
                csp_value = ""
                for line in config_snippet.split("\n"):
                    match = re.search(r'Content-Security-Policy\s+"(.+?)"', line)
                    if match:
                        csp_value = match.group(1)
                        break
                if csp_value:
                    meta = soup.new_tag(
                        "meta",
                        attrs={
                            "http-equiv": "Content-Security-Policy",
                            "content": csp_value,
                        },
                    )
                    head.insert(0, meta)
                    file_path = self._url_to_filename(url)
                    return str(soup), file_path

        if category == "content_type_missing_charset":
            if head:
                meta = soup.find("meta", attrs={"http-equiv": "Content-Type"})
                if not meta:
                    meta = soup.find("meta", charset=True)
                if not meta:
                    meta = soup.new_tag("meta", charset="utf-8")
                    head.insert(0, meta)
                    file_path = self._url_to_filename(url)
                    return str(soup), file_path

        return page_content, file_path

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _config_file_path(category: str, url: str) -> str:
        """Return the recommended config file path for this fix."""
        if category.startswith("info_leak"):
            # Server-level config
            return "server-config/server.conf"
        if "security" in category or "hsts" in category:
            return "server-config/security-headers.conf"
        if "cache" in category or "etag" in category or "vary" in category:
            return "server-config/caching.conf"
        if "compression" in category:
            return "server-config/compression.conf"
        if "x_robots" in category:
            return "server-config/robots-headers.conf"
        if "content_type" in category or "charset" in category:
            return "server-config/content-type.conf"
        return "server-config/http-headers.conf"

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """Convert a URL to a likely source filename."""
        from urllib.parse import urlparse
        path = urlparse(url).path.strip("/")
        if not path or path.endswith("/"):
            return (path or "index") + "index.html"
        if "." not in path.split("/")[-1]:
            return path + "/index.html"
        return path

    @staticmethod
    def _make_diff(before: str, after: str, config: str) -> str:
        """Build a human-readable diff showing the config changes."""
        parts = []
        if before != after:
            diff = difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
            parts.append("".join(diff))
        parts.append(f"\n# === Recommended Server Config ===\n{config}")
        return "\n".join(parts)
