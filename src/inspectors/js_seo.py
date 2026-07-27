from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Known JS framework mount-point patterns (empty = CSR, no server-rendered content)
CSR_MOUNT_PATTERNS = [
    # React
    r'<div\s[^>]*id\s*=\s*["\']root["\'][^>]*>\s*</div>',
    # Vue
    r'<div\s[^>]*id\s*=\s*["\']app["\'][^>]*>\s*</div>',
    # Angular
    r'<[a-z0-9-]+\s[^>]*ng-app[^>]*>\s*</[a-z0-9-]+>',
    r'<[a-z0-9-]+\s[^>]*ng-version[^>]*>',
    # Svelte
    r'<div\s[^>]*id\s*=\s*["\']svelte["\'][^>]*>\s*</div>',
    # Generic
    r'<div\s[^>]*id\s*=\s*["\']mount["\'][^>]*>\s*</div>',
    r'<div\s[^>]*id\s*=\s*["\']render["\'][^>]*>\s*</div>',
]

# Patterns that suggest JS-powered redirects
JS_REDIRECT_PATTERNS = [
    r'window\.location\s*=\s*["\'][^"\']+["\']',
    r'window\.location\.href\s*=\s*["\'][^"\']+["\']',
    r'window\.location\.replace\s*\(\s*["\'][^"\']+["\']\s*\)',
    r'document\.location\s*=\s*["\'][^"\']+["\']',
    r'location\.replace\s*\(\s*["\'][^"\']+["\']\s*\)',
    r'history\.pushState\s*\(\s*[^)]*?["\'][^"\']+["\']',
    r'window\.open\s*\(\s*["\'][^"\']+["\']',
]

# Common SSR framework indicators (positive signals)
SSR_INDICATORS = [
    r'data-server-rendered\s*=\s*["\']true["\']',   # Vue SSR
    r'__NEXT_DATA__',                                 # Next.js
    r'__NUXT__',                                      # Nuxt.js
    r'data-reactroot\s*=',                            # React (but may still be CSR)
    r'<script[^>]*id\s*=\s*["\']__NEXT_DATA__["\']', # Next.js (explicit)
    r'<script[^>]*window\.__NUXT__',                  # Nuxt.js (explicit)
    r'<script[^>]*window\.__INITIAL_STATE__',         # SSR state injection
    r'data-ssr\s*=\s*["\']true["\']',                 # Qwik / generic SSR marker
]


class JSSeoInspector(BaseInspector):
    """Inspect JavaScript-related SEO issues: CSR detection, JS redirects,
    script bloat, noscript fallback, and content-to-markup ratio."""

    inspector_name = "js_seo"

    # Thresholds
    MIN_VISIBLE_TEXT_RATIO = 0.05   # visible text / HTML size — below = red flag
    MAX_SCRIPT_TAGS_WARN = 15       # external + inline scripts
    MAX_SCRIPT_TAGS_CRITICAL = 30
    MAX_INLINE_SCRIPT_SIZE = 50_000  # 50 KB of inline JS
    MIN_PAGE_TEXT_LENGTH = 100       # characters — below = likely CSR

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        html_size = len(html_content)
        soup = BeautifulSoup(html_content, "html.parser")

        # ── 1. CSR Detection: empty framework mount points ──────
        findings.extend(self._check_csr_mount_points(url, html_content))

        # ── 2. Content-to-HTML ratio analysis ───────────────────
        # Pass a COPY of soup — this method decomposes script/style/noscript
        # tags, but later methods (_check_inline_scripts, _check_noscript) need them.
        content_soup = BeautifulSoup(str(soup), "html.parser")
        findings.extend(self._check_content_ratio(url, content_soup, html_content, html_size))

        # ── 3. JS redirects in inline scripts ───────────────────
        findings.extend(self._check_js_redirects(url, html_content))

        # ── 4. Script count & bloat ─────────────────────────────
        findings.extend(self._check_script_count(url, html_content))

        # ── 5. Large inline scripts ─────────────────────────────
        findings.extend(self._check_inline_scripts(url, soup))

        # ── 6. Noscript fallback ────────────────────────────────
        findings.extend(self._check_noscript(url, soup, html_content))

        # ── 7. SSR indicators (positive: note if present) ───────
        findings.extend(self._check_ssr_indicators(url, html_content))

        return findings

    # ── CSR Mount Points ────────────────────────────────────────────

    def _check_csr_mount_points(
        self, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        html_lower = html_content.lower()

        for pattern in CSR_MOUNT_PATTERNS:
            m = re.search(pattern, html_content, re.IGNORECASE)
            if m:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="js_csr_empty_mount",
                    description=(
                        f"Empty JS framework mount point detected: "
                        f"'{m.group(0)[:120]}'. The page appears to render "
                        f"content entirely in the browser (CSR). Search engines "
                        f"may not index this content reliably."
                    ),
                    element=m.group(0)[:200],
                    suggested_value=(
                        "Implement Server-Side Rendering (SSR), Static Site "
                        "Generation (SSG), or use a prerendering service to "
                        "ensure search engines receive complete HTML."
                    ),
                ))
                break  # one finding per page

        return findings

    # ── Content-to-HTML Ratio ───────────────────────────────────────

    def _check_content_ratio(
        self, url: str, soup: BeautifulSoup, html_content: str, html_size: int,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        body = soup.find("body")
        if not body:
            return findings

        # Extract visible text
        for tag in body.find_all(["script", "style", "noscript", "template"]):
            tag.decompose()
        visible_text = body.get_text(separator=" ", strip=True)
        text_len = len(visible_text)

        # Check for extremely low content
        if text_len < self.MIN_PAGE_TEXT_LENGTH and html_size > 5000:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="js_low_visible_text",
                description=(
                    f"Very little visible text ({text_len} chars) despite "
                    f"large HTML payload ({html_size:_} bytes). Content is likely "
                    f"rendered via JavaScript and invisible to crawlers that "
                    f"don't execute JS."
                ),
                current_value=f"{text_len} chars visible, {html_size} bytes HTML",
                raw_metadata={
                    "visible_chars": text_len,
                    "html_bytes": html_size,
                },
                suggested_value=(
                    "Use SSR/SSG to deliver content in the initial HTML. "
                    "Test your page with 'Fetch as Google' in Search Console."
                ),
            ))

        # Check content ratio
        if html_size > 10_000 and text_len > 0:
            ratio = text_len / html_size
            if ratio < self.MIN_VISIBLE_TEXT_RATIO:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="js_low_content_ratio",
                    description=(
                        f"Low content-to-markup ratio ({ratio:.1%}). "
                        f"Only {text_len} chars of text in {html_size:_} bytes of HTML — "
                        f"the page is mostly markup and scripts. Crawlers see very "
                        f"little substance."
                    ),
                    current_value=f"{ratio:.1%} (visible/markup)",
                    suggested_value=(
                        "Reduce JS bundle size, inline critical content, "
                        "or adopt SSR/SSG."
                    ),
                    raw_metadata={
                        "ratio": round(ratio, 4),
                        "visible_chars": text_len,
                        "html_bytes": html_size,
                    },
                ))

        return findings

    # ── JS Redirects ────────────────────────────────────────────────

    def _check_js_redirects(
        self, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        for pattern in JS_REDIRECT_PATTERNS:
            m = re.search(pattern, html_content, re.IGNORECASE)
            if m:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="js_redirect_detected",
                    description=(
                        f"JavaScript redirect detected: '{m.group(0)[:120]}'. "
                        f"Search engines may not follow JS redirects consistently; "
                        f"use 301/302 HTTP redirects instead."
                    ),
                    element=m.group(0)[:200],
                    suggested_value=(
                        "Replace JS redirects with server-side 301 (permanent) "
                        "or 302 (temporary) HTTP redirects."
                    ),
                ))
                break  # one finding per page

        return findings

    # ── Script Count & Bloat ────────────────────────────────────────

    def _check_script_count(
        self, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        external_scripts = re.findall(
            r'<script\s[^>]*src\s*=\s*["\'][^"\']+["\']',
            html_content, re.IGNORECASE,
        )
        inline_scripts = re.findall(
            r'<script\b(?![^>]*\bsrc\s*=)[^>]*>',
            html_content, re.IGNORECASE,
        )
        total = len(external_scripts) + len(inline_scripts)

        if total >= self.MAX_SCRIPT_TAGS_CRITICAL:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="js_excessive_scripts",
                description=(
                    f"Excessive script tags ({total} total: "
                    f"{len(external_scripts)} external, {len(inline_scripts)} inline). "
                    f"This severely impacts page load performance and Core Web Vitals."
                ),
                current_value=str(total),
                suggested_value=(
                    "Bundle and minify scripts, use code splitting, "
                    "defer non-critical JS with 'async' or 'defer' attributes."
                ),
                raw_metadata={
                    "script_count": total,
                    "external": len(external_scripts),
                    "inline": len(inline_scripts),
                },
            ))
        elif total >= self.MAX_SCRIPT_TAGS_WARN:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="js_many_scripts",
                description=(
                    f"High number of script tags ({total} total: "
                    f"{len(external_scripts)} external, {len(inline_scripts)} inline). "
                    f"Consider bundling to reduce HTTP requests."
                ),
                current_value=str(total),
                suggested_value=(
                    "Consolidate into fewer bundles; use async/defer for "
                    "non-critical scripts."
                ),
                raw_metadata={
                    "script_count": total,
                    "external": len(external_scripts),
                    "inline": len(inline_scripts),
                },
            ))

        # Check for scripts without async/defer on external scripts
        blocking_scripts = 0
        for match in re.finditer(
            r'<script\s([^>]*?)src\s*=\s*["\']([^"\']+)["\']([^>]*)>',
            html_content, re.IGNORECASE,
        ):
            attrs = (match.group(1) + match.group(3)).lower()
            if "async" not in attrs and "defer" not in attrs and "type=module" not in attrs:
                blocking_scripts += 1

        if blocking_scripts > 5:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="js_blocking_scripts",
                description=(
                    f"{blocking_scripts} external scripts are render-blocking "
                    f"(no async/defer/module attributes). These delay First "
                    f"Contentful Paint and LCP."
                ),
                current_value=str(blocking_scripts),
                suggested_value=(
                    "Add 'async' (independent scripts) or 'defer' "
                    "(order-dependent scripts) to non-critical <script> tags."
                ),
                raw_metadata={"blocking_scripts": blocking_scripts},
            ))

        return findings

    # ── Large Inline Scripts ────────────────────────────────────────

    def _check_inline_scripts(
        self, url: str, soup: BeautifulSoup,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        for script in soup.find_all("script"):
            if script.get("src"):
                continue  # external script
            content = script.string or ""
            if len(content) > self.MAX_INLINE_SCRIPT_SIZE:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="js_large_inline_script",
                    description=(
                        f"Large inline script ({len(content):_} bytes). "
                        f"Large inline scripts block rendering and increase "
                        f"HTML payload size — move to external file with 'defer' "
                        f"or 'async'."
                    ),
                    current_value=f"{len(content):_} bytes",
                    suggested_value="Extract to external .js file with async/defer.",
                    raw_metadata={"inline_script_bytes": len(content)},
                ))

        return findings

    # ── Noscript Fallback ───────────────────────────────────────────

    def _check_noscript(
        self, url: str, soup: BeautifulSoup, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        noscript_tags = soup.find_all("noscript")

        # Check if there's a CSR mount point but no noscript
        has_csr = any(
            re.search(p, html_content, re.IGNORECASE)
            for p in CSR_MOUNT_PATTERNS
        )
        if has_csr and not noscript_tags:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="js_missing_noscript",
                description=(
                    "Page uses a JS framework mount point but has no "
                    "<noscript> fallback. Users and crawlers without JS "
                    "see an empty page."
                ),
                suggested_value=(
                    "Add a <noscript> tag with a meaningful message and "
                    "basic navigation links."
                ),
            ))

        # Check noscript quality: non-empty content?
        for tag in noscript_tags:
            text = tag.get_text(strip=True)
            if len(text) < 20:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="js_thin_noscript",
                    description=(
                        f"<noscript> tag has very little content "
                        f"({len(text)} chars): '{text[:80]}'. "
                        f"Provide useful navigation and content summary."
                    ),
                    element=str(tag)[:200],
                    current_value=text[:100],
                    suggested_value=(
                        "Include at minimum: site name, navigation links, "
                        "and a message about JavaScript requirements."
                    ),
                ))

        return findings

    # ── SSR Indicators (Positive) ───────────────────────────────────

    def _check_ssr_indicators(
        self, url: str, html_content: str,
    ) -> list[RawFinding]:
        """Check for SSR framework markers — absence is noted for JS-heavy sites."""
        findings: list[RawFinding] = []

        has_ssr = any(
            re.search(p, html_content, re.IGNORECASE)
            for p in SSR_INDICATORS
        )

        # Only flag absence if the page has JS framework patterns
        has_framework = any(
            re.search(p, html_content, re.IGNORECASE)
            for p in CSR_MOUNT_PATTERNS
        ) or bool(re.findall(
            r'<script\s[^>]*src\s*=\s*["\'][^"\']*(?:react|vue|angular|svelte|next|nuxt)[^"\']*["\']',
            html_content, re.IGNORECASE,
        ))

        if has_framework and not has_ssr:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="js_no_ssr_indicator",
                description=(
                    "JavaScript framework detected but no SSR indicators found "
                    "(e.g. __NEXT_DATA__, __NUXT__, data-server-rendered). "
                    "The page likely renders entirely client-side, which is "
                    "suboptimal for SEO."
                ),
                suggested_value=(
                    "Migrate to an SSR/SSG framework: Next.js for React, "
                    "Nuxt for Vue, SvelteKit for Svelte, or Angular Universal."
                ),
            ))

        return findings
