from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urlparse, urljoin

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Characters that should NOT appear in clean URLs
# (reserved characters that have meaning in URIs are OK: / ? # & = +)
PROBLEMATIC_CHARS = re.compile(r"[A-Z]")  # uppercase = problematic for case-duplication
SPECIAL_CHARS = re.compile(r"[_]")  # underscores (hyphens preferred for SEO)
NON_ASCII = re.compile(r"[^\x00-\x7F]")  # non-ASCII characters
STOP_WORDS_IN_URL = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those",
}
# Characters that should be encoded in URLs
UNSAFE_CHARS = re.compile(r'[\s<>"#%{}|\\^~\[\]`]')

# Dynamic URL indicators
DYNAMIC_EXTENSIONS = {".php", ".asp", ".aspx", ".jsp", ".cfm", ".cgi", ".pl", ".do"}

# Max recommended values
MAX_URL_LENGTH = 115  # characters (Bing limit is ~2000 but shorter = better)
MAX_URL_DEPTH = 4     # folder levels


class URLAuditor(BaseInspector):
    """Audit URL structure for SEO best practices: length, characters,
    consistency, depth, dynamic patterns, and readability."""

    inspector_name = "url_audit"

    def __init__(self) -> None:
        self._crawled_urls: list[str] = []

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_crawled_urls(self, urls: list[str]) -> None:
        self._crawled_urls = urls

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not url:
            return findings

        parsed = urlparse(url)
        path = parsed.path or "/"

        findings.extend(self._check_length(url, parsed))
        findings.extend(self._check_case(url, path))
        findings.extend(self._check_special_chars(url, path))
        findings.extend(self._check_depth(url, path))
        findings.extend(self._check_dynamic_patterns(url, parsed))
        findings.extend(self._check_readability(url, path))
        findings.extend(self._check_trailing_slash(url, path))
        findings.extend(self._check_encoding(url, parsed))

        return findings

    # ── URL Length ──────────────────────────────────────────────────

    def _check_length(
        self, url: str, parsed,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        total_len = len(url)
        path_len = len(parsed.path)

        if total_len > 200:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_too_long",
                description=(
                    f"URL is {total_len} characters long. Long URLs are "
                    f"truncated in search results, harder to share, and "
                    f"dilute keyword relevance."
                ),
                current_value=f"{total_len} chars",
                suggested_value="Shorten to under 115 characters; remove unnecessary path segments and parameters.",
                raw_metadata={"url_length": total_len},
            ))
        elif total_len > MAX_URL_LENGTH:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_long",
                description=(
                    f"URL is {total_len} characters (recommend ≤{MAX_URL_LENGTH}). "
                    f"Shorter URLs perform better in SERPs and are easier to share."
                ),
                current_value=f"{total_len} chars",
                suggested_value="Trim filler words and unnecessary path depth.",
                raw_metadata={"url_length": total_len},
            ))

        return findings

    # ── Case Issues ─────────────────────────────────────────────────

    def _check_case(
        self, url: str, path: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Uppercase in path = potential case-duplication
        if PROBLEMATIC_CHARS.search(path):
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_uppercase",
                description=(
                    f"URL path contains uppercase characters. Web servers may "
                    f"treat '/Page' and '/page' as different URLs, creating "
                    f"duplicate content issues."
                ),
                current_value=url[:200],
                suggested_value="Use only lowercase letters in URLs. Set up 301 redirects from uppercase to lowercase variants.",
            ))

        # Mixed case across the site (check against other crawled URLs)
        if self._crawled_urls:
            normalized = path.lower()
            for other in self._crawled_urls:
                other_path = urlparse(other).path
                if other_path.lower() == normalized and other_path != path:
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="url_case_duplicate",
                        description=(
                            f"Case variation detected: '{path}' and "
                            f"'{other_path}' resolve to different URLs. "
                            f"Search engines may see these as duplicate content."
                        ),
                        current_value=url[:200],
                        suggested_value=(
                            "Choose one canonical casing (lowercase preferred); "
                            "301 redirect the variant to the canonical."
                        ),
                        raw_metadata={"variant": other},
                    ))
                    break

        return findings

    # ── Special Characters ──────────────────────────────────────────

    def _check_special_chars(
        self, url: str, path: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Underscores (hyphens are SEO-preferred)
        underscore_count = len(SPECIAL_CHARS.findall(path))
        if underscore_count > 0:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_underscores",
                description=(
                    f"URL contains {underscore_count} underscore(s). Google "
                    f"treats underscores as word joiners (my_page = 'mypage'), "
                    f"while hyphens are word separators (my-page = 'my page')."
                ),
                current_value=url[:200],
                suggested_value="Replace underscores with hyphens for word separation in URLs.",
                raw_metadata={"underscore_count": underscore_count},
            ))

        # Non-ASCII characters
        non_ascii = NON_ASCII.findall(url)
        if non_ascii:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_non_ascii",
                description=(
                    f"URL contains non-ASCII characters ({len(non_ascii)} found). "
                    f"These may be encoded inconsistently across browsers and "
                    f"cause crawling issues."
                ),
                current_value=url[:200],
                suggested_value=(
                    "Transliterate non-ASCII characters to ASCII equivalents "
                    "(e.g., 'ü' → 'ue', 'ñ' → 'n')."
                ),
                raw_metadata={"non_ascii_chars": non_ascii[:5]},
            ))

        # Unsafe/unencoded characters
        unsafe = UNSAFE_CHARS.findall(url)
        if unsafe:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_unsafe_chars",
                description=(
                    f"URL contains unsafe characters: {set(unsafe)}. "
                    f"These should be percent-encoded to avoid parsing errors."
                ),
                current_value=url[:200],
                suggested_value="Percent-encode unsafe characters (e.g., spaces → %20).",
                raw_metadata={"unsafe_chars": list(set(unsafe))},
            ))

        return findings

    # ── URL Depth ───────────────────────────────────────────────────

    def _check_depth(
        self, url: str, path: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        segments = [s for s in path.strip("/").split("/") if s]
        depth = len(segments)

        if depth > MAX_URL_DEPTH:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_too_deep",
                description=(
                    f"URL is {depth} folder levels deep (recommend ≤{MAX_URL_DEPTH}). "
                    f"Deep URLs signal lower importance to crawlers and may "
                    f"not be crawled as frequently."
                ),
                current_value=f"{depth} levels: {' / '.join(segments[-3:])}",
                suggested_value="Flatten the URL structure; move content closer to the root.",
                raw_metadata={"depth": depth, "segments": segments},
            ))
        elif depth == MAX_URL_DEPTH and len(segments[-1]) > 40:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_deep_with_long_slug",
                description=(
                    f"URL is {depth} levels deep with a long final slug "
                    f"({len(segments[-1])} chars). Consider flattening."
                ),
                current_value=f"depth {depth}, slug length {len(segments[-1])}",
                suggested_value="Shorten the final slug or reduce folder depth.",
                raw_metadata={"depth": depth, "slug_length": len(segments[-1])},
            ))

        return findings

    # ── Dynamic Patterns ────────────────────────────────────────────

    def _check_dynamic_patterns(
        self, url: str, parsed,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        path = parsed.path.lower()

        # Dynamic extensions
        for ext in DYNAMIC_EXTENSIONS:
            if path.endswith(ext):
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="url_dynamic_extension",
                    description=(
                        f"URL uses a dynamic extension ('{ext}'). "
                        f"Static, descriptive URLs are preferred for SEO."
                    ),
                    current_value=url[:200],
                    suggested_value=(
                        f"Rewrite '{ext}' URLs to clean, static paths "
                        f"(e.g., /products.php?id=5 → /products/blue-widget)."
                    ),
                ))
                break

        # Numeric IDs in path (no descriptive slug)
        if re.search(r"/\d{3,}(?:/|$)", path) and not re.search(r"/[a-z]{4,}", path):
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_numeric_only",
                description=(
                    "URL path contains only numeric identifiers without "
                    "descriptive keywords. Search engines and users prefer "
                    "human-readable URLs."
                ),
                current_value=path[:200],
                suggested_value="Rewrite to include descriptive keywords (e.g., /12345 → /product-name-12345).",
            ))

        # Query parameter count
        if parsed.query:
            params = [p.split("=")[0] for p in parsed.query.split("&") if p]
            if len(params) >= 4:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="url_many_query_params",
                    description=(
                        f"URL has {len(params)} query parameters. "
                        f"Static URLs are easier to crawl and rank better."
                    ),
                    current_value=f"{len(params)} params",
                    suggested_value="Reduce query parameters; use path-based URLs for key landing pages.",
                    raw_metadata={"param_count": len(params)},
                ))

        return findings

    # ── Readability / Keyword ───────────────────────────────────────

    def _check_readability(
        self, url: str, path: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        segments = [s for s in path.strip("/").split("/") if s]
        if not segments:
            return findings

        last_segment = segments[-1]

        # Check for stop words filling the URL
        words = re.findall(r"[a-z]+", last_segment.lower())
        stop_count = sum(1 for w in words if w in STOP_WORDS_IN_URL)
        if stop_count >= 3:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_stop_words",
                description=(
                    f"URL slug contains {stop_count} stop words (e.g., "
                    f"'the', 'and', 'of'). These dilute keyword density "
                    f"without adding value."
                ),
                current_value=last_segment[:100],
                suggested_value="Remove filler words from the URL slug; keep only meaningful keywords.",
                raw_metadata={"stop_word_count": stop_count},
            ))

        # Check for numbers that look like dates
        date_pattern = re.search(
            r"(?:20\d{2}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]20\d{2})",
            last_segment,
        )
        if date_pattern:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_date_in_slug",
                description=(
                    f"URL contains what looks like a date "
                    f"('{date_pattern.group()}'). "
                    f"Dated URLs make content appear stale even if updated."
                ),
                current_value=last_segment[:100],
                suggested_value=(
                    "Remove dates from URLs unless essential (e.g., news). "
                    "Update the slug when refreshing evergreen content."
                ),
            ))

        # Slug is just a single character or very short
        meaningful = re.sub(r"[^a-zA-Z]", "", last_segment)
        if len(meaningful) < 3 and len(last_segment) > 0:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_short_slug",
                description=(
                    f"URL slug is very short ('{last_segment}'). "
                    f"Descriptive slugs improve CTR in search results."
                ),
                current_value=last_segment,
                suggested_value="Use a descriptive slug with target keywords (e.g., '/blue-widget' not '/bw').",
            ))

        return findings

    # ── Trailing Slash Consistency ──────────────────────────────────

    def _check_trailing_slash(
        self, url: str, path: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        has_slash = path.endswith("/")
        no_slash = not has_slash and path != "/"

        if not self._crawled_urls:
            return findings

        # Compare with other crawled URLs to detect inconsistency
        for other in self._crawled_urls:
            if other == url:
                continue
            other_path = urlparse(other).path
            # Check if one has trailing slash and the other doesn't
            if other_path.rstrip("/") == path.rstrip("/"):
                if (other_path.endswith("/")) != (path.endswith("/")):
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="url_trailing_slash_inconsistent",
                        description=(
                            f"Trailing slash inconsistency: this URL "
                            f"{'has' if has_slash else 'lacks'} a trailing "
                            f"slash while '{other}' does the opposite. "
                            f"This can cause duplicate content."
                        ),
                        current_value=url[:200],
                        suggested_value=(
                            "Choose one trailing slash policy (with or without) "
                            "and 301 redirect the non-canonical variant. Set "
                            "the canonical tag consistently."
                        ),
                        raw_metadata={"other_url": other},
                    ))
                    break

        return findings

    # ── Encoding Issues ─────────────────────────────────────────────

    def _check_encoding(
        self, url: str, parsed,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Double-encoding detection
        if "%25" in url:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_double_encoded",
                description=(
                    "URL appears to be double-encoded (contains '%25', "
                    "which is an encoded '%' sign). Double encoding can "
                    "cause crawling errors."
                ),
                current_value=url[:200],
                suggested_value="Fix URL generation to encode only once.",
            ))

        # Spaces in URL (should be %20)
        if " " in url:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="url_unencoded_spaces",
                description=(
                    "URL contains literal space characters. Spaces must "
                    "be encoded as %20 to be valid."
                ),
                current_value=url[:200],
                suggested_value="Encode spaces as %20 in URL generation.",
            ))

        return findings
