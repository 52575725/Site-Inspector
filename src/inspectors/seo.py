from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding


class SEOInspector(BaseInspector):
    """Inspect SEO basics: meta tags, headings, canonical, hreflang, JSON-LD, OG tags,
       Twitter cards, image SEO, internal link structure."""

    inspector_name = "seo"

    def __init__(self):
        self._all_urls: list[str] = []
        self._incoming_links: dict[str, set[str]] = {}
        self._target_languages: dict[str, str] = {}

    def set_target_languages(self, languages: dict[str, str]) -> None:
        """Set language→path mapping from target config (e.g. {"en": "/", "ja": "/jp/"})."""
        self._target_languages = languages

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_all_urls(self, urls: list[str]) -> None:
        self._all_urls = urls

    def add_incoming_links(self, target_url: str, source_urls: list[str]) -> None:
        """Record incoming internal links for orphan page detection."""
        normalized = target_url.rstrip("/")
        if normalized not in self._incoming_links:
            self._incoming_links[normalized] = set()
        self._incoming_links[normalized].update(u.rstrip("/") for u in source_urls)

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="empty_page", description="Page has no HTML content",
            ))
            return findings

        soup = BeautifulSoup(html_content, "html.parser")

        findings.extend(self._check_title(soup, url))
        findings.extend(self._check_meta_description(soup, url))
        findings.extend(self._check_h1(soup, url))
        findings.extend(self._check_heading_hierarchy(soup, url))
        findings.extend(self._check_canonical(soup, url))
        findings.extend(self._check_hreflang(soup, url))
        findings.extend(self._check_og_tags(soup, url))
        findings.extend(self._check_twitter_card(soup, url))
        findings.extend(self._check_jsonld(soup, url))
        findings.extend(self._check_image_seo(soup, url))
        findings.extend(self._check_internal_links(soup, url))
        findings.extend(self._check_geo_tags(soup, url))

        return findings

    def _check_title(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        title_tag = soup.find("title")
        if not title_tag or not title_tag.string:
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_title",
                description="Page has no <title> tag",
            )]

        title = title_tag.string.strip()
        findings = []

        if len(title) < 30:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="title_too_short",
                description=f"Title is too short ({len(title)} chars, min 30): '{title}'",
                current_value=title,
            ))
        elif len(title) > 60:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="title_too_long",
                description=f"Title is too long ({len(title)} chars, max 60): '{title}'",
                current_value=title,
            ))

        return findings

    def _check_meta_description(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        desc = soup.find("meta", attrs={"name": "description"})
        if not desc or not desc.get("content"):
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_meta_description",
                description="Page has no meta description",
            )]

        content = desc["content"].strip()
        findings = []

        if len(content) < 120:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="meta_description_too_short",
                description=f"Meta description too short ({len(content)} chars, min 120)",
                current_value=content,
            ))
        elif len(content) > 160:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="meta_description_too_long",
                description=f"Meta description too long ({len(content)} chars, max 160)",
                current_value=content,
            ))

        return findings

    def _check_h1(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        h1_tags = soup.find_all("h1")
        if not h1_tags:
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_h1",
                description="Page has no H1 tag",
            )]
        if len(h1_tags) > 1:
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="multiple_h1",
                description=f"Page has {len(h1_tags)} H1 tags (should have exactly 1)",
            )]
        return []

    def _check_heading_hierarchy(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        headings: list[int] = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(tag.name[1])
            headings.append(level)

        if not headings:
            return []

        # Check for skipped levels
        prev = headings[0]
        for curr in headings[1:]:
            if curr > prev + 1:
                return [RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="h_tag_skip",
                    description=f"Heading level skip: H{prev} → H{curr}",
                    current_value=f"H{prev}→H{curr}",
                )]
            prev = curr

        return []

    def _check_canonical(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        canonical = soup.find("link", attrs={"rel": "canonical"})
        if not canonical or not canonical.get("href"):
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_canonical",
                description="Page has no canonical URL",
            )]
        return []

    def _check_hreflang(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        """Check for hreflang alternate links (critical for multilingual sites)."""
        langs_config = self._target_languages
        if not langs_config or len(langs_config) <= 1:
            return []  # Single-language site, hreflang not needed

        hreflangs = soup.find_all("link", attrs={"rel": "alternate"})
        hreflangs = [h for h in hreflangs if h.get("hreflang")]

        if not hreflangs:
            lang_names = ", ".join(langs_config.keys())
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_hreflang",
                description=f"Page has no hreflang alternate links (multilingual site: {lang_names})",
            )]

        # Check for reciprocal hreflang across all configured languages
        present_langs = {h.get("hreflang") for h in hreflangs}
        expected_langs = set(langs_config.keys())
        if present_langs != expected_langs:
            missing = expected_langs - present_langs
            extra = present_langs - expected_langs - {"x-default"}
            parts = []
            if missing:
                parts.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                parts.append(f"unexpected: {', '.join(sorted(extra))}")
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="incomplete_hreflang",
                description=f"Hreflang mismatch — {'; '.join(parts)}",
                current_value=", ".join(sorted(present_langs)),
                suggested_value=", ".join(sorted(expected_langs)),
            )]

        return []

    def _check_og_tags(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        required_og = ["og:title", "og:description", "og:image", "og:url", "og:type"]
        present = set()
        for meta in soup.find_all("meta"):
            prop = meta.get("property", "")
            if prop.startswith("og:"):
                if meta.get("content"):
                    present.add(prop)

        missing = [og for og in required_og if og not in present]
        if missing:
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_og_tags",
                description=f"Missing Open Graph tags: {', '.join(missing)}",
                current_value=", ".join(sorted(present)) if present else "none",
            )]
        return []

    def _check_twitter_card(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        required = ["twitter:card", "twitter:title", "twitter:description", "twitter:image"]
        present = set()
        for meta in soup.find_all("meta"):
            name = meta.get("name", "")
            if name.startswith("twitter:") and meta.get("content"):
                present.add(name)

        missing = [t for t in required if t not in present]
        if missing:
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_twitter_cards",
                description=f"Missing Twitter card tags: {', '.join(missing)}",
                current_value=", ".join(sorted(present)) if present else "none",
            )]
        return []

    def _check_image_seo(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        findings = []
        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt")
            if alt is None:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="image_missing_alt",
                    description=f"Image missing alt attribute: src='{src[:80]}'",
                    element=str(img)[:200],
                    current_value="none",
                    suggested_value="Descriptive alt text",
                ))
            elif alt.strip() == "":
                # Empty alt is OK for decorative images (role="presentation" or in <button>)
                role = img.get("role", "")
                parent = img.parent.name if img.parent else ""
                if role != "presentation" and parent not in ("button",):
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="image_empty_alt",
                        description=f"Image has empty alt attribute (may need descriptive text): "
                                    f"src='{src[:80]}'",
                        element=str(img)[:200],
                        current_value="(empty)",
                        suggested_value="Descriptive alt text or role='presentation'",
                    ))
        return findings

    def _check_internal_links(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        findings = []

        # Collect all internal links from this page
        internal_links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if href.startswith("/") and not href.startswith("//"):
                internal_links.append(href)

        unique_links = len(set(internal_links))
        if unique_links == 0 and "/blog/" not in url:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="internal_deep_page",
                description="Page has no internal outgoing links (potential orphan or dead-end page)",
            ))

        # Check link depth (rough: count path segments)
        path = urlparse(url).path.strip("/")
        depth = len([p for p in path.split("/") if p]) if path else 0
        if depth > 3:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="internal_deep_page",
                description=f"Page is {depth} levels deep from root (recommend ≤3)",
                current_value=str(depth),
            ))

        return findings

    def _check_geo_tags(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        """Check for geo location meta tags (geo.region, geo.placename, geo.position)."""
        findings = []
        head = soup.find("head")
        if not head:
            return findings

        geo_checks = [
            ("geo.region", "missing_geo_region",
             "<meta name=\"geo.region\" content=\"HK\" />"),
            ("geo.placename", "missing_geo_placename",
             "<meta name=\"geo.placename\" content=\"Mong Kok, Kowloon, Hong Kong\" />"),
            ("geo.position", "missing_geo_position",
             "<meta name=\"geo.position\" content=\"22.3193;114.1694\" />"),
        ]
        for name, category, suggested in geo_checks:
            existing = head.find("meta", attrs={"name": name})
            if not existing:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=category,
                    description=f"Missing geo meta tag: {name} — helps search engines understand business location",
                    suggested_value=suggested,
                ))
        return findings

    def _check_jsonld(self, soup: BeautifulSoup, url: str) -> list[RawFinding]:
        scripts = soup.find_all("script", type="application/ld+json")
        if not scripts:
            page_type = self._guess_page_type(url)
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_structured_data",
                description=f"No JSON-LD structured data found (would benefit {page_type} page)",
            )]

        # Basic validation: check if JSON-LD is parseable
        import json
        for script in scripts:
            try:
                json.loads(script.string)
            except (json.JSONDecodeError, TypeError):
                return [RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="invalid_jsonld",
                    description="JSON-LD script has invalid JSON",
                    element=str(script)[:200],
                )]
        return []

    def _guess_page_type(self, url: str) -> str:
        path = urlparse(url).path.lower()
        # Strip configured language path prefixes
        for lang_path in self._target_languages.values():
            prefix = lang_path.strip("/")
            if prefix and path.startswith(f"/{prefix}"):
                path = path[len(prefix) + 1:]
                break
        if path in ("/", ""):
            return "Organization/Home"
        if "/blog/" in path:
            return "Article"
        if "/products/" in path:
            return "Product"
        if "/about/" in path:
            return "About"
        if "/contact/" in path:
            return "Contact"
        return "WebPage"
