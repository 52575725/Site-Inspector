from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Required schema types per page type
REQUIRED_SCHEMAS: dict[str, list[str]] = {
    "homepage": ["Organization", "WebSite", "BreadcrumbList"],
    "product": ["Product", "Organization", "BreadcrumbList"],
    "blog_list": ["Organization", "BreadcrumbList", "ItemList"],
    "blog_article": ["Article", "Organization", "BreadcrumbList"],
    "about": ["Organization", "BreadcrumbList"],
    "contact": ["Organization", "BreadcrumbList", "ContactPoint"],
    "faq": ["FAQPage", "Organization", "BreadcrumbList"],
    "howto": ["HowTo", "Organization", "BreadcrumbList"],
}

# Required fields per Schema.org type (minimal set)
REQUIRED_FIELDS: dict[str, list[str]] = {
    "Organization": ["name", "url"],
    "WebSite": ["name", "url"],
    "BreadcrumbList": ["itemListElement"],
    "Product": ["name"],
    "Article": ["headline"],
    "ItemList": ["itemListElement"],
    "ContactPoint": ["contactType"],
    "FAQPage": ["mainEntity"],
    "HowTo": ["name", "step"],
    "LocalBusiness": ["name", "address"],
    "PostalAddress": ["streetAddress", "addressLocality", "addressCountry"],
}

# Patterns that suggest FAQ-style content
_FAQ_PATTERNS = [
    ("<h2>Frequently Asked", "<h2>FAQ"),
    ("<h3>", "</h3>", "<p>"),  # H3 + paragraph = likely Q&A
    ("question", "answer"),
]
# Min number of Q&A pairs to consider it an FAQ page
_FAQ_MIN_PAIRS = 2

# Patterns that suggest HowTo content
_HOWTO_PATTERNS = [
    ("step 1", "step 2"),
    ("first", "second", "finally"),
    ("how to", "you'll need", "step"),
]


class StructuredDataValidator(BaseInspector):
    """Deep-validate JSON-LD structured data beyond syntax checking."""

    inspector_name = "structured_data"

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")

        if not scripts:
            page_type = self._classify_page(url)
            expected = REQUIRED_SCHEMAS.get(page_type, [])
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="schema_missing_type",
                description=f"No JSON-LD structured data. Expected types for {page_type} page: "
                            f"{', '.join(expected)}",
                current_value="none",
                suggested_value=", ".join(expected),
            )]

        # Parse all JSON-LD blocks
        parsed_blocks: list[dict] = []
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    parsed_blocks.append(data)
                elif isinstance(data, list):
                    parsed_blocks.extend(data)
            except (json.JSONDecodeError, TypeError):
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="schema_invalid_value",
                    description="JSON-LD script contains invalid JSON",
                    element=str(script)[:200],
                ))
                continue

        if not parsed_blocks:
            return findings

        # 1. Check for duplicate @type declarations
        type_counts: dict[str, int] = {}
        for block in parsed_blocks:
            types = block.get("@type", [])
            if isinstance(types, str):
                types = [types]
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1

        for schema_type, count in type_counts.items():
            if count > 1:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="schema_duplicate",
                    description=f"Schema type '{schema_type}' is declared {count} times "
                                f"(should appear at most once)",
                    current_value=str(count),
                    suggested_value="1",
                ))

        # 2. Check required types for this page (with content-aware detection)
        page_type = self._classify_page(url, soup)
        # Augment page_type with content detection
        enhanced_type = self._detect_content_schema(soup, page_type)
        expected_types = REQUIRED_SCHEMAS.get(enhanced_type, REQUIRED_SCHEMAS.get(page_type, []))
        present_types = set(type_counts.keys())

        for required_type in expected_types:
            if required_type not in present_types:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="schema_missing_type",
                    description=f"Missing required schema type '{required_type}' "
                                f"for {page_type} page",
                    current_value=", ".join(sorted(present_types)) if present_types else "none",
                    suggested_value=required_type,
                ))

        # 3. Check required fields per present schema type
        parsed_by_type: dict[str, list[dict]] = {}
        for block in parsed_blocks:
            types = block.get("@type", [])
            if isinstance(types, str):
                types = [types]
            for t in types:
                parsed_by_type.setdefault(t, []).append(block)

        for schema_type, blocks in parsed_by_type.items():
            required_fields = REQUIRED_FIELDS.get(schema_type, [])
            if not required_fields:
                continue

            for block in blocks:
                for field in required_fields:
                    if not block.get(field):
                        findings.append(RawFinding(
                            url=url, inspector=self.inspector_name,
                            category="schema_missing_field",
                            description=f"Schema '{schema_type}' is missing required field '{field}'",
                            current_value=str(block.get("@type", "")),
                            suggested_value=f"{field}: <value>",
                        ))

        return findings

    @staticmethod
    def _classify_page(url: str, soup: BeautifulSoup | None = None) -> str:
        path = urlparse(url).path.lower().rstrip("/")
        if path in ("", "/", "/jp"):
            return "homepage"
        if "/products/" in path:
            return "product"
        if "/blog/" in path and path.count("/") > 2:
            return "blog_article"
        if "/blog" in path:
            return "blog_list"
        if "/about/" in path:
            return "about"
        if "/contact/" in path:
            return "contact"
        if "/faq" in path or "/questions" in path:
            return "faq"
        if "/guide" in path or "/how-to" in path or "/tutorial" in path:
            return "howto"
        return "homepage"

    @staticmethod
    def _detect_content_schema(soup: BeautifulSoup, page_type: str) -> str:
        """Detect FAQ / HowTo patterns from page content.

        Upgrades page_type when content matches known patterns,
        even if the URL doesn't explicitly contain /faq/ or /guide/.
        """
        # FAQ detection: count <details> + <h3> followed closely by <p>
        details_count = len(soup.find_all("details"))
        h3_count = 0
        for h3 in soup.find_all("h3"):
            next_elem = h3.find_next_sibling()
            if next_elem and next_elem.name in ("p", "div", "ul", "ol"):
                h3_count += 1
        qa_pairs = details_count + h3_count

        if qa_pairs >= _FAQ_MIN_PAIRS and page_type not in ("product",):
            return "faq"

        # HowTo detection: sequential numbered steps
        h2_texts = [h.get_text(strip=True).lower() for h in soup.find_all("h2")]
        step_count = sum(1 for t in h2_texts
                        if t.startswith("step ") or t.startswith("step-")
                        or any(t.startswith(f"{i}.") for i in range(1, 10))
                        or any(t.startswith(f"{i})") for i in range(1, 10)))
        if step_count >= 3:
            return "howto"

        return page_type
