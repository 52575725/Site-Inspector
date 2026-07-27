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
                    category="invalid_jsonld",
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

        # ── 4. Field value quality validation ─────────────────────────
        for block in parsed_blocks:
            value_findings = self._validate_field_values(block, url)
            findings.extend(value_findings)

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
    def _validate_field_values(block: dict, url: str) -> list[RawFinding]:
        """Validate the quality/format of field values (not just presence)."""
        findings: list[RawFinding] = []
        import re
        from urllib.parse import urlparse

        schema_type = block.get("@type", "Unknown")

        # ── URL fields ──────────────────────────────────────────────
        url_fields = ["url", "sameAs", "logo", "image", "thumbnailUrl"]
        for field in url_fields:
            val = block.get(field)
            if not val:
                continue
            if isinstance(val, dict):
                val = val.get("url") or val.get("@id") or ""
            if isinstance(val, list):
                val = val[0] if val else ""
            if not isinstance(val, str) or not val:
                continue
            parsed = urlparse(val)
            # Relative URLs are valid in JSON-LD context (schema.org allows them)
            if parsed.scheme and parsed.scheme not in ("http", "https"):
                findings.append(RawFinding(
                    url=url, inspector="structured_data",
                    category="schema_invalid_url",
                    description=(
                        f"Schema '{schema_type}' field '{field}' has invalid URL: "
                        f"'{val[:100]}'"
                    ),
                    current_value=val[:150],
                    suggested_value="Use a valid https:// URL",
                    raw_metadata={"schema_type": schema_type, "field": field},
                ))

        # ── Phone / email fields ────────────────────────────────────
        phone_fields = ["telephone", "phone", "faxNumber"]
        for field in phone_fields:
            val = block.get(field)
            if not val or not isinstance(val, str):
                continue
            # Must contain at least some digits
            digits = re.sub(r"\D", "", val)
            if len(digits) < 7:
                findings.append(RawFinding(
                    url=url, inspector="structured_data",
                    category="schema_invalid_phone",
                    description=(
                        f"Schema '{schema_type}' field '{field}' value "
                        f"'{val[:50]}' doesn't look like a valid phone number"
                    ),
                    current_value=val[:100],
                    suggested_value="Use international format: +1-555-123-4567",
                    raw_metadata={"schema_type": schema_type, "field": field},
                ))

        email_fields = ["email"]
        for field in email_fields:
            val = block.get(field)
            if not val or not isinstance(val, str):
                continue
            if "@" not in val or "." not in val.split("@")[-1]:
                findings.append(RawFinding(
                    url=url, inspector="structured_data",
                    category="schema_invalid_email",
                    description=(
                        f"Schema '{schema_type}' field '{field}' value "
                        f"'{val[:50]}' doesn't look like a valid email"
                    ),
                    current_value=val[:100],
                    suggested_value="Use a valid email address",
                    raw_metadata={"schema_type": schema_type, "field": field},
                ))

        # ── Price fields ────────────────────────────────────────────
        price_fields = ["price", "lowPrice", "highPrice", "minPrice", "maxPrice"]
        for field in price_fields:
            val = block.get(field)
            if val is None:
                continue
            if isinstance(val, str):
                # Should contain digits
                if not re.search(r"\d", val):
                    findings.append(RawFinding(
                        url=url, inspector="structured_data",
                        category="schema_invalid_price",
                        description=(
                            f"Schema '{schema_type}' field '{field}' value "
                            f"'{val[:50]}' doesn't contain a numeric price"
                        ),
                        current_value=val[:100],
                        suggested_value="Use a numeric price value",
                        raw_metadata={"schema_type": schema_type, "field": field},
                    ))

        # ── Date fields ─────────────────────────────────────────────
        date_fields = [
            "datePublished", "dateModified", "startDate", "endDate",
            "validFrom", "validThrough",
        ]
        for field in date_fields:
            val = block.get(field)
            if not val or not isinstance(val, str):
                continue
            # ISO 8601 check
            iso_pattern = r"^\d{4}-\d{2}-\d{2}"
            if not re.match(iso_pattern, val) and not re.match(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", val
            ):
                findings.append(RawFinding(
                    url=url, inspector="structured_data",
                    category="schema_invalid_date",
                    description=(
                        f"Schema '{schema_type}' field '{field}' value "
                        f"'{val[:50]}' is not in ISO 8601 date format"
                    ),
                    current_value=val[:100],
                    suggested_value="Use ISO 8601 format: 2024-01-15 or 2024-01-15T09:00:00",
                    raw_metadata={"schema_type": schema_type, "field": field},
                ))
            else:
                # Check if date is in the future for published/modified
                from datetime import datetime
                try:
                    date_str = val[:10]
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    now = datetime.utcnow()
                    if field in ("datePublished", "dateModified") and dt > now:
                        findings.append(RawFinding(
                            url=url, inspector="structured_data",
                            category="schema_future_date",
                            description=(
                                f"Schema '{schema_type}' field '{field}' is "
                                f"set to {date_str}, which is in the future"
                            ),
                            current_value=val[:100],
                            suggested_value="Use the actual publication date",
                            raw_metadata={"schema_type": schema_type, "field": field},
                        ))
                except ValueError:
                    pass

        # ── Author fields ───────────────────────────────────────────
        author_fields = ["author", "creator"]
        for field in author_fields:
            val = block.get(field)
            if not val:
                continue
            name = val
            if isinstance(val, dict):
                name = val.get("name", "")
            if not name or not isinstance(name, str):
                continue
            # Flag obviously non-name values
            if len(name) < 2 or len(name) > 100:
                findings.append(RawFinding(
                    url=url, inspector="structured_data",
                    category="schema_invalid_author",
                    description=(
                        f"Schema '{schema_type}' field '{field}' value "
                        f"'{name[:80]}' doesn't look like a real name"
                    ),
                    current_value=str(name)[:100],
                    suggested_value="Use the author's actual name",
                    raw_metadata={"schema_type": schema_type, "field": field},
                ))

        return findings

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
