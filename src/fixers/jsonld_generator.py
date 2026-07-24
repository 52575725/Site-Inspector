from __future__ import annotations

import difflib
import json
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


# Required schema types per page type (for completeness checking)
REQUIRED_SCHEMAS: dict[str, list[str]] = {
    "Organization": ["Organization", "WebSite", "BreadcrumbList"],
    "Product": ["Product", "Organization", "BreadcrumbList"],
    "Article": ["Article", "Organization", "BreadcrumbList"],
    "BreadcrumbList": ["Organization", "BreadcrumbList", "ItemList"],
    "WebPage": ["Organization", "BreadcrumbList"],
    "FAQ": ["FAQPage", "Organization", "BreadcrumbList"],
    "HowTo": ["HowTo", "Organization", "BreadcrumbList"],
}


class JsonLdGenerator(BaseFixer):
    """Auto-generate JSON-LD structured data for pages."""

    fixer_name = "jsonld_generator"
    fix_type = "fully_auto"
    supported_categories = ["missing_structured_data", "invalid_jsonld", "schema_missing_type", "schema_missing_field"]

    def __init__(self, org_name: str = "", org_alt_name: str = "",
                 org_address: dict | None = None, domain: str = "",
                 geo_lat: float | None = None, geo_lon: float | None = None):
        self.org_name = org_name or "Hong Kong Changjiang International Limited"
        self.org_alt_name = org_alt_name or "Helin Silver"
        self.org_address = org_address or {"@type": "PostalAddress", "addressCountry": "HK"}
        self.domain = domain or "https://www.helinsilver.com"
        self.geo_lat = geo_lat
        self.geo_lon = geo_lon

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        url = issue.get("url", "")
        category = issue.get("category", "")

        # Extract page data
        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else ""

        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag.get("content", "") if desc_tag else ""

        # Detect existing schema types on the page
        existing_types = self._get_existing_types(soup)

        # Determine which schemas to generate
        page_type = self._guess_page_type(url, soup)
        required = REQUIRED_SCHEMAS.get(page_type, ["Organization", "WebSite"])
        missing_types = [t for t in required if t not in existing_types]

        # For schema_missing_type, only generate the specific missing type
        if category == "schema_missing_type":
            suggested = issue.get("suggested_value", "")
            if suggested and suggested in required and suggested not in existing_types:
                missing_types = [suggested]

        # IMPORTANT: no fallback that re-introduces already-present types.
        # If existing_types already covers all required schemas, skip.
        # The previous fallback was the root cause of duplicate JSON-LD blocks.
        if not missing_types:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message="All required schema types already present",
            )

        head = soup.find("head")
        if not head:
            soup.insert(0, BeautifulSoup("<head></head>", "html.parser").find("head"))
            head = soup.find("head")

        generated_count = 0
        for schema_type in missing_types:
            ld_json = self._generate_jsonld(schema_type, url, title, description, soup)
            if ld_json:
                script_tag = soup.new_tag("script", type="application/ld+json")
                script_tag.string = json.dumps(ld_json, ensure_ascii=False, indent=2)
                head.append(script_tag)
                generated_count += 1

        if generated_count == 0:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=issue.get("file_path", ""),
                before_content=page_content, after_content=page_content,
                error_message="Could not generate any missing schema types",
            )

        new_content = str(soup)
        diff = difflib.unified_diff(
            page_content.splitlines(True),
            new_content.splitlines(True),
            lineterm="",
        )

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=page_content,
            after_content=new_content,
            diff="\n".join(diff),
        )

    def _generate_jsonld(self, page_type: str, url: str, title: str,
                         description: str, soup: BeautifulSoup) -> dict | None:
        if page_type == "Organization":
            return self._org_schema(url, title, description)
        if page_type == "Product":
            return self._product_schema(url, title, description, soup)
        if page_type == "Article":
            return self._article_schema(url, title, description, soup)
        if page_type == "BreadcrumbList":
            return self._breadcrumb_schema(url)
        if page_type == "FAQ":
            return self._faq_schema(soup)
        if page_type == "HowTo":
            return self._howto_schema(url, title, description, soup)
        return self._website_schema(url, title, description)

    def _org_schema(self, url: str, title: str, description: str) -> dict:
        schema = {
            "@context": "https://schema.org",
            "@type": "LocalBusiness",
            "name": self.org_name,
            "alternateName": self.org_alt_name,
            "url": url,
            "description": description or title,
            "address": self.org_address,
        }
        if self.geo_lat is not None and self.geo_lon is not None:
            schema["geo"] = {
                "@type": "GeoCoordinates",
                "latitude": self.geo_lat,
                "longitude": self.geo_lon,
            }
        return schema

    def _product_schema(self, url: str, title: str, description: str,
                        soup: BeautifulSoup) -> dict:
        """Generate Product schema with Offers, extracting price/currency from page if present."""
        # Try to extract price from page content
        import re
        body_text = soup.get_text(separator=" ", strip=True) if soup else ""
        price_match = re.search(
            r"(?:USD|US\$|\\$|HK\$|HKD)\s*([\d,]+\.?\d*)",
            body_text, re.IGNORECASE,
        )
        currency = "USD"
        price = None
        if price_match:
            price = price_match.group(1).replace(",", "")
            if "HK$" in price_match.group(0) or "HKD" in price_match.group(0):
                currency = "HKD"

        schema: dict = {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": title or "Silver Products",
            "description": description or title,
            "url": url,
            "offers": {
                "@type": "Offer",
                "availability": "https://schema.org/InStock",
                "priceCurrency": currency,
            },
        }
        if price:
            schema["offers"]["price"] = price
        # Add image if page has a product image
        main_img = soup.find("img", attrs={"class": re.compile(r"product|hero|main", re.I)})
        if not main_img:
            main_img = soup.find("img", src=True)
        if main_img and main_img.get("src"):
            src = main_img["src"]
            if src.startswith("/"):
                from urllib.parse import urljoin
                src = urljoin(self.domain, src)
            schema["image"] = src
        return schema

    def _article_schema(self, url: str, title: str, description: str,
                        soup: BeautifulSoup) -> dict:
        return {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": description,
            "url": url,
        }

    def _breadcrumb_schema(self, url: str) -> dict:
        path_parts = urlparse(url).path.strip("/").split("/")
        items = []
        accumulated = ""
        position = 1
        for part in path_parts:
            if part == "jp":
                continue
            accumulated += f"/{part}"
            name = part.replace("-", " ").title() if part else "Home"
            items.append({
                "@type": "ListItem",
                "position": position,
                "name": name,
                "item": f"{self.domain}{accumulated}",
            })
            position += 1
        return {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": items,
        }

    def _faq_schema(self, soup: BeautifulSoup) -> dict | None:
        """Generate FAQPage schema from <details> or H3+P Q&A pairs on the page."""
        main_entities: list[dict] = []

        # Strategy 1: <details> elements (common FAQ markup)
        for details in soup.find_all("details"):
            summary = details.find("summary")
            question = summary.get_text(strip=True) if summary else ""
            # Answer = everything in <details> except <summary>
            if summary:
                summary.decompose()
            answer = details.get_text(separator=" ", strip=True)
            if summary:
                # Restore for idempotency
                details.insert(0, summary)
            if question and answer and len(question) > 5 and len(answer) > 20:
                main_entities.append({
                    "@type": "Question",
                    "name": question[:200],
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": answer[:1000],
                    },
                })
                if len(main_entities) >= 10:
                    break

        # Strategy 2: H3 followed by P/div as Q&A
        if len(main_entities) < 2:
            for h3 in soup.find_all("h3"):
                question = h3.get_text(strip=True)
                if not question or len(question) < 5:
                    continue
                next_sib = h3.find_next_sibling()
                if next_sib and next_sib.name in ("p", "div", "ul", "ol"):
                    answer = next_sib.get_text(separator=" ", strip=True)
                    if answer and len(answer) > 20:
                        main_entities.append({
                            "@type": "Question",
                            "name": question[:200],
                            "acceptedAnswer": {
                                "@type": "Answer",
                                "text": answer[:1000],
                            },
                        })
                if len(main_entities) >= 10:
                    break

        if not main_entities:
            return None

        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": main_entities,
        }

    def _howto_schema(self, url: str, title: str, description: str,
                      soup: BeautifulSoup) -> dict | None:
        """Generate HowTo schema from step-by-step content on the page."""
        steps: list[dict] = []
        h2_list = soup.find_all("h2") if soup else []

        for i, h2 in enumerate(h2_list, 1):
            heading = h2.get_text(strip=True)
            if not heading or len(heading) < 3:
                continue

            # Collect content until the next h2
            content_parts: list[str] = []
            elem = h2.find_next_sibling()
            while elem and elem.name != "h2":
                text = elem.get_text(separator=" ", strip=True)
                if text and len(text) > 10:
                    content_parts.append(text)
                elem = elem.find_next_sibling()

            instruction = " ".join(content_parts[:3]) if content_parts else heading

            if len(instruction) > 20:
                steps.append({
                    "@type": "HowToStep",
                    "position": len(steps) + 1,
                    "name": heading[:120],
                    "itemListElement": [{
                        "@type": "HowToDirection",
                        "text": instruction[:500],
                    }],
                })

        if len(steps) < 2:
            return None

        # Estimate total time from content
        import re
        body_text = soup.get_text(separator=" ", strip=True) if soup else ""
        time_match = re.search(
            r"(\d+)\s*(?:minute|min)s?", body_text, re.IGNORECASE,
        )
        total_time = None
        if time_match:
            total_time = f"PT{time_match.group(1)}M"

        schema: dict = {
            "@context": "https://schema.org",
            "@type": "HowTo",
            "name": title or "Step-by-Step Guide",
            "description": description or title,
            "step": steps,
        }
        if total_time:
            schema["totalTime"] = total_time

        return schema

    def _website_schema(self, url: str, title: str, description: str) -> dict:
        return {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": title or "Helin Silver",
            "description": description,
            "url": url,
        }

    @staticmethod
    def _get_existing_types(soup: BeautifulSoup) -> set[str]:
        """Get all @type values from existing JSON-LD scripts on the page."""
        types = set()
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    t = data.get("@type", "")
                    if isinstance(t, str):
                        types.add(t)
                    elif isinstance(t, list):
                        types.update(t)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            t = item.get("@type", "")
                            if isinstance(t, str):
                                types.add(t)
            except (json.JSONDecodeError, TypeError):
                pass
        return types

    @staticmethod
    def _guess_page_type(url: str, soup: BeautifulSoup | None = None) -> str:
        path = urlparse(url).path.lower().rstrip("/")
        if path in ("", "/"):
            return "Organization"
        if "/blog/" in path and path.count("/") > 2:
            return "Article"
        if "/products/" in path:
            return "Product"
        if "/blog" == path:
            return "BreadcrumbList"
        if "/faq" in path or "/questions" in path:
            return "FAQ"
        if "/guide" in path or "/how-to" in path or "/tutorial" in path:
            return "HowTo"
        # Content-based detection
        if soup is not None:
            details = len(soup.find_all("details"))
            h3_with_content = 0
            for h3 in soup.find_all("h3"):
                next_sib = h3.find_next_sibling()
                if next_sib and next_sib.name in ("p", "div", "ul", "ol"):
                    h3_with_content += 1
            if details + h3_with_content >= 2:
                return "FAQ"
            h2_texts = [h.get_text(strip=True).lower() for h in soup.find_all("h2")]
            steps = sum(1 for t in h2_texts if t.startswith("step ") or t.startswith("step-"))
            if steps >= 3:
                return "HowTo"
        return "WebPage"
