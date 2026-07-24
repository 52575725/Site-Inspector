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
        page_type = self._guess_page_type(url)
        required = REQUIRED_SCHEMAS.get(page_type, ["Organization", "WebSite"])
        missing_types = [t for t in required if t not in existing_types]

        # For schema_missing_type, only generate the specific missing type
        if category == "schema_missing_type":
            suggested = issue.get("suggested_value", "")
            if suggested and suggested in required:
                missing_types = [suggested]

        # Fallback: if nothing missing but category indicates schema issues,
        # generate at least Organization + WebSite as baseline
        if not missing_types and category in (
            "missing_structured_data", "schema_missing_type", "schema_missing_field",
        ):
            baseline = ["Organization", "WebSite"]
            missing_types = [t for t in baseline if t not in existing_types]

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
        return {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": title or "Silver Products",
            "description": description,
            "url": url,
            "offers": {
                "@type": "Offer",
                "availability": "https://schema.org/InStock",
            },
        }

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
    def _guess_page_type(url: str) -> str:
        path = urlparse(url).path.lower().rstrip("/")
        if path in ("", "/"):
            return "Organization"
        if "/blog/" in path and path.count("/") > 2:
            return "Article"
        if "/products/" in path:
            return "Product"
        if "/blog" == path:
            return "BreadcrumbList"
        return "WebPage"
