from __future__ import annotations

import difflib
import re

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class MetaFixer(BaseFixer):
    """Auto-fix meta tags: title, description, OG tags, canonical, viewport."""

    fixer_name = "meta_fixer"
    fix_type = "fully_auto"
    supported_categories = [
        "missing_title", "title_too_short", "title_too_long",
        "missing_meta_description", "meta_description_too_short", "meta_description_too_long",
        "missing_og_tags", "missing_canonical",
        "missing_viewport_meta",  # from mobile inspector
        "missing_form_label",     # add aria-label to unlabeled form fields
        "missing_geo_region", "missing_geo_placename", "missing_geo_position",
    ]

    def __init__(self, default_og_image: str = "",
                 default_og_width: str = "1200",
                 default_og_height: str = "630",
                 site_name: str = "Helin Silver"):
        self.default_og_image = default_og_image or \
            "https://www.helinsilver.com/images/silver-ingots.jpg"
        self.default_og_width = default_og_width
        self.default_og_height = default_og_height
        self.site_name = site_name

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        category = issue.get("category", "")

        if category.startswith("missing_title") or category.startswith("title_too_"):
            page_content = self._fix_title(soup, page_content, issue)

        if category.startswith("missing_meta_description") or category.startswith("meta_description_too_"):
            page_content = self._fix_meta_description(soup, page_content, issue)

        if category == "missing_og_tags":
            page_content = self._fix_og_tags(soup, page_content, issue)

        if category == "missing_canonical":
            page_content = self._fix_canonical(soup, page_content, issue)

        if category == "missing_viewport_meta":
            page_content = self._fix_viewport(soup, page_content)

        if category == "missing_form_label":
            page_content = self._fix_form_label(soup, page_content, issue)

        if category.startswith("missing_geo_"):
            page_content = self._fix_geo_tags(soup, page_content, category)

        diff = difflib.unified_diff(
            (issue.get("before_content") or issue.get("current_value") or "").splitlines(True),
            page_content.splitlines(True),
            lineterm="",
        )

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=issue.get("before_content", ""),
            after_content=page_content,
            diff="\n".join(diff),
        )

    def _fix_title(self, soup: BeautifulSoup, html: str,
                   issue: dict) -> str:
        title_tag = soup.find("title")
        url = issue.get("url", "")

        if not title_tag:
            head = soup.find("head")
            if head:
                new_title = soup.new_tag("title")
                new_title.string = self._generate_title(url)
                head.insert(0, new_title)
        elif title_tag.string and len(title_tag.string.strip()) < 30:
            title_tag.string = self._generate_title(url, existing=title_tag.string.strip())
        elif title_tag.string and len(title_tag.string.strip()) > 65:
            # Only truncate if significantly over — and truncate at word boundary
            text = title_tag.string.strip()
            # Find last space before char 60
            cut = text.rfind(" ", 0, 60)
            if cut > 30:
                title_tag.string = text[:cut]

        return str(soup)

    def _fix_meta_description(self, soup: BeautifulSoup, html: str,
                              issue: dict) -> str:
        desc = soup.find("meta", attrs={"name": "description"})

        page_text = soup.get_text(separator=" ", strip=True)[:300]

        if not desc:
            head = soup.find("head")
            if head:
                new_desc = soup.new_tag("meta", attrs={
                    "name": "description",
                    "content": page_text[:160],
                })
                head.append(new_desc)
        elif desc.get("content") and len(desc["content"].strip()) < 120:
            desc["content"] = page_text[:160]
        elif desc.get("content") and len(desc["content"].strip()) > 165:
            content = desc["content"].strip()
            cut = content.rfind(" ", 0, 160)
            if cut > 80:
                desc["content"] = content[:cut]

        return str(soup)

    def _fix_og_tags(self, soup: BeautifulSoup, html: str,
                     issue: dict) -> str:
        head = soup.find("head")
        if not head:
            return html

        title_tag = soup.find("title")
        desc_tag = soup.find("meta", attrs={"name": "description"})

        title = title_tag.string.strip() if title_tag and title_tag.string else ""
        desc = desc_tag.get("content", "") if desc_tag else ""
        url = issue.get("url", "")

        existing_og = {
            meta.get("property", "")
            for meta in soup.find_all("meta", attrs={"property": True})
        }

        og_tags = [
            ("og:title", title),
            ("og:description", desc),
            ("og:image", self.default_og_image),
            ("og:image:width", self.default_og_width),
            ("og:image:height", self.default_og_height),
            ("og:url", url),
            ("og:type", "website"),
        ]

        for prop, content in og_tags:
            if prop not in existing_og and content:
                new_tag = soup.new_tag("meta", property=prop, content=content)
                head.append(new_tag)

        return str(soup)

    def _fix_canonical(self, soup: BeautifulSoup, html: str,
                       issue: dict) -> str:
        head = soup.find("head")
        if not head:
            return html

        url = issue.get("url", "")
        existing = soup.find("link", rel="canonical")
        if not existing and url:
            new_tag = soup.new_tag("link", rel="canonical", href=url)
            head.append(new_tag)

        return str(soup)

    def _fix_viewport(self, soup: BeautifulSoup, html: str) -> str:
        head = soup.find("head")
        if not head:
            return html

        existing = soup.find("meta", attrs={"name": "viewport"})
        if not existing:
            new_tag = soup.new_tag(
                "meta",
                attrs={"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
            )
            head.insert(0, new_tag)

        return str(soup)

    def _fix_form_label(self, soup: BeautifulSoup, html: str, issue: dict) -> str:
        """Add aria-label to form fields missing associated <label> elements."""
        element_selector = issue.get("element", "")
        if not element_selector:
            return html

        # Try to find the form element — the element field may contain a tag+type hint
        # Format is typically: "<select type='text'>" or "<input type='email'>"
        tag_match = __import__('re').search(r"<(\w+)", element_selector)
        if not tag_match:
            return html
        tag_name = tag_match.group(1)

        # Find unlabeled form elements of this type
        for elem in soup.find_all(tag_name):
            # Skip if already has aria-label or associated label
            if elem.get("aria-label"):
                continue
            elem_id = elem.get("id")
            if elem_id and soup.find("label", attrs={"for": elem_id}):
                continue

            # Generate a descriptive aria-label from name/id/placeholder
            label_text = (
                elem.get("placeholder")
                or elem.get("name")
                or elem.get("id")
                or f"{tag_name} field"
            )
            if label_text:
                elem["aria-label"] = label_text.replace("_", " ").title()

        return str(soup)

    def _generate_title(self, url: str, existing: str = "") -> str:
        path = url.rstrip("/").split("/")[-1] if "/" in url else "Home"
        title_from_path = path.replace("-", " ").title()
        if existing and len(existing) > 10:
            return existing  # Keep existing if it's somewhat meaningful
        return f"{title_from_path} | {self.site_name}"

    def _fix_geo_tags(self, soup, html: str, category: str) -> str:
        """Insert geo meta tags into <head>."""
        head = soup.find("head")
        if not head:
            return html

        geo_tags = {
            "missing_geo_region": ("geo.region", "HK"),
            "missing_geo_placename": ("geo.placename", "Mong Kok, Kowloon, Hong Kong"),
            "missing_geo_position": ("geo.position", "22.3193;114.1694"),
        }
        if category not in geo_tags:
            return html

        name, content = geo_tags[category]
        existing = head.find("meta", attrs={"name": name})
        if not existing:
            new_tag = soup.new_tag("meta", attrs={"name": name, "content": content})
            head.append(new_tag)

        return str(soup)
