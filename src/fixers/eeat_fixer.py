from __future__ import annotations

import difflib
import json
import re
from datetime import datetime

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class EEATFixer(BaseFixer):
    """Auto-fix E-E-A-T signals: author schema, publication dates, byline structure."""

    fixer_name = "eeat_fixer"
    fix_type = "semi_auto"
    supported_categories = [
        "eeat_no_author",
        "eeat_author_no_schema",
        "eeat_author_no_credentials",
        "eeat_no_date",
        "eeat_date_no_schema",
        "eeat_date_schema_only",
    ]

    def __init__(self, site_name: str = "Site Name"):
        self.site_name = site_name

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        category = issue.get("category", "")
        url = issue.get("url", "")

        original = page_content

        if category in ("eeat_author_no_schema",):
            page_content = self._add_author_schema(soup, page_content, url)
        elif category == "eeat_no_author":
            page_content = self._add_author_schema(soup, page_content, url)
            page_content = self._add_meta_author(soup, page_content)
        elif category in ("eeat_no_date", "eeat_date_no_schema"):
            page_content = self._add_date_schema(soup, page_content, url)
            page_content = self._add_visible_date(soup, page_content)
        elif category == "eeat_date_schema_only":
            page_content = self._add_visible_date(soup, page_content)
        elif category == "eeat_author_no_credentials":
            page_content = self._add_credentials_placeholder(soup, page_content)

        if page_content == original:
            return FixResult(
                success=False, issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path="", before_content=original, after_content="",
                error_message="No changes were applicable",
            )

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            page_content.splitlines(keepends=True),
            fromfile="before", tofile="after",
        ))

        return FixResult(
            success=True, issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name, fix_type=self.fix_type,
            file_path=self._url_to_filename(url),
            before_content=original,
            after_content=page_content,
            diff=diff,
        )

    # ── Author Schema ───────────────────────────────────────────────

    def _add_author_schema(
        self, soup: BeautifulSoup, page_content: str, url: str,
    ) -> str:
        """Add or enhance Person/Author schema in JSON-LD."""
        # Try to find author name from page content
        author_name = self._extract_author_name(soup)
        org_name = self.site_name

        # Build Person schema
        person_schema = {
            "@context": "https://schema.org",
            "@type": "Person",
            "name": author_name,
            "url": url,
        }

        # Build or update Article schema with author
        article_schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": self._extract_title(soup),
            "author": {
                "@type": "Person",
                "name": author_name,
            },
            "publisher": {
                "@type": "Organization",
                "name": org_name,
            },
        }

        # Add to existing JSON-LD or create new
        existing_scripts = soup.find_all("script", type="application/ld+json")
        has_article = False
        for script in existing_scripts:
            try:
                data = json.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    if isinstance(block, dict) and block.get("@type") in ("Article", "BlogPosting", "NewsArticle"):
                        if "author" not in block:
                            block["author"] = {"@type": "Person", "name": author_name}
                        if "publisher" not in block:
                            block["publisher"] = {"@type": "Organization", "name": org_name}
                        has_article = True
                script.string = json.dumps(data, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass

        if not has_article:
            # Add new Article schema
            head = soup.find("head")
            if head:
                new_script = soup.new_tag("script", type="application/ld+json")
                new_script.string = json.dumps(article_schema, ensure_ascii=False)
                head.append(new_script)

        return str(soup)

    # ── Date Schema ─────────────────────────────────────────────────

    def _add_date_schema(
        self, soup: BeautifulSoup, page_content: str, url: str,
    ) -> str:
        """Add datePublished/dateModified to schema."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        scripts = soup.find_all("script", type="application/ld+json")
        has_date = False
        for script in scripts:
            try:
                data = json.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    if isinstance(block, dict):
                        if block.get("@type") in ("Article", "BlogPosting", "NewsArticle", "WebPage"):
                            if "datePublished" not in block:
                                block["datePublished"] = today
                            if "dateModified" not in block:
                                block["dateModified"] = today
                            has_date = True
                script.string = json.dumps(data, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass

        if not has_date:
            # Add a minimal WebPage schema with dates
            head = soup.find("head")
            if head:
                schema = {
                    "@context": "https://schema.org",
                    "@type": "WebPage",
                    "datePublished": today,
                    "dateModified": today,
                }
                tag = soup.new_tag("script", type="application/ld+json")
                tag.string = json.dumps(schema, ensure_ascii=False)
                head.append(tag)

        return str(soup)

    # ── Visible Date ────────────────────────────────────────────────

    def _add_visible_date(self, soup: BeautifulSoup, page_content: str) -> str:
        """Add a visible 'Published on' line near the article header."""
        today = datetime.utcnow().strftime("%B %d, %Y")
        date_html = f'<p class="article-date">Published: {today}</p>'

        # Insert after H1 or at the top of the article body
        h1 = soup.find("h1")
        if h1:
            date_tag = BeautifulSoup(date_html, "html.parser")
            h1.insert_after(date_tag)
            return str(soup)

        # Fallback: insert at top of body
        body = soup.find("body")
        if body:
            date_tag = BeautifulSoup(date_html, "html.parser")
            body.insert(0, date_tag)

        return str(soup)

    # ── Meta Author ─────────────────────────────────────────────────

    def _add_meta_author(self, soup: BeautifulSoup, page_content: str) -> str:
        """Add <meta name='author'> to <head>."""
        head = soup.find("head")
        if not head:
            return page_content

        existing = soup.find("meta", attrs={"name": "author"})
        if existing:
            return page_content

        author_name = self._extract_author_name(soup)
        tag = soup.new_tag("meta", attrs={"name": "author", "content": author_name})
        head.insert(0, tag)
        return str(soup)

    # ── Credentials Placeholder ─────────────────────────────────────

    def _add_credentials_placeholder(
        self, soup: BeautifulSoup, page_content: str,
    ) -> str:
        """Add a placeholder for author credentials (needs manual review)."""
        # Find the author byline and append a credentials note
        byline = soup.find(string=re.compile(
            r"(?:By|Written by|Author:)\s+\w+", re.IGNORECASE,
        ))
        if byline and byline.parent:
            note = soup.new_tag(
                "small",
                attrs={"class": "author-credentials", "style": "display:block;color:#666;"},
            )
            note.string = "<!-- TODO: Add author credentials here (e.g., 'Dr. Name, MD') -->"
            byline.parent.insert_after(note)
        return str(soup)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_author_name(soup: BeautifulSoup) -> str:
        """Try to extract author name from the page."""
        # Check meta author
        meta = soup.find("meta", attrs={"name": "author"})
        if meta and meta.get("content"):
            return meta["content"]

        # Check rel=author links
        author_link = soup.find("a", rel="author")
        if author_link:
            return author_link.get_text(strip=True)

        # Check "By [Name]" pattern
        by_match = re.search(
            r"(?:By|Written by|Author:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
            soup.get_text(separator=" ", strip=True)[:2000],
        )
        if by_match:
            return by_match.group(1)

        return "Author Name"  # placeholder

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    @staticmethod
    def _url_to_filename(url: str) -> str:
        from urllib.parse import urlparse
        path = urlparse(url).path.strip("/")
        if not path or path.endswith("/"):
            return (path or "index") + "index.html"
        if "." not in path.split("/")[-1]:
            return path + "/index.html"
        return path
