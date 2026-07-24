from __future__ import annotations

import json
import logging
import re
from typing import Optional, Set
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Trust page URL patterns (partial match)
TRUST_PAGE_PATTERNS = {
    "about": ["/about", "/about-us", "/aboutus", "/our-story", "/company",
              "/team", "/who-we-are", "/overview"],
    "contact": ["/contact", "/contact-us", "/contactus", "/get-in-touch",
                "/reach-us", "/support", "/help"],
    "privacy": ["/privacy", "/privacy-policy", "/privacy-policy", "/privacy-notice",
                "/data-protection", "/personal-data"],
    "terms": ["/terms", "/terms-of-service", "/tos", "/terms-and-conditions",
              "/legal", "/conditions", "/disclaimer"],
}

# Patterns that suggest YMYL (Your Money or Your Life) content
YMYL_PATTERNS = [
    r"\b(medical|health|disease|cancer|diabetes|symptom|treatment|diagnosis|"
    r"therapy|surgery|medication|prescription|doctor|hospital|clinic)\b",
    r"\b(financ|invest|loan|mortgage|credit|debt|tax|insurance|retirement|"
    r"stock|trading|crypto|bitcoin|forex|saving|budget)\b",
    r"\b(legal|law|attorney|court|litigation|contract|compliance|regulation|"
    r"gdpr|privacy law|constitution)\b",
    r"\b(news|journalism|breaking|report|correspondent|editorial)\b",
    r"\b(government|election|vote|candidate|policy|legislation|civic)\b",
]

# Credential indicators in author bios
CREDENTIAL_PATTERNS = [
    r"\b(PhD|MD|DO|RN|CPA|CFA|Esq\.?|JD|PE|DDS|PharmD|PsyD|EdD|DVM)\b",
    r"\b(Certified|Licensed|Registered|Board[-\s]Certified|Accredited)\b",
    r"\b(Professor|Researcher|Scientist|Engineer|Attorney|Doctor|Specialist|Expert)\b",
]

# Date patterns for content freshness
DATE_META_PATTERNS = [
    r'<meta[^>]*name\s*=\s*["\'](?:article:published_time|date|pubdate|dc\.date|'
    r'dc\.date\.issued|citation_publication_date)["\']',
    r'<meta[^>]*property\s*=\s*["\']article:(?:published_time|modified_time)["\']',
]

# Visible date display patterns (near author/byline)
BYLINE_DATE_PATTERNS = [
    r'(?:Published|Posted|Written|Updated|Last\s+updated)[:\s]+'
    r'(?:on\s+)?(?:\w+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})',
]


class EEATInspector(BaseInspector):
    """Inspect E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
    signals for content pages.

    Post-HCU, Google heavily weights these signals, especially for YMYL content.
    """

    inspector_name = "eeat"

    def __init__(self) -> None:
        self._crawled_urls: list[str] = []
        self._trust_pages: dict[str, str | None] = {
            "about": None, "contact": None, "privacy": None, "terms": None,
        }

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_crawled_urls(self, urls: list[str]) -> None:
        """Receive all crawled URLs to check for trust page existence."""
        self._crawled_urls = urls
        self._map_trust_pages()

    def _map_trust_pages(self) -> None:
        """Pre-compute which trust pages exist in the crawled set."""
        for page_type, patterns in TRUST_PAGE_PATTERNS.items():
            for url in self._crawled_urls:
                path = urlparse(url).path.lower().rstrip("/")
                for pattern in patterns:
                    if path == pattern or path.endswith(pattern):
                        self._trust_pages[page_type] = url
                        break
                if self._trust_pages[page_type]:
                    break

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        # Only inspect article/blog/content pages (not trust pages themselves)
        path = urlparse(url).path.lower()
        is_trust_page = any(
            any(pattern in path for pattern in patterns)
            for patterns in TRUST_PAGE_PATTERNS.values()
        )
        if is_trust_page:
            return findings

        # ── 1. Author signals ─────────────────────────────────
        findings.extend(self._check_author(soup, url, html_content))

        # ── 2. Publication date signals ───────────────────────
        findings.extend(self._check_dates(soup, url, html_content))

        # ── 3. External references / citations ────────────────
        findings.extend(self._check_references(soup, url))

        # ── 4. YMYL content + disclaimers ─────────────────────
        findings.extend(self._check_ymyl(soup, url, text))

        # ── 5. Cross-site trust signals ───────────────────────
        findings.extend(self._check_trust_coverage(url))

        return findings

    # ── Author Signals ──────────────────────────────────────────────

    def _check_author(
        self, soup: BeautifulSoup, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Check JSON-LD for author
        has_schema_author = False
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    # Direct author field
                    if block.get("author"):
                        has_schema_author = True
                    # Article -> author
                    if isinstance(block, dict):
                        for key in ("author", "creator", "publisher"):
                            if block.get(key):
                                has_schema_author = True
            except (json.JSONDecodeError, TypeError):
                pass

        # HTML-level author signals
        has_html_author = bool(
            soup.find("a", rel="author")
            or soup.find("a", href=re.compile(r"/author/|/authors/|/by/"))
            or soup.find("meta", attrs={"name": "author"})
        )

        # Heuristic: look for "By [Name]" pattern near the top of the article
        by_pattern = re.search(
            r'(?:By|Written by|Author:|Author\s)\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})',
            soup.get_text(separator=" ", strip=True)[:2000],
        )
        has_visible_byline = bool(by_pattern)

        # Check for author credentials in the visible byline area
        author_area_text = ""
        if by_pattern:
            # Grab ~500 chars around the byline
            idx = by_pattern.start()
            author_area_text = soup.get_text(separator=" ", strip=True)[
                max(0, idx - 200):idx + 300
            ]

        has_credentials = any(
            re.search(p, author_area_text, re.IGNORECASE)
            for p in CREDENTIAL_PATTERNS
        ) if author_area_text else False

        # ── Report ──

        if not has_schema_author and not has_html_author and not has_visible_byline:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_no_author",
                description=(
                    "No author information found: no schema author, no meta author, "
                    "no rel=author link, and no visible byline. Content pages "
                    "should clearly indicate authorship for E-E-A-T."
                ),
                suggested_value=(
                    "Add Author schema (Person type), a visible byline "
                    "('By [Author Name]'), and ideally a link to the author's "
                    "bio page."
                ),
            ))

        if has_visible_byline and not has_schema_author:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_author_no_schema",
                description=(
                    "Author name is visible but no structured data (schema.org "
                    "Person/Author) found. Schema helps search engines verify "
                    "authorship for E-E-A-T."
                ),
                suggested_value=(
                    "Add Author/Person JSON-LD with name, url (bio page), "
                    "and optionally sameAs (social profiles)."
                ),
            ))

        if has_visible_byline and not has_html_author and not has_schema_author:
            # Downgrade: at least the byline exists
            pass  # Don't double-report

        if has_visible_byline and not has_credentials:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_author_no_credentials",
                description=(
                    "Author byline found but no credentials or qualifications "
                    "detected in the author area. For YMYL topics, Google "
                    "values demonstrated expertise."
                ),
                suggested_value=(
                    "Include author credentials, qualifications, or experience "
                    'in the author bio area (e.g., "Dr. Jane Smith, MD, '
                    'Board-Certified Dermatologist").'
                ),
            ))

        return findings

    # ── Publication Date ────────────────────────────────────────────

    def _check_dates(
        self, soup: BeautifulSoup, url: str, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        has_meta_date = any(
            re.search(p, html_content, re.IGNORECASE)
            for p in DATE_META_PATTERNS
        )
        has_visible_date = bool(re.search(
            BYLINE_DATE_PATTERNS[0],
            soup.get_text(separator=" ", strip=True)[:3000],
            re.IGNORECASE,
        ))
        # Schema date
        has_schema_date = False
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    if isinstance(block, dict):
                        if block.get("datePublished") or block.get("dateModified"):
                            has_schema_date = True
            except (json.JSONDecodeError, TypeError):
                pass

        if not has_meta_date and not has_schema_date and not has_visible_date:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_no_date",
                description=(
                    "No publication or modification date found. Content "
                    "freshness is an important E-E-A-T signal — Google wants "
                    "to show users current information."
                ),
                suggested_value=(
                    "Add datePublished and dateModified to your Article "
                    "schema, and show a visible 'Published on [date]' "
                    "(and 'Updated on [date]' if revised)."
                ),
            ))

        if has_visible_date and not has_schema_date:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_date_no_schema",
                description=(
                    "Date is visible on the page but no schema datePublished/"
                    "dateModified found. Schema dates help Google display "
                    "date-rich snippets in SERPs."
                ),
                suggested_value="Add datePublished to your Article/BlogPosting JSON-LD.",
            ))

        # Check dateModified exists (not just datePublished)
        if has_schema_date and not has_visible_date:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_date_schema_only",
                description=(
                    "Date is in schema markup but not visibly displayed "
                    "on the page. Users and some evaluators may miss this signal."
                ),
                suggested_value="Make the publication/update date visible to users.",
            ))

        return findings

    # ── External References ─────────────────────────────────────────

    def _check_references(
        self, soup: BeautifulSoup, url: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        body = soup.find("body")
        if not body:
            return findings

        # Count outbound links to different domains
        base_domain = urlparse(url).netloc
        external_links: list[str] = []
        for a in body.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and urlparse(href).netloc != base_domain:
                external_links.append(href)

        # Check for links to authoritative domains
        authoritative_domains = [
            "wikipedia.org", "scholar.google", "pubmed.ncbi.nlm.nih.gov",
            "who.int", "cdc.gov", "nih.gov", "mayoclinic.org", "webmd.com",
            "edu", ".gov", "arxiv.org", "ieee.org", "acm.org", "springer.com",
            "nature.com", "science.org", "nejm.org", "thelancet.com",
            "sec.gov", "irs.gov", "fdic.gov", "bloomberg.com", "reuters.com",
        ]
        authoritative_count = sum(
            1 for link in external_links
            if any(ad in urlparse(link).netloc.lower() for ad in authoritative_domains)
        )

        # For article-length content (>1000 words), check citations
        word_count = len(body.get_text(separator=" ", strip=True).split())
        if word_count > 1000:
            if len(external_links) < 2:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="eeat_no_references",
                    description=(
                        f"Long-form content ({word_count} words) has only "
                        f"{len(external_links)} external reference links. "
                        f"Citing authoritative sources strengthens E-E-A-T."
                    ),
                    current_value=str(len(external_links)),
                    suggested_value=(
                        "Cite and link to reputable external sources "
                        "(studies, official data, expert publications)."
                    ),
                    raw_metadata={"word_count": word_count, "external_links": len(external_links)},
                ))
            elif authoritative_count == 0:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="eeat_no_authoritative_refs",
                    description=(
                        f"Page has {len(external_links)} external links but "
                        f"none to recognized authoritative sources (.gov, .edu, "
                        f"peer-reviewed journals, etc.)."
                    ),
                    current_value=f"{len(external_links)} links, 0 authoritative",
                    suggested_value=(
                        "Prioritize linking to .gov, .edu, and peer-reviewed "
                        "sources when citing factual claims."
                    ),
                    raw_metadata={
                        "external_links": len(external_links),
                        "authoritative": authoritative_count,
                    },
                ))

        return findings

    # ── YMYL Content ────────────────────────────────────────────────

    def _check_ymyl(
        self, soup: BeautifulSoup, url: str, text: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []
        text_lower = text.lower()

        # Detect YMYL topics
        matched_categories: list[str] = []
        for pattern in YMYL_PATTERNS:
            matches = re.findall(pattern, text_lower[:5000])
            if len(matches) >= 3:  # strong signal
                # Classify into broad category
                if any(w in matches for w in ("medical", "health", "disease", "doctor")):
                    matched_categories.append("medical/health")
                elif any(w in matches for w in ("financ", "invest", "loan", "tax")):
                    matched_categories.append("financial")
                elif any(w in matches for w in ("legal", "law", "attorney")):
                    matched_categories.append("legal")
                elif any(w in matches for w in ("news", "journalism")):
                    matched_categories.append("news")
                break  # one category per page

        if not matched_categories:
            return findings

        # YMYL content: check for disclaimers
        has_disclaimer = bool(
            soup.find(string=re.compile(
                r"(?:disclaimer|not (?:medical|financial|legal) advice|"
                r"consult (?:a|your|with a) (?:doctor|physician|financial advisor|"
                r"lawyer|attorney)|for (?:informational|educational) purposes only)",
                re.IGNORECASE,
            ))
        )

        has_reviewed_by = bool(
            soup.find(string=re.compile(
                r"(?:Reviewed by|Medically reviewed|Fact.checked by|"
                r"Editorially reviewed|Expert review)",
                re.IGNORECASE,
            ))
        )

        cat_str = ", ".join(matched_categories)

        if not has_disclaimer:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_ymyl_no_disclaimer",
                description=(
                    f"Page covers {cat_str} topics (YMYL) but has no "
                    f"visible disclaimer. YMYL content should include "
                    f"appropriate disclaimers."
                ),
                suggested_value=(
                    f"Add a clear disclaimer: e.g., 'This article is for "
                    f"informational purposes only and does not constitute "
                    f"{cat_str} advice. Consult a qualified professional.'"
                ),
                raw_metadata={"ymyl_category": cat_str},
            ))

        if not has_reviewed_by:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="eeat_ymyl_no_reviewer",
                description=(
                    f"Page covers {cat_str} topics (YMYL) but has no "
                    f"'Reviewed by' or 'Fact-checked by' attribution. "
                    f"Expert review is a strong E-E-A-T signal."
                ),
                suggested_value=(
                    "Have content reviewed by a qualified expert and add "
                    "a visible 'Reviewed by [Name], [Credentials]' line."
                ),
                raw_metadata={"ymyl_category": cat_str},
            ))

        return findings

    # ── Cross-site Trust Coverage ───────────────────────────────────

    def _check_trust_coverage(self, url: str) -> list[RawFinding]:
        """Check whether the site has About, Contact, Privacy, Terms pages."""
        findings: list[RawFinding] = []

        # Only report once per scan (not per page) — check if this is the first page
        # We use a simple heuristic: report only from the root/homepage
        path = urlparse(url).path.rstrip("/")
        if path not in ("", "/"):
            return findings

        for page_type, found_url in self._trust_pages.items():
            if not found_url:
                label = page_type.capitalize()
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"eeat_no_{page_type}_page",
                    description=(
                        f"No '{label}' page detected in crawled URLs. "
                        f"A '{label}' page is an important trust signal "
                        f"for E-E-A-T evaluation."
                    ),
                    suggested_value=(
                        f"Create a /{page_type} page and link to it from "
                        f"the site footer or header."
                    ),
                ))

        return findings
