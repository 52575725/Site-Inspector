from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Thresholds
STALE_MONTHS = 18       # content older than 18 months without update = stale
VERY_STALE_MONTHS = 36  # 3+ years
WARN_MONTHS = 12        # approaching staleness

# Patterns to extract dates from visible HTML
VISIBLE_DATE_PATTERNS = [
    # "Published: June 15, 2024" or "Updated: 2024-06-15"
    re.compile(
        r"(?:Published|Posted|Written|Updated|Last\s+updated|Modified)[:\s]+"
        r"(?:on\s+)?"
        r"(\w+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        re.IGNORECASE,
    ),
    # ISO 8601 in <time> elements
    re.compile(
        r'<time[^>]*datetime\s*=\s*["\'](\d{4}-\d{2}-\d{2})[^"\']*["\']',
        re.IGNORECASE,
    ),
]

# Year references that date content (e.g., "in 2023 we launched...")
YEAR_REFERENCE_PATTERN = re.compile(
    r"\b((?:20)\d{2})\b"  # years 2000-2099
)

OUTDATED_PHRASE_PATTERN = re.compile(
    r"\b(?:last year|earlier this year|in \d{4}|recently launched|"
    r"newly released|just announced|coming soon in \d{4}|"
    r"this (?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December))\b",
    re.IGNORECASE,
)


class ContentFreshnessInspector(BaseInspector):
    """Inspect content freshness: stale content, outdated references,
    date consistency, and update recency.

    Freshness is a known ranking factor — Google favors current,
    recently-updated content for time-sensitive queries.
    """

    inspector_name = "content_freshness"

    def __init__(self) -> None:
        self._today = datetime.now(timezone.utc)

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

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract dates from multiple sources
        visible_date = self._extract_visible_date(soup, html_content)
        schema_dates = self._extract_schema_dates(soup)

        # ── 1. Visible date freshness ────────────────────────
        findings.extend(self._check_visible_date_freshness(url, visible_date))

        # ── 2. Schema date freshness ─────────────────────────
        findings.extend(self._check_schema_date_freshness(url, schema_dates))

        # ── 3. Date consistency ──────────────────────────────
        findings.extend(self._check_date_consistency(url, visible_date, schema_dates))

        # ── 4. Outdated content references ───────────────────
        findings.extend(self._check_outdated_references(url, soup, html_content))

        return findings

    # ── Date Extraction ─────────────────────────────────────────────

    def _extract_visible_date(
        self, soup: BeautifulSoup, html_content: str,
    ) -> Optional[datetime]:
        """Extract the most specific date visible on the page."""
        candidates: list[datetime] = []

        # Check meta tags first (most reliable)
        for meta in soup.find_all("meta"):
            prop = (meta.get("property") or "").lower()
            name = (meta.get("name") or "").lower()
            content = meta.get("content", "")
            if "published_time" in prop or "modified_time" in prop or \
               name in ("date", "pubdate", "dc.date", "dc.date.issued"):
                dt = self._parse_date(content)
                if dt:
                    candidates.append(dt)

        # Check visible text patterns
        text_block = soup.get_text(separator=" ", strip=True)[:3000]
        for pattern in VISIBLE_DATE_PATTERNS:
            match = pattern.search(text_block)
            if match:
                dt = self._parse_date(match.group(1))
                if dt:
                    candidates.append(dt)
            # Also check raw HTML for <time> elements
            match = pattern.search(html_content[:10000])
            if match:
                dt = self._parse_date(match.group(1))
                if dt:
                    candidates.append(dt)

        # Check JSON-LD dates
        schema_dates = self._extract_schema_dates(soup)
        if schema_dates.get("datePublished"):
            candidates.append(schema_dates["datePublished"])
        if schema_dates.get("dateModified"):
            candidates.append(schema_dates["dateModified"])

        if not candidates:
            return None

        # Return the most recent date (likely dateModified)
        return max(candidates)

    def _extract_schema_dates(self, soup: BeautifulSoup) -> dict[str, datetime]:
        """Extract datePublished and dateModified from JSON-LD."""
        result: dict[str, datetime] = {}
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                blocks = data if isinstance(data, list) else [data]
                for block in blocks:
                    if isinstance(block, dict):
                        for field in ("datePublished", "dateModified"):
                            if field in block and field not in result:
                                dt = self._parse_date(block[field])
                                if dt:
                                    result[field] = dt
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats into a UTC datetime."""
        if not date_str:
            return None

        date_str = date_str.strip()

        # ISO 8601: 2024-06-15 or 2024-06-15T10:30:00Z
        iso_match = re.match(
            r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2}))?",
            date_str,
        )
        if iso_match:
            try:
                return datetime(
                    int(iso_match.group(1)),
                    int(iso_match.group(2)),
                    int(iso_match.group(3)),
                    tzinfo=timezone.utc,
                )
            except ValueError:
                pass

        # US format: June 15, 2024 or Jun 15, 2024
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
                     "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # mm/dd/yyyy or dd/mm/yyyy
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    # ── Visible Date Freshness ──────────────────────────────────────

    def _check_visible_date_freshness(
        self, url: str, visible_date: Optional[datetime],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not visible_date:
            return findings

        age_months = self._months_since(visible_date)

        if age_months >= VERY_STALE_MONTHS:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="freshness_very_stale",
                description=(
                    f"Content has not been updated in {age_months} months "
                    f"(last update: {visible_date.strftime('%Y-%m-%d')}). "
                    f"Severely stale content may be devalued in search rankings, "
                    f"especially for time-sensitive topics."
                ),
                current_value=f"{age_months} months old",
                suggested_value=(
                    "Review and update the content with current information, "
                    "or add a prominent archival notice if intentionally preserved. "
                    "Update the dateModified in schema + visible 'Updated on' date."
                ),
                raw_metadata={
                    "age_months": age_months,
                    "last_date": visible_date.isoformat(),
                },
            ))
        elif age_months >= STALE_MONTHS:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="freshness_stale",
                description=(
                    f"Content is {age_months} months old "
                    f"(last update: {visible_date.strftime('%Y-%m-%d')}). "
                    f"Consider refreshing with up-to-date information."
                ),
                current_value=f"{age_months} months old",
                suggested_value=(
                    "Update statistics, examples, and references. "
                    "Refresh the 'Updated on' date after meaningful changes."
                ),
                raw_metadata={
                    "age_months": age_months,
                    "last_date": visible_date.isoformat(),
                },
            ))
        elif age_months >= WARN_MONTHS:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="freshness_aging",
                description=(
                    f"Content is approaching staleness ({age_months} months "
                    f"since last update). Schedule a content review."
                ),
                current_value=f"{age_months} months",
                suggested_value=(
                    "Plan a content refresh within the next few months. "
                    "Add a review schedule to the editorial calendar."
                ),
                raw_metadata={
                    "age_months": age_months,
                    "last_date": visible_date.isoformat(),
                },
            ))

        return findings

    # ── Schema Date Freshness ───────────────────────────────────────

    def _check_schema_date_freshness(
        self, url: str, schema_dates: dict[str, datetime],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Check if dateModified exists when datePublished is old
        pub_date = schema_dates.get("datePublished")
        mod_date = schema_dates.get("dateModified")

        if pub_date and not mod_date:
            pub_age = self._months_since(pub_date)
            if pub_age >= WARN_MONTHS:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="freshness_no_date_modified",
                    description=(
                        f"Article published {pub_age} months ago "
                        f"({pub_date.strftime('%Y-%m-%d')}) but has no "
                        f"dateModified in schema. Even if content is evergreen, "
                        f"Google may interpret the absence as staleness."
                    ),
                    current_value=f"datePublished={pub_date.strftime('%Y-%m-%d')}, no dateModified",
                    suggested_value=(
                        "If the content has been reviewed/updated, add "
                        "dateModified to the schema. For truly evergreen content, "
                        "set dateModified equal to datePublished."
                    ),
                    raw_metadata={
                        "age_months": pub_age,
                        "datePublished": pub_date.isoformat(),
                    },
                ))

        if mod_date:
            mod_age = self._months_since(mod_date)
            if mod_age >= STALE_MONTHS:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="freshness_schema_date_modified_old",
                    description=(
                        f"Schema dateModified is {mod_age} months old "
                        f"({mod_date.strftime('%Y-%m-%d')}). "
                        f"An old modified date signals stale content."
                    ),
                    current_value=f"dateModified={mod_date.strftime('%Y-%m-%d')}",
                    suggested_value="Review and update the content; set dateModified to the current date after meaningful changes.",
                    raw_metadata={
                        "age_months": mod_age,
                        "dateModified": mod_date.isoformat(),
                    },
                ))

        return findings

    # ── Date Consistency ────────────────────────────────────────────

    def _check_date_consistency(
        self, url: str, visible_date: Optional[datetime],
        schema_dates: dict[str, datetime],
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not visible_date:
            return findings

        # Check visible date vs schema datePublished
        pub_date = schema_dates.get("datePublished")
        if pub_date:
            delta = abs((visible_date - pub_date).days)
            if delta > 30:  # more than 30 days off
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="freshness_date_mismatch",
                    description=(
                        f"Date mismatch: visible date "
                        f"({visible_date.strftime('%Y-%m-%d')}) differs from "
                        f"schema datePublished "
                        f"({pub_date.strftime('%Y-%m-%d')}) by {delta} days. "
                        f"Inconsistent dates harm trust signals."
                    ),
                    current_value=(
                        f"visible={visible_date.strftime('%Y-%m-%d')}, "
                        f"schema={pub_date.strftime('%Y-%m-%d')}"
                    ),
                    suggested_value="Ensure visible and schema dates are consistent.",
                    raw_metadata={
                        "visible_date": visible_date.isoformat(),
                        "schema_date": pub_date.isoformat(),
                        "delta_days": delta,
                    },
                ))

        # Check visible date vs schema dateModified
        mod_date = schema_dates.get("dateModified")
        if mod_date:
            delta = abs((visible_date - mod_date).days)
            if delta > 30:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="freshness_modified_date_mismatch",
                    description=(
                        f"Visible 'Updated' date "
                        f"({visible_date.strftime('%Y-%m-%d')}) does not match "
                        f"schema dateModified "
                        f"({mod_date.strftime('%Y-%m-%d')}, {delta} days off)."
                    ),
                    current_value=(
                        f"visible={visible_date.strftime('%Y-%m-%d')}, "
                        f"schema dateModified={mod_date.strftime('%Y-%m-%d')}"
                    ),
                    suggested_value="Update the visible 'Updated on' date or the schema dateModified to match.",
                    raw_metadata={
                        "visible_date": visible_date.isoformat(),
                        "schema_mod_date": mod_date.isoformat(),
                        "delta_days": delta,
                    },
                ))

        return findings

    # ── Outdated References ─────────────────────────────────────────

    def _check_outdated_references(
        self, url: str, soup: BeautifulSoup, html_content: str,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        # Get the text content
        body = soup.find("body")
        if not body:
            return findings
        text = body.get_text(separator=" ", strip=True)[:5000]

        # Find year references that are stale
        current_year = self._today.year
        years = set(int(y) for y in YEAR_REFERENCE_PATTERN.findall(text))

        stale_years = [y for y in years if y < current_year - 2]
        if stale_years:
            stale_list = ", ".join(str(y) for y in sorted(stale_years)[:3])
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="freshness_outdated_year_refs",
                description=(
                    f"Content references outdated years: {stale_list}. "
                    f"Year-specific references (e.g., 'best of 2023') make "
                    f"content appear dated in {current_year}."
                ),
                current_value=f"References to: {stale_list}",
                suggested_value=(
                    f"Update year references to {current_year} or remove "
                    f"time-specific claims from evergreen content."
                ),
                raw_metadata={
                    "stale_years": stale_years,
                    "current_year": current_year,
                },
            ))

        # Check for "outdated" phrase patterns relative to page age
        # Look for "recently", "just announced", "coming soon in [past year]"
        outdated_phrases = OUTDATED_PHRASE_PATTERN.findall(text)
        if outdated_phrases:
            unique_phrases = list(set(outdated_phrases))[:3]
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="freshness_relative_time_refs",
                description=(
                    f"Content uses relative time references that may age poorly: "
                    f"{', '.join(unique_phrases)}. "
                    f"These phrases assume the reader is in the original "
                    f"publication timeframe."
                ),
                current_value=", ".join(unique_phrases),
                suggested_value=(
                    "Replace relative time references with absolute dates "
                    "(e.g., 'in March 2024' not 'last month')."
                ),
                raw_metadata={
                    "outdated_phrases": unique_phrases,
                },
            ))

        return findings

    # ── Utility ─────────────────────────────────────────────────────

    def _months_since(self, dt: datetime) -> int:
        """Calculate whole months between a date and today."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = self._today - dt
        return delta.days // 30
