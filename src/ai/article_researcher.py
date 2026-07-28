"""Pre-generation research: find authoritative sources to cite in articles.

Before generating an article, searches configured authoritative sources for
relevant data, statistics, news, and context.  Feeds findings into the article
prompt so generated content includes real citations.

Two modes:
1. Live search (DuckDuckGo) — for real-time data from authoritative domains
2. Static source list — authoritative domains to cite when live search is unavailable

Source freshness: results are tagged with retrieval timestamp and stale after 24h.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ── Authoritative domains per topic category ──────────────────────────

AUTHORITY_SOURCES = {
    "silver": [
        {"domain": "lbma.org.uk", "label": "LBMA", "type": "pricing"},
        {"domain": "lme.com", "label": "LME", "type": "pricing"},
        {"domain": "kitco.com", "label": "Kitco", "type": "pricing"},
        {"domain": "reuters.com", "label": "Reuters", "type": "news"},
        {"domain": "bloomberg.com", "label": "Bloomberg", "type": "news"},
        {"domain": "worldgoldcouncil.org", "label": "World Gold Council", "type": "research"},
        {"domain": "silverinstitute.org", "label": "Silver Institute", "type": "research"},
        {"domain": "usgs.gov", "label": "USGS", "type": "government"},
        {"domain": "cmegroup.com", "label": "CME Group", "type": "pricing"},
        {"domain": "spglobal.com", "label": "S&P Global", "type": "research"},
        {"domain": "mining.com", "label": "Mining.com", "type": "news"},
        {"domain": "metalbulletin.com", "label": "Metal Bulletin", "type": "research"},
    ],
    "trade": [
        {"domain": "trade.gov", "label": "US Trade Data", "type": "government"},
        {"domain": "customs.gov.hk", "label": "HK Customs", "type": "government"},
        {"domain": "wto.org", "label": "WTO", "type": "government"},
        {"domain": "hktdc.com", "label": "HK TDC", "type": "trade_body"},
    ],
    "logistics": [
        {"domain": "customs.gov.hk", "label": "HK Customs", "type": "government"},
        {"domain": "lloydslist.com", "label": "Lloyd's List", "type": "news"},
        {"domain": "freightos.com", "label": "Freightos", "type": "data"},
    ],
    "finance": [
        {"domain": "reuters.com", "label": "Reuters", "type": "news"},
        {"domain": "bloomberg.com", "label": "Bloomberg", "type": "news"},
        {"domain": "investing.com", "label": "Investing.com", "type": "data"},
        {"domain": "tradingview.com", "label": "TradingView", "type": "data"},
    ],
}


@dataclass
class ResearchFinding:
    """One piece of research data found for citation."""
    title: str
    url: str
    snippet: str
    source_label: str
    source_type: str  # pricing, news, research, government, data
    found_at: str  # ISO timestamp
    relevance_score: float = 0.5

    @property
    def is_fresh(self) -> bool:
        """Findings older than 24h are considered stale."""
        try:
            found_dt = datetime.fromisoformat(self.found_at)
            if found_dt.tzinfo is None:
                found_dt = found_dt.replace(tzinfo=UTC)
            return datetime.now(UTC) - found_dt < timedelta(hours=24)
        except Exception:
            return False


@dataclass
class ResearchResult:
    """Complete research output for one article topic."""
    topic: str
    findings: list[ResearchFinding] = field(default_factory=list)
    searched_at: str = ""
    source_count: int = 0
    error: str = ""


async def research_topic(
    topic: str,
    keywords: list[str] | None = None,
    topic_area: str = "silver",
    max_sources: int = 5,
) -> ResearchResult:
    """Search authoritative sources for data to cite in an article.

    Args:
        topic: The article topic
        keywords: Extra search keywords
        topic_area: Which authority source list to use (silver/trade/logistics/finance)
        max_sources: Max number of sources to return

    Returns:
        ResearchResult with findings ready to inject into the article prompt
    """
    result = ResearchResult(
        topic=topic,
        searched_at=datetime.now(UTC).isoformat(),
    )

    sources = AUTHORITY_SOURCES.get(topic_area, AUTHORITY_SOURCES["silver"])
    keywords = keywords or []

    # Build search queries: topic + authority domain
    search_terms = " ".join([topic] + keywords[:3])

    findings: list[ResearchFinding] = []

    for source in sources[:max_sources]:
        try:
            # Search: "topic site:authority-domain.com"
            query = f"{search_terms} site:{source['domain']}"

            # Try DuckDuckGo instant answer API (no API key needed)
            finding = await _search_duckduckgo(query, source, topic)
            if finding:
                findings.append(finding)

            if len(findings) >= max_sources:
                break
        except Exception as e:
            logger.debug(f"Search failed for {source['domain']}: {e}")
            continue

    result.findings = findings
    result.source_count = len(findings)

    return result


async def _search_duckduckgo(
    query: str, source: dict, topic: str,
) -> Optional[ResearchFinding]:
    """Search DuckDuckGo for a specific authority source."""
    import httpx
    from urllib.parse import quote

    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SiteInspector/1.0)"},
            )
            resp.raise_for_status()
    except Exception:
        return None

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract search result snippets
    results = soup.find_all("div", class_="result")
    if not results:
        return None

    for r in results[:3]:
        link = r.find("a", class_="result__a")
        snippet_el = r.find("a", class_="result__snippet")
        if link and snippet_el:
            href = link.get("href", "")
            if source["domain"] in href:
                return ResearchFinding(
                    title=link.get_text(strip=True)[:200],
                    url=href[:500],
                    snippet=snippet_el.get_text(strip=True)[:500],
                    source_label=source["label"],
                    source_type=source["type"],
                    found_at=datetime.now(UTC).isoformat(),
                    relevance_score=0.7,
                )

    return None


def build_citation_prompt(findings: list[ResearchFinding]) -> str:
    """Convert research findings into a citation requirements prompt.

    This gets injected into the article generation prompt to require
    the AI to cite real sources with actual data.
    """
    if not findings:
        return ""

    lines = [
        "\n## RESEARCH FINDINGS — YOU MUST CITE THESE\n",
        "The following data was found from authoritative sources. "
        "You MUST incorporate these findings into the article with "
        "proper inline citations. Link descriptive source text directly to the "
        "exact source URL using HTML, for example: "
        "<a href=\"SOURCE_URL\" target=\"_blank\" rel=\"noopener noreferrer\">source title</a>.",
        "",
        "At the END of the article, add a 'Sources' section listing each source "
        "as a clickable HTML link.",
        "",
    ]

    for i, f in enumerate(findings, 1):
        freshness = "FRESH" if f.is_fresh else "MAY BE STALE"
        lines.append(
            f"Source [{i}] — {f.source_label} ({f.source_type}, {freshness}):\n"
            f"  Title: {f.title}\n"
            f"  URL: {f.url}\n"
            f"  Key data: {f.snippet}\n"
        )

    lines.append(
        "\nIMPORTANT: Only cite data that you can verify from the snippets above. "
        "Do NOT invent prices, dates, or statistics. If a source is marked MAY BE STALE, "
        "note in the text that the data is 'as of [date].' Never invent or modify a URL.\n"
    )

    return "\n".join(lines)


def get_static_citations(topic_area: str = "silver") -> str:
    """Fallback: provide static authoritative URLs to cite when live search fails.

    These are well-known, stable URLs that can be cited without live search.
    """
    static_sources = {
        "silver": [
            ("LBMA Precious Metal Prices", "https://www.lbma.org.uk/prices-and-data/precious-metal-prices"),
            ("LME Silver", "https://www.lme.com/Metals/Precious-metals/LME-Silver"),
            ("USGS Silver Statistics", "https://www.usgs.gov/centers/national-minerals-information-center/silver-statistics-and-information"),
        ],
        "trade": [
            ("US International Trade Administration", "https://www.trade.gov/"),
            ("HK Customs and Excise", "https://www.customs.gov.hk/"),
        ],
    }

    sources = static_sources.get(topic_area, static_sources["silver"])

    lines = ["\n## AUTHORITATIVE SOURCES TO CITE\n"]
    for i, (name, url) in enumerate(sources, 1):
        lines.append(f"[{i}] {name}: {url}")

    lines.append(
        "\nWhere relevant, cite these as clickable HTML links using their exact URLs. "
        "Do not invent additional sources or URLs.\n"
    )

    return "\n".join(lines)
