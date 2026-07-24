"""Auto-discover competitor websites from search engines.

Extracts core keywords from page data and searches DuckDuckGo to find
competing domains — no manual competitor URL configuration needed.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DUCKDUCKGO_LITE = "https://lite.duckduckgo.com/lite/"


async def discover_competitors(
    keywords: list[str],
    your_domain: str,
    max_competitors: int = 5,
    timeout: int = 15,
) -> list[str]:
    """Search for each keyword and extract unique competitor domains.

    Args:
        keywords: Search queries to use (e.g., ["silver trading hong kong"])
        your_domain: Your site's domain to exclude (e.g., "helinsilver.com")
        max_competitors: Max number of competitor domains to return
        timeout: HTTP request timeout in seconds

    Returns:
        List of competitor homepage URLs (deduplicated, sorted by frequency)
    """
    domain_counts: Counter[str] = Counter()
    your_domain_clean = your_domain.lower().replace("www.", "")

    async with httpx.AsyncClient(
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
        follow_redirects=False,
    ) as client:
        for keyword in keywords[:10]:  # limit to avoid rate-limiting
            try:
                resp = await client.post(
                    DUCKDUCKGO_LITE,
                    data={"q": keyword},
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Extract result links (DuckDuckGo Lite uses a simple table layout)
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    # DuckDuckGo Lite wraps real URLs in redirects like
                    # //duckduckgo.com/l/?uddg=https://example.com/...
                    if "uddg=" in href:
                        import urllib.parse
                        parsed = urllib.parse.urlparse(href)
                        qs = urllib.parse.parse_qs(parsed.query)
                        real_url = qs.get("uddg", [href])[0]
                    elif href.startswith("http"):
                        real_url = href
                    else:
                        continue

                    try:
                        domain = urlparse(real_url).netloc.lower().replace("www.", "")
                    except Exception:
                        continue

                    # Skip your own domain, empty domains, and common non-competitors
                    if not domain or domain == your_domain_clean:
                        continue
                    skip_patterns = (
                        "duckduckgo.com", "google.com", "bing.com",
                        "youtube.com", "facebook.com", "twitter.com",
                        "reddit.com", "wikipedia.org", "amazon.com",
                        "instagram.com", "linkedin.com",
                    )
                    if any(p in domain for p in skip_patterns):
                        continue
                    domain_counts[domain] += 1
            except Exception as e:
                logger.debug(f"Competitor search failed for '{keyword}': {e}")

    # Return top domains, sorted by frequency
    competitors = [
        f"https://{domain}"
        for domain, _ in domain_counts.most_common(max_competitors)
    ]
    if competitors:
        logger.info(
            f"Auto-discovered {len(competitors)} competitors: "
            f"{', '.join(competitors)}"
        )
    return competitors


def extract_keywords_from_pages(
    pages: list[dict],
    your_domain: str = "",
    gsc_queries: list[str] | None = None,
    max_keywords: int = 8,
) -> list[str]:
    """Extract the most important keywords to search for competitors.

    Combines page H1/title keywords with GSC top queries to build
    a representative set of search terms.

    Args:
        pages: List of {url, title, h1} dicts from crawled pages
        your_domain: Your domain (used to build branded queries)
        gsc_queries: Optional list of top GSC search queries
        max_keywords: Max number of search queries to return

    Returns:
        List of search query strings
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "silver", "hong", "kong", "international", "limited", "co", "ltd",
    }

    # Extract n-grams from page titles
    word_counter: Counter[str] = Counter()
    for page in pages:
        title = page.get("title", "")
        h1 = page.get("h1", "")
        combined = f"{title} {h1}".lower()
        # Extract meaningful words (4+ chars)
        words = re.findall(r"[a-z]{4,}", combined)
        significant = [w for w in words if w not in stopwords]
        # Single keywords
        for w in significant:
            word_counter[w] += 1
        # Bigrams
        for i in range(len(significant) - 1):
            word_counter[f"{significant[i]} {significant[i+1]}"] += 1

    # Build search queries from top keywords
    top_words = [w for w, _ in word_counter.most_common(12) if " " not in w]
    top_phrases = [w for w, _ in word_counter.most_common(8) if " " in w]

    queries: list[str] = []

    # Add phrase queries (most specific, best for finding competitors)
    for phrase in top_phrases[:4]:
        queries.append(phrase)

    # Build compound queries from top single words
    for i in range(0, len(top_words) - 1, 2):
        queries.append(f"{top_words[i]} {top_words[i+1]}")

    # Add GSC queries if available
    if gsc_queries:
        for q in gsc_queries[:5]:
            if q not in queries:
                queries.append(q)

    # Add location-specific queries for local business
    if your_domain:
        base_domain = your_domain.replace("www.", "").split(".")[0]
        queries.append(f"{base_domain} trading")

    # Deduplicate and cap
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_clean = q.strip().lower()
        if q_clean and q_clean not in seen and len(q_clean) > 5:
            seen.add(q_clean)
            unique.append(q_clean)
        if len(unique) >= max_keywords:
            break

    logger.info(f"Extracted {len(unique)} competitor search keywords: {unique}")
    return unique
