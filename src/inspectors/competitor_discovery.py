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

DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"


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
                    DUCKDUCKGO_HTML,
                    data={"q": keyword},
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Extract result links (DuckDuckGo HTML uses class='result__a')
                for link in soup.find_all("a", class_="result__a", href=True):
                    href = link["href"]
                    # The HTML endpoint returns direct URLs (not redirects)
                    if href.startswith("http"):
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
    # Brand-specific words to exclude from competitor search queries
    brand_words = {
        "helin", "helinsilver", "changjiang", "helinsilvercom",
    }
    # Generic stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "limited", "co", "ltd", "hong", "kong", "international",
        "this", "that", "these", "those", "has", "have", "its",
    }

    # Dynamically extract brand words from the site's own domain.
    # Only domain name parts are unambiguous brand identifiers.
    # Title/site-name words may also be topic keywords (e.g., "Silver" in
    # "Helin Silver" is both a brand component and a topic).
    brand_words: set[str] = set()
    for page in pages:
        url = page.get("url", "")
        if not url:
            continue
        from urllib.parse import urlparse
        domain_name = urlparse(url).netloc.lower().replace("www.", "").split(".")[0]
        # Use only the full domain name as a brand token.
        # Do NOT split into substrings — "silver" in "helinsilver" is a topic word.
        brand_words.add(domain_name)

    # Extract n-grams from page titles
    word_counter: Counter[str] = Counter()
    for page in pages:
        title = page.get("title", "")
        h1 = page.get("h1", "")
        combined = f"{title} {h1}".lower()
        words = re.findall(r"[a-z]{4,}", combined)
        significant = [w for w in words if w not in stopwords and w not in brand_words]
        # Single keywords
        for w in significant:
            word_counter[w] += 1
        # Bigrams
        for i in range(len(significant) - 1):
            word_counter[f"{significant[i]} {significant[i+1]}"] += 1

    # Build search queries from top keywords, filtering out brand-specific ones
    top_words = [w for w, _ in word_counter.most_common(12) if " " not in w]
    top_phrases = [w for w, _ in word_counter.most_common(8) if " " in w]

    queries: list[str] = []

    # Filter: keep only phrases that don't contain brand words
    for phrase in top_phrases:
        if not any(bw in phrase for bw in brand_words):
            queries.append(phrase)

    # Build compound queries from top non-brand single words
    topic_words = [w for w in top_words if w not in brand_words]
    for i in range(0, len(topic_words) - 1, 2):
        q = f"{topic_words[i]} {topic_words[i+1]}"
        if not any(bw in q for bw in brand_words):
            queries.append(q)

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
