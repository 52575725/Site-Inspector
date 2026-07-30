"""Small public web-search fallback chain for evidence discovery."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import urllib.request
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _unwrap_duckduckgo_url(href: str) -> str:
    absolute = urljoin("https://html.duckduckgo.com", href)
    parsed = urlparse(absolute)
    redirected = parse_qs(parsed.query).get("uddg")
    return unquote(redirected[0]) if redirected else absolute


def _parse_duckduckgo_html(html: str, query: str, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for link in soup.select("a.result__a[href]"):
        url = _unwrap_duckduckgo_url(link.get("href", ""))
        if not url.startswith(("http://", "https://")):
            continue
        result = link.find_parent(class_="result")
        snippet_node = result.select_one(".result__snippet") if result else None
        results.append({
            "query": query,
            "title": link.get_text(" ", strip=True)[:200],
            "url": url,
            "snippet": snippet_node.get_text(" ", strip=True)[:500] if snippet_node else "",
            "provider": "duckduckgo",
        })
        if len(results) >= limit:
            break
    return results


def _parse_bing_rss(xml_text: str, query: str, limit: int) -> list[dict]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    results = []
    for item in root.findall(".//item"):
        title = " ".join((item.findtext("title") or "").split())[:200]
        url = (item.findtext("link") or "").strip()
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append({
            "query": query,
            "title": title,
            "url": url,
            "snippet": " ".join((item.findtext("description") or "").split())[:500],
            "provider": "bing-rss",
        })
        if len(results) >= limit:
            break
    return results


def _parse_bing_html(html: str, query: str, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.select("li.b_algo"):
        link = item.select_one("h2 a[href]")
        if link is None:
            continue
        url = _unwrap_bing_url(link.get("href", ""))
        if not url.startswith(("http://", "https://")):
            continue
        snippet_node = item.select_one(".b_caption p") or item.find("p")
        results.append({
            "query": query,
            "title": link.get_text(" ", strip=True)[:200],
            "url": url,
            "snippet": snippet_node.get_text(" ", strip=True)[:500] if snippet_node else "",
            "provider": "bing",
        })
        if len(results) >= limit:
            break
    return results


def _unwrap_bing_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname not in {"bing.com", "www.bing.com"} or not parsed.path.startswith("/ck/"):
        return url
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return url
    payload = encoded[2:]
    try:
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return url
    return decoded if decoded.startswith(("http://", "https://")) else url


def _unwrap_yahoo_url(url: str) -> str:
    match = re.search(r"/RU=([^/]+)/RK=", url)
    if not match:
        return url
    decoded = unquote(match.group(1))
    return decoded if decoded.startswith(("http://", "https://")) else url


def _parse_yahoo_html(html: str, query: str, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.select("div.algo"):
        link = item.select_one(".compTitle a[href]")
        title_node = item.select_one("h3")
        if link is None or title_node is None:
            continue
        url = _unwrap_yahoo_url(link.get("href", ""))
        if not url.startswith(("http://", "https://")):
            continue
        snippet_node = item.select_one(".compText p")
        results.append({
            "query": query,
            "title": title_node.get_text(" ", strip=True)[:200],
            "url": url,
            "snippet": snippet_node.get_text(" ", strip=True)[:500] if snippet_node else "",
            "provider": "yahoo",
        })
        if len(results) >= limit:
            break
    return results


def _fetch_yahoo_html(query: str, timeout: float) -> str:
    url = "https://search.yahoo.com/search?" + urlencode({"p": query})
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(3_000_000).decode("utf-8", "replace")


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "best", "can", "do", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "safely", "the", "to", "what",
    "where", "which", "with",
}
_QUERY_CONTEXT_TERMS = {
    "best", "business", "buy", "company", "export", "find", "guide", "hong",
    "industrial", "japan", "kong", "latest", "list", "online", "supplier",
    "trend", "united", "use", "wholesale", "year",
}


def _filter_relevant(results: list[dict], query: str) -> list[dict]:
    terms = {
        token
        for token in re.findall(r"[a-z0-9]+", query.casefold())
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    }
    if not terms:
        return results
    required = 1 if len(terms) == 1 else 2
    specific_terms = terms - _QUERY_CONTEXT_TERMS
    relevant = []
    for result in results:
        primary_tokens = set(re.findall(
            r"[a-z0-9]+",
            f"{result['title']} {result['url']}".casefold(),
        ))
        all_tokens = primary_tokens | set(re.findall(
            r"[a-z0-9]+",
            result["snippet"].casefold(),
        ))
        if len(terms & all_tokens) < required:
            continue
        if specific_terms and not specific_terms & primary_tokens:
            continue
        relevant.append(result)
    return relevant


async def search_public_web(
    client: httpx.AsyncClient,
    query: str,
    *,
    semaphore: asyncio.Semaphore,
    timeout: float = 12,
    limit: int = 10,
) -> list[dict]:
    """Search public result pages, falling back when a provider is unavailable."""
    try:
        async with semaphore:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                timeout=timeout,
            )
        if response.status_code < 400:
            results = _filter_relevant(_parse_duckduckgo_html(response.text, query, limit), query)
            if results:
                return results
    except Exception as exc:
        logger.info("DuckDuckGo search unavailable for %r: %s", query, exc)

    try:
        async with semaphore:
            html = await asyncio.to_thread(_fetch_yahoo_html, query, max(timeout, 20))
        results = _filter_relevant(_parse_yahoo_html(html, query, limit), query)
        if results:
            return results
    except Exception as exc:
        logger.info("Yahoo search unavailable for %r: %s", query, exc)

    try:
        async with semaphore:
            response = await client.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss", "setlang": "en-us", "cc": "us"},
                timeout=max(timeout, 20),
            )
        if response.status_code < 400:
            results = _filter_relevant(_parse_bing_rss(response.text, query, limit), query)
            if results:
                return results
    except Exception as exc:
        logger.info("Bing RSS search unavailable for %r: %s", query, exc)

    try:
        async with semaphore:
            response = await client.get(
                "https://www.bing.com/search",
                params={"q": query, "setlang": "en-us", "cc": "us"},
                timeout=max(timeout, 20),
            )
        if response.status_code < 400:
            return _filter_relevant(_parse_bing_html(response.text, query, limit), query)
    except Exception as exc:
        logger.info("Bing HTML search unavailable for %r: %s", query, exc)
    return []


async def suggest_public_queries(
    client: httpx.AsyncClient,
    keyword: str,
    *,
    semaphore: asyncio.Semaphore,
    language: str = "en",
    limit: int = 12,
) -> list[str]:
    """Collect observed query formulations from Google autocomplete."""
    if language == "zh":
        seeds = [keyword, f"如何{keyword}", f"哪里可以买{keyword}"]
        locale = "zh-CN"
    else:
        seeds = [keyword, f"how to {keyword}", f"where to buy {keyword}"]
        locale = "en"

    async def fetch(seed: str) -> list[str]:
        try:
            async with semaphore:
                response = await client.get(
                    "https://suggestqueries.google.com/complete/search",
                    params={"client": "firefox", "q": seed, "hl": locale},
                    timeout=12,
                )
            if response.status_code >= 400:
                return []
            payload = response.json()
            if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
                return []
            return [" ".join(str(item).split())[:200] for item in payload[1] if str(item).strip()]
        except Exception as exc:
            logger.info("Autocomplete unavailable for %r: %s", seed, exc)
            return []

    groups = await asyncio.gather(*(fetch(seed) for seed in seeds))
    suggestions = []
    seen = set()
    for group in groups:
        for suggestion in group:
            key = suggestion.casefold()
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(suggestion)
            if len(suggestions) >= limit:
                return suggestions
    return suggestions
