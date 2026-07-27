"""Auto-search relevant images for blog articles via free image APIs.

Uses Unsplash → Pexels → Pixabay as a fallback chain.
All APIs have free tiers — no API key required for basic usage,
but keys improve rate limits and are supported via Settings.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """A single image search result."""
    url: str           # Full-size image URL
    thumb_url: str     # Thumbnail/preview URL
    alt_text: str      # Suggested alt text
    photographer: str  # Attribution
    source: str        # "unsplash", "pexels", "pixabay"
    width: int = 0
    height: int = 0
    page_url: str = ""
    license_name: str = ""
    license_url: str = ""


# ── Public API ──────────────────────────────────────────────────────────

def search_images(
    query: str,
    count: int = 3,
    unsplash_key: str = "",
    pexels_key: str = "",
    pixabay_key: str = "",
) -> list[ImageResult]:
    """Search for images across multiple free APIs with fallback chaining.

    Args:
        query: Search query (article topic / keywords).
        count: Number of images to return (max 10).
        unsplash_key: Optional Unsplash API access key.
        pexels_key: Optional Pexels API key.
        pixabay_key: Optional Pixabay API key.

    Returns:
        List of ImageResult objects (may be empty if all sources fail).
    """
    results: list[ImageResult] = []

    # Use official providers only. Random placeholder services are not search.
    try:
        unsplash = _search_unsplash(query, count, unsplash_key)
        results.extend(unsplash)
        logger.info(f"Unsplash: {len(unsplash)} results for '{query}'")
    except Exception as e:
        logger.debug(f"Unsplash failed: {e}")

    if len(results) >= count:
        return _deduplicate(results)[:count]

    # Pexels fallback
    remaining = count - len(results)
    try:
        pexels = _search_pexels(query, remaining, pexels_key)
        results.extend(pexels)
        logger.info(f"Pexels: {len(pexels)} results for '{query}'")
    except Exception as e:
        logger.debug(f"Pexels failed: {e}")

    if len(results) >= count:
        return _deduplicate(results)[:count]

    # Pixabay fallback
    remaining = count - len(results)
    try:
        pixabay = _search_pixabay(query, remaining, pixabay_key)
        results.extend(pixabay)
        logger.info(f"Pixabay: {len(pixabay)} results for '{query}'")
    except Exception as e:
        logger.debug(f"Pixabay failed: {e}")

    if len(results) < count:
        remaining = count - len(results)
        try:
            commons = _search_wikimedia(query, remaining)
            results.extend(commons)
            logger.info(f"Wikimedia Commons: {len(commons)} results for '{query}'")
        except Exception as e:
            logger.debug(f"Wikimedia Commons failed: {e}")

    return _deduplicate(results)[:count]


def _deduplicate(results: list[ImageResult]) -> list[ImageResult]:
    unique = []
    seen = set()
    for result in results:
        key = result.url.split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def download_image(url: str, dest_dir: str | Path, filename: str | None = None) -> str | None:
    """Download an image to local disk. Returns the local file path or None."""
    import urllib.request
    import urllib.error

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # Generate filename from URL hash + extension
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            ext = ".jpg"
        filename = f"article-{url_hash}{ext}"

    filepath = dest / filename

    # Don't re-download
    if filepath.exists():
        logger.debug(f"Image already exists: {filepath}")
        return str(filepath)

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "SiteInspector/1.0 Image Downloader",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        filepath.write_bytes(data)
        size_kb = len(data) // 1024
        logger.info(f"Downloaded: {filepath.name} ({size_kb}KB) from {url[:80]}")
        return str(filepath)
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP {e.code} downloading {url[:80]}")
        return None
    except Exception as e:
        logger.warning(f"Failed to download {url[:80]}: {e}")
        return None


def extract_keywords_from_html(html_content: str, max_queries: int = 3) -> list[str]:
    """Extract search queries from article HTML content.

    Uses title + headings. Returns a list of keyword phrases for image search.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    queries: list[str] = []

    # 1. Title as primary query
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()
        # Remove site name suffix (after | or -)
        for sep in (" | ", " - ", " — ", " – "):
            if sep in title:
                title = title.split(sep)[0].strip()
                break
        if len(title) > 5:
            queries.append(title)

    # 2. H1
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(strip=True)
        if h1_text and len(h1_text) > 10 and h1_text not in queries:
            queries.append(h1_text)

    # 3. Use section headings so each image can match a different part of the article.
    h2_tags = soup.find_all("h2")
    for h2 in h2_tags:
        h2_text = h2.get_text(strip=True)
        if h2_text and len(h2_text) > 5:
            # Use H2 as a standalone query if distinct enough
            if h2_text not in " ".join(queries):
                queries.append(h2_text)
            if len(queries) >= max_queries:
                break

    # Truncate to max
    result = queries[:max_queries]

    # If nothing found, try to extract from body text
    if not result:
        body = soup.find("body")
        if body:
            for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = body.get_text(separator=" ", strip=True)
            # Take first meaningful sentence as query
            words = text.split()[:15]
            if words:
                result.append(" ".join(words))

    logger.debug(f"Extracted keywords: {result}")
    return result


# ── Private: API-specific implementations ───────────────────────────────

def _search_unsplash(query: str, count: int, api_key: str = "") -> list[ImageResult]:
    """Search Unsplash. With API key uses official API; without uses source.unsplash.com."""
    import json
    import urllib.request
    import urllib.error

    if api_key:
        url = (
            f"https://api.unsplash.com/search/photos"
            f"?query={quote(query)}&per_page={min(count, 30)}&orientation=landscape"
        )
        req = urllib.request.Request(url, headers={
            "Authorization": f"Client-ID {api_key}",
            "Accept-Version": "v1",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results: list[ImageResult] = []
        for photo in data.get("results", []):
            results.append(ImageResult(
                url=photo["urls"]["regular"],
                thumb_url=photo["urls"]["thumb"],
                alt_text=photo.get("alt_description") or photo.get("description") or query,
                photographer=photo["user"]["name"],
                source="unsplash",
                width=photo.get("width", 0),
                height=photo.get("height", 0),
                page_url=photo.get("links", {}).get("html", ""),
                license_name="Unsplash License",
                license_url="https://unsplash.com/license",
            ))
        return results
    return []


def _search_pexels(query: str, count: int, api_key: str = "") -> list[ImageResult]:
    """Search Pexels API. Requires API key (free tier: 200 req/hr)."""
    import json
    import urllib.request
    import urllib.error

    if not api_key:
        logger.debug("Pexels: no API key, skipping")
        return []

    url = (
        f"https://api.pexels.com/v1/search"
        f"?query={quote(query)}&per_page={min(count, 30)}&orientation=landscape"
    )
    req = urllib.request.Request(url, headers={
        "Authorization": api_key,
        "User-Agent": "SiteInspector/1.0",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    results: list[ImageResult] = []
    for photo in data.get("photos", []):
        results.append(ImageResult(
            url=photo["src"]["large"],
            thumb_url=photo["src"]["small"],
            alt_text=photo.get("alt") or query,
            photographer=photo["photographer"],
            source="pexels",
            width=photo.get("width", 0),
            height=photo.get("height", 0),
            page_url=photo.get("url", ""),
            license_name="Pexels License",
            license_url="https://www.pexels.com/license/",
        ))
    return results


def _search_wikimedia(query: str, count: int) -> list[ImageResult]:
    """Search Wikimedia Commons for attributable, reusable images."""
    import html
    import json
    import re
    import urllib.request

    results = []
    seen_urls = set()
    query_variants = _wikimedia_query_variants(query)
    visual_intent = query_variants[0]
    for search_query in query_variants:
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
            f"&gsrsearch={quote('file:' + search_query)}&gsrnamespace=6"
            f"&gsrlimit={min(max(count * 6, 12), 40)}"
            "&prop=imageinfo&iiprop=url%7Cmime%7Cextmetadata&iiurlwidth=1200"
            "&format=json&origin=*"
        )
        req = urllib.request.Request(url, headers={
            "User-Agent": "SiteInspector/1.0 (article image research)",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            image_url = info.get("thumburl") or info.get("url", "")
            key = image_url.split("?", 1)[0]
            if not image_url or key in seen_urls:
                continue
            metadata = info.get("extmetadata") or {}

            def meta(name: str) -> str:
                value = metadata.get(name, {}).get("value", "")
                value = re.sub(r"<[^>]+>", " ", value)
                return " ".join(html.unescape(value).split())

            title = page.get("title", "").removeprefix("File:")
            description = meta("ImageDescription") or title or search_query
            categories = meta("Categories")
            if not _matches_visual_intent(visual_intent, title, description, categories):
                continue
            license_name = meta("LicenseShortName") or meta("UsageTerms")
            if not license_name:
                continue
            seen_urls.add(key)
            results.append(ImageResult(
                url=image_url,
                thumb_url=image_url,
                alt_text=description[:180],
                photographer=meta("Artist") or "Wikimedia Commons contributor",
                source="wikimedia",
                width=info.get("thumbwidth", 0),
                height=info.get("thumbheight", 0),
                page_url=f"https://commons.wikimedia.org/?curid={page.get('pageid', '')}",
                license_name=license_name,
                license_url=metadata.get("LicenseUrl", {}).get("value", ""),
            ))
            if len(results) >= count:
                return results
    return results


def _wikimedia_query_variants(query: str) -> list[str]:
    """Create progressively broader visual queries for long editorial headings."""
    words = re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+", query)
    if not words:
        return [query]
    stopwords = {
        "a", "an", "and", "are", "as", "at", "complete", "conclusion",
        "delivery", "explained", "for", "from", "good", "guide", "how",
        "in", "introduction", "is", "list", "of", "on", "quality", "standards",
        "specifications", "the", "to", "understanding", "what", "why", "with",
    }
    compact = [word for word in words if word.lower() not in stopwords and not word.isdigit()]
    without_acronyms = [word for word in compact if not (word.isupper() and len(word) > 1)]
    visual_query = _visual_query_for(query)
    variants = [visual_query, query]
    if compact:
        variants.append(" ".join(compact[:5]))
    if without_acronyms:
        variants.append(" ".join(without_acronyms[:4]))
        if len(without_acronyms) > 2:
            variants.append(" ".join(without_acronyms[-2:]))
    return list(dict.fromkeys(value.strip() for value in variants if value.strip()))


def _visual_query_for(query: str) -> str:
    """Map editorial language to a concrete subject that can be photographed."""
    lowered = query.lower()
    if any(term in lowered for term in ("solar", "photovoltaic")):
        return "solar panels industry"
    if any(term in lowered for term in ("air freight", "air cargo", "airport")):
        return "cargo aircraft freight"
    if any(term in lowered for term in ("sea freight", "ocean freight", "container ship")):
        return "container ship cargo"
    if any(term in lowered for term in ("customs", "import", "export", "tariff")):
        return "customs cargo inspection"
    if any(term in lowered for term in ("shipping", "logistics", "freight")):
        return "international cargo freight"
    if any(term in lowered for term in ("mine", "mining", "geological", "origin")):
        return "silver mining"
    if any(term in lowered for term in ("jewelry", "jewellery", "ring", "necklace")):
        return "silver jewelry"
    if any(term in lowered for term in (
        "price", "pricing", "market", "benchmark", "futures", "inventory", "etf",
    )):
        return "silver price chart"
    if any(term in lowered for term in (
        "lbma", "bullion", "ingot", "silver bar", "assay", "refinery", "purity",
    )):
        return "silver bullion ingot"
    if "silver" in lowered:
        return "silver bullion ingot"
    return ""


def _matches_visual_intent(
    search_query: str,
    title: str,
    description: str,
    categories: str,
) -> bool:
    """Reject obvious homonyms before an image reaches editorial review."""
    query = search_query.lower()
    candidate = f"{title} {description} {categories}".lower()
    if "silver" in query and "silver" not in candidate:
        return False
    if any(term in query for term in ("bullion", "ingot")):
        return any(term in candidate for term in ("bullion", "ingot"))
    intent_groups = (
        (("solar", "panel"), ("solar", "photovoltaic", "panel")),
        (("aircraft",), ("aircraft", "airplane", "aeroplane", "air cargo")),
        (("container", "ship"), ("container", "cargo ship", "container ship")),
        (("customs",), ("customs", "cargo", "freight")),
        (("cargo", "freight"), ("cargo", "freight", "logistics")),
        (("mining",), ("mine", "mining", "ore")),
        (("jewelry",), ("jewelry", "jewellery", "ring", "necklace")),
        (("price", "chart"), ("price", "chart", "market graph")),
    )
    for query_terms, candidate_terms in intent_groups:
        if any(term in query for term in query_terms):
            return any(term in candidate for term in candidate_terms)
    return True


def _search_pixabay(query: str, count: int, api_key: str = "") -> list[ImageResult]:
    """Search Pixabay API. Requires API key (free tier: 100 req/min)."""
    import json
    import urllib.request
    import urllib.error

    if not api_key:
        logger.debug("Pixabay: no API key, skipping")
        return []

    url = (
        f"https://pixabay.com/api/"
        f"?key={api_key}&q={quote(query)}&per_page={min(count, 30)}"
        f"&orientation=horizontal&safesearch=true"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    results: list[ImageResult] = []
    for hit in data.get("hits", []):
        results.append(ImageResult(
            url=hit["largeImageURL"],
            thumb_url=hit["previewURL"],
            alt_text=hit.get("tags", query),
            photographer=hit.get("user", "Pixabay"),
            source="pixabay",
            width=hit.get("imageWidth", 0),
            height=hit.get("imageHeight", 0),
            page_url=hit.get("pageURL", ""),
            license_name="Pixabay Content License",
            license_url="https://pixabay.com/service/license-summary/",
        ))
    return results
