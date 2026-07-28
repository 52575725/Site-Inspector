"""Auto-search relevant images for blog articles via free image APIs.

Uses Unsplash → Pexels → Pixabay as a fallback chain.
All APIs have free tiers — no API key required for basic usage,
but keys improve rate limits and are supported via Settings.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

_WIKIMEDIA_REQUEST_LOCK = threading.Lock()
_WIKIMEDIA_LAST_REQUEST = 0.0
_WIKIMEDIA_MIN_INTERVAL_SECONDS = 0.8


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


_LOCATION_NAMES = (
    "Hong Kong", "Mainland China", "China", "Japan", "Tokyo", "Osaka",
    "Singapore", "United States", "United Kingdom", "New York", "London",
    "Europe", "European Union", "Asia", "Mexico", "Peru", "Chile",
    "Australia", "Canada", "India", "Switzerland", "Shanghai", "Shenzhen",
)

_VISUAL_SCENES = (
    (r"\b(?:air freight|air cargo|cargo aircraft|airport)\b", "air cargo airport"),
    (r"\b(?:sea freight|ocean freight|container ships?|seaport)\b", "container ship port"),
    (r"\b(?:customs clearance|customs inspection|border inspection)\b", "customs cargo inspection"),
    (r"\b(?:shipping logistics|freight forwarding|supply chain)\b", "cargo logistics"),
    (r"\b(?:warehouse|warehousing|secure storage|vault)\b", "warehouse storage"),
    (r"\b(?:silver min(?:e|es|ing)|underground min(?:e|es|ing)|open-pit min(?:e|es|ing))\b", "silver mining"),
    (r"\b(?:refinery|refining|smelter|smelting)\b", "silver refinery"),
    (r"\b(?:assay|laboratory testing|quality inspection|quality control)\b", "silver assay laboratory"),
    (r"\b(?:solar panels?|photovoltaic|solar farm)\b", "solar panels"),
    (r"\b(?:jewelry|jewellery|silver rings?|silver necklaces?)\b", "silver jewelry"),
    (r"\b(?:price chart|price history|market chart|trading chart)\b", "silver price chart"),
    (r"\b(?:futures exchange|commodity exchange|trading floor)\b", "commodities exchange trading"),
    (r"\b(?:factory|manufacturing plant|production line)\b", "industrial factory"),
    (r"\b(?:rail freight|freight train)\b", "freight train"),
    (r"\b(?:road freight|cargo truck|trucking)\b", "cargo truck"),
)


def _extract_visual_facets(text: str) -> list[str]:
    """Extract concrete, photographable places and scenes in source order."""
    locations = []
    for name in _LOCATION_NAMES:
        match = re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE)
        if match:
            locations.append((match.start(), name))

    # Also recognize proper place names attached to explicit geographic nouns.
    place_pattern = re.compile(
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+"
        r"(?:Airport|Harbou?r|Port|Terminal|Exchange|Mine|Refinery))\b"
    )
    locations.extend((match.start(), match.group(1)) for match in place_pattern.finditer(text))

    # Capture unlisted geographic names from natural route/location phrases.
    location_cue_pattern = re.compile(
        r"\b(?:in|from|to|via|through|across|near|at)\s+"
        r"((?:the\s+)?[A-Z][A-Za-z.'-]*(?:\s+(?:and|of|the|[A-Z][A-Za-z.'-]*)){0,2})"
    )
    ignored_locations = {
        "Conclusion", "Introduction", "Summary", "This", "That", "What", "Why",
    }
    for match in location_cue_pattern.finditer(text):
        location = match.group(1).strip()
        if location not in ignored_locations:
            locations.append((match.start(1), location))

    deduplicated_locations = []
    seen_locations = set()
    for position, location in sorted(locations, key=lambda item: item[0]):
        key = location.lower()
        if key not in seen_locations:
            seen_locations.add(key)
            deduplicated_locations.append((position, location))
    locations = deduplicated_locations
    locations.sort(key=lambda item: item[0])

    scenes = []
    for pattern, search_phrase in _VISUAL_SCENES:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            scenes.append((match.start(), search_phrase))
    scenes.sort(key=lambda item: item[0])

    facets: list[str] = []
    used_locations = set()
    for position, scene in scenes:
        nearby = [item for item in locations if abs(item[0] - position) <= 500]
        if nearby:
            _, location = min(nearby, key=lambda item: abs(item[0] - position))
            facet = f"{location} {scene}"
            used_locations.add(location.lower())
        else:
            facet = scene
        if facet.lower() not in {value.lower() for value in facets}:
            facets.append(facet)

    for _, location in locations:
        if location.lower() not in used_locations:
            facets.append(location)
    return facets


def extract_keywords_from_html(html_content: str, max_queries: int = 3) -> list[str]:
    """Extract search queries from article HTML content.

    Prioritizes concrete places, transport modes, and photographable scenes,
    then falls back to title and headings.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    queries: list[str] = []

    article = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in article.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    article_text = article.get_text(separator=" ", strip=True)

    # Concrete article details produce more varied images than editorial titles.
    queries.extend(_extract_visual_facets(article_text))

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
            if title.lower() not in {query.lower() for query in queries}:
                queries.append(title)

    # 2. H1
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(strip=True)
        if (
            h1_text
            and len(h1_text) > 10
            and h1_text.lower() not in {query.lower() for query in queries}
        ):
            queries.append(h1_text)

    # 3. Use section headings so each image can match a different part of the article.
    h2_tags = soup.find_all("h2")
    for h2 in h2_tags:
        h2_text = h2.get_text(strip=True)
        if h2_text and len(h2_text) > 5:
            # Use H2 as a standalone query if distinct enough
            if h2_text.lower() not in " ".join(queries).lower():
                queries.append(h2_text)
            if len(queries) >= max_queries:
                break

    # Truncate to max
    result = queries[:max_queries]

    # If nothing found, try to extract from body text
    if not result:
        if article_text:
            # Take first meaningful sentence as query
            words = article_text.split()[:15]
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
    for search_query in query_variants[:3]:
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
        data = json.loads(_read_wikimedia_response(req).decode("utf-8"))

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
            visual_intent = _visual_query_for(search_query) or search_query
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


def _read_wikimedia_response(request) -> bytes:
    """Serialize Commons requests and retry a single rate-limited response."""
    import urllib.error
    import urllib.request

    global _WIKIMEDIA_LAST_REQUEST
    with _WIKIMEDIA_REQUEST_LOCK:
        elapsed = time.monotonic() - _WIKIMEDIA_LAST_REQUEST
        if elapsed < _WIKIMEDIA_MIN_INTERVAL_SECONDS:
            time.sleep(_WIKIMEDIA_MIN_INTERVAL_SECONDS - elapsed)
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = response.read()
                _WIKIMEDIA_LAST_REQUEST = time.monotonic()
                return payload
            except urllib.error.HTTPError as exc:
                _WIKIMEDIA_LAST_REQUEST = time.monotonic()
                if exc.code != 429 or attempt == 1:
                    raise
                retry_after = exc.headers.get("Retry-After", "3")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 3.0
                time.sleep(min(max(delay, 1.0), 10.0))
    return b"{}"


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
    # Preserve the full article intent before trying broader visual phrases.
    # This avoids filling the result set with generic stock imagery too early.
    variants = [query, visual_query]
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
    location = next(
        (name for name in _LOCATION_NAMES if re.search(
            rf"\b{re.escape(name)}\b", query, flags=re.IGNORECASE
        )),
        "",
    )
    explicit_place = re.search(
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+"
        r"(?:Airport|Harbou?r|Port|Terminal|Exchange|Mine|Refinery))\b",
        query,
    )
    location = location or (explicit_place.group(1) if explicit_place else "")

    visual_subject = ""
    if any(term in lowered for term in ("solar", "photovoltaic")):
        visual_subject = "solar panels industry"
    elif any(term in lowered for term in ("air freight", "air cargo", "airport")):
        visual_subject = "cargo aircraft freight"
    elif any(term in lowered for term in ("sea freight", "ocean freight", "container ship")):
        visual_subject = "container ship cargo"
    elif any(term in lowered for term in ("customs", "import", "export", "tariff")):
        visual_subject = "customs cargo inspection"
    elif any(term in lowered for term in ("shipping", "logistics", "freight")):
        visual_subject = "international cargo freight"
    elif any(term in lowered for term in ("mine", "mining", "geological", "origin")):
        visual_subject = "silver mining"
    elif any(term in lowered for term in ("jewelry", "jewellery", "ring", "necklace")):
        visual_subject = "silver jewelry"
    elif any(term in lowered for term in (
        "price", "pricing", "market", "benchmark", "futures", "inventory", "etf",
    )):
        visual_subject = "silver price chart"
    elif any(term in lowered for term in (
        "lbma", "bullion", "ingot", "silver bar", "assay", "refinery", "purity",
    )):
        visual_subject = "silver bullion ingot"
    elif location:
        return location
    elif "silver" in lowered:
        visual_subject = "silver bullion ingot"

    if location and visual_subject and location.lower() not in visual_subject.lower():
        return f"{location} {visual_subject}"
    return visual_subject


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
