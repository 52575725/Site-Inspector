from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.fixers.article_image_fixer import ArticleImageFixer
from src.integrations.image_generation import OpenAIImageGenerator
from src.integrations.image_search import (
    ImageResult,
    download_image,
    extract_keywords_from_html,
    search_images,
)
from src.integrations.image_webp import convert_to_webp
from src.sources.base import resolve_within
from src.web.deps import templates
from src.web.security import validate_public_http_url

router = APIRouter(tags=["article-images"])

SEARCH_TTL_SECONDS = 30 * 60
MAX_CANDIDATES = 16


class ImageSearchRequest(BaseModel):
    article_path: str = Field(min_length=1, max_length=500)
    target_count: int = Field(default=4, ge=3, le=4)


class ImageProposalRequest(BaseModel):
    search_id: str = Field(min_length=32, max_length=32)
    candidate_ids: list[str] = Field(default_factory=list, max_length=4)
    allow_ai_fallback: bool = False


def _source_root(settings) -> Path:
    target = settings.load_target(settings.target_name)
    source = target.get("source", {})
    if source.get("type") != "local" or not source.get("local_path"):
        raise HTTPException(
            status_code=409,
            detail="Article image workspace requires a configured local source repository",
        )
    root = Path(source["local_path"])
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if not root.is_dir():
        raise HTTPException(status_code=409, detail=f"Local source directory not found: {root}")
    return root


def _proposal_root(settings, proposal_id: str) -> Path:
    if len(proposal_id) != 32 or any(char not in "0123456789abcdef" for char in proposal_id):
        raise HTTPException(status_code=404, detail="Proposal not found")
    base = (settings.data_dir / "fixed" / "article-images").resolve()
    return resolve_within(base, proposal_id)


def _article_summary(path: Path, root: Path) -> dict | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    soup = BeautifulSoup(content, "html.parser")
    article = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=lambda value: value and any(
            token in " ".join(value if isinstance(value, list) else [value]).lower()
            for token in ("article", "post-content", "entry-content")
        ))
    )
    relative = path.relative_to(root).as_posix()
    if article is None and not any(token in relative.lower() for token in ("/blog/", "blog/", "/insights/")):
        return None
    article = article or soup.body or soup
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else path.stem
    return {
        "path": relative,
        "title": title,
        "word_count": len(article.get_text(" ", strip=True).split()),
        "image_count": len(article.find_all("img")),
        "sections": [tag.get_text(" ", strip=True) for tag in article.find_all("h2")[:8]],
    }


def _search_cache(request: Request) -> dict:
    cache = getattr(request.app.state, "article_image_searches", None)
    if cache is None:
        cache = {}
        request.app.state.article_image_searches = cache
    cutoff = time.monotonic() - SEARCH_TTL_SECONDS
    for key in [key for key, value in cache.items() if value["created_at"] < cutoff]:
        cache.pop(key, None)
    return cache


def _serialize_candidate(candidate_id: str, query: str, result: ImageResult) -> dict:
    return {
        "id": candidate_id,
        "query": query,
        "thumbnail_url": result.thumb_url or result.url,
        "source_page": result.page_url,
        "alt_text": result.alt_text,
        "photographer": result.photographer,
        "source": result.source,
        "license_name": result.license_name,
        "license_url": result.license_url,
        "width": result.width,
        "height": result.height,
    }


def _contextualize_queries(queries: list[str], article_title: str) -> list[str]:
    """Keep section-heading searches anchored to the article's visual subject."""
    title = article_title.lower()
    context = ""
    if "silver" in title:
        context = "silver"
    elif "gold" in title:
        context = "gold"
    if not context:
        return queries
    context_tokens = set(context.split())
    return [
        query if context_tokens & set(query.lower().split()) else f"{query} {context}"
        for query in queries
    ]


@router.get("/article-images")
async def article_images_page(request: Request):
    return templates.TemplateResponse(request, "article_images.html")


@router.get("/api/article-images/articles")
async def list_articles(request: Request):
    root = _source_root(request.app.state.settings)
    articles = []
    for path in root.rglob("*.html"):
        if "templates" in {part.lower() for part in path.relative_to(root).parts}:
            continue
        summary = _article_summary(path, root)
        if summary and summary["word_count"] >= ArticleImageFixer.MIN_WORDS:
            articles.append(summary)
    articles.sort(key=lambda item: item["path"])
    return {"root": str(root), "articles": articles[:250]}


@router.post("/api/article-images/search")
async def search_article_images(request: Request, body: ImageSearchRequest):
    settings = request.app.state.settings
    root = _source_root(settings)
    try:
        article_path = resolve_within(root, body.article_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if article_path.suffix.lower() not in {".html", ".htm"} or not article_path.is_file():
        raise HTTPException(status_code=404, detail="Article file not found")

    content = article_path.read_text(encoding="utf-8")
    summary = _article_summary(article_path, root)
    if summary is None:
        raise HTTPException(status_code=400, detail="The selected file is not an article")
    needed = max(0, body.target_count - summary["image_count"])
    if needed == 0:
        raise HTTPException(status_code=409, detail="This article already meets the image target")

    queries = _contextualize_queries(
        extract_keywords_from_html(content, max_queries=body.target_count),
        summary["title"],
    )
    if not queries:
        raise HTTPException(status_code=400, detail="No useful image search queries found")

    tasks = [
        asyncio.to_thread(
            search_images,
            query,
            4,
            settings.unsplash_api_key,
            settings.pexels_api_key,
            settings.pixabay_api_key,
        )
        for query in queries
    ]
    grouped = await asyncio.gather(*tasks)
    candidates: dict[str, tuple[str, ImageResult]] = {}
    seen_urls = set()
    round_number = 0
    while len(candidates) < MAX_CANDIDATES:
        added = False
        for query, results in zip(queries, grouped):
            if round_number >= len(results):
                continue
            result = results[round_number]
            key = result.url.split("?", 1)[0]
            if not result.url or key in seen_urls or not result.license_name:
                continue
            seen_urls.add(key)
            candidate_id = f"image-{len(candidates) + 1}"
            candidates[candidate_id] = (query, result)
            added = True
        if not added:
            break
        round_number += 1

    search_id = uuid.uuid4().hex
    _search_cache(request)[search_id] = {
        "created_at": time.monotonic(),
        "article_path": body.article_path,
        "content": content,
        "summary": summary,
        "target_count": body.target_count,
        "needed": needed,
        "queries": queries,
        "candidates": candidates,
    }
    return {
        "search_id": search_id,
        "article": summary,
        "target_count": body.target_count,
        "needed": needed,
        "queries": queries,
        "ai_fallback_available": bool(
            settings.image_generation_enabled and settings.openai_api_key
        ),
        "candidates": [
            _serialize_candidate(candidate_id, query, result)
            for candidate_id, (query, result) in candidates.items()
        ],
    }


@router.post("/api/article-images/proposals")
async def create_article_image_proposal(request: Request, body: ImageProposalRequest):
    settings = request.app.state.settings
    search = _search_cache(request).get(body.search_id)
    if search is None:
        raise HTTPException(status_code=404, detail="Search expired; search again")

    unique_ids = list(dict.fromkeys(body.candidate_ids))
    if len(unique_ids) > search["needed"]:
        raise HTTPException(status_code=400, detail="Too many images selected")
    selected = []
    for candidate_id in unique_ids:
        candidate = search["candidates"].get(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=400, detail=f"Unknown candidate: {candidate_id}")
        selected.append(candidate)

    can_use_ai = bool(
        body.allow_ai_fallback
        and settings.image_generation_enabled
        and settings.openai_api_key
    )
    if len(selected) < search["needed"] and not can_use_ai:
        raise HTTPException(
            status_code=400,
            detail=f"Select {search['needed']} images or enable the configured AI fallback",
        )

    proposal_id = uuid.uuid4().hex
    proposal_root = _proposal_root(settings, proposal_id)
    images_dir = proposal_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for index, (query, result) in enumerate(selected):
        await validate_public_http_url(result.url)
        filename = f"article-{proposal_id[:8]}-{index + 1}.jpg"
        local_path = await asyncio.to_thread(download_image, result.url, images_dir, filename)
        if not local_path:
            continue
        webp_path = await asyncio.to_thread(convert_to_webp, local_path, 82)
        asset_path = Path(webp_path or local_path)
        original_path = Path(local_path)
        if webp_path and asset_path != original_path:
            original_path.unlink(missing_ok=True)
        downloaded.append({
            "local_path": f"/images/{asset_path.name}",
            "alt_text": result.alt_text or query,
            "caption": result.alt_text or query,
            "width": result.width or 1200,
            "height": result.height or 800,
            "source": result.source,
            "photographer": result.photographer,
            "page_url": result.page_url,
            "license_name": result.license_name,
            "license_url": result.license_url,
        })

    generator = None
    if can_use_ai and len(downloaded) < search["needed"]:
        generator = OpenAIImageGenerator(
            settings.openai_api_key,
            settings.image_generation_model,
        )
    while generator and len(downloaded) < search["needed"]:
        index = len(downloaded)
        query = search["queries"][index % len(search["queries"])]
        filename = f"article-{proposal_id[:8]}-ai-{index + 1}.webp"
        generated_path = await generator.generate(
            ArticleImageFixer._build_generation_prompt(query),
            images_dir / filename,
        )
        if not generated_path:
            break
        downloaded.append({
            "local_path": f"/images/{generated_path.name}",
            "alt_text": query,
            "caption": query,
            "width": 1536,
            "height": 1024,
            "source": "ai-generated",
            "photographer": "",
            "page_url": "",
            "license_name": "AI generated",
            "license_url": "",
        })

    if len(downloaded) < search["needed"]:
        raise HTTPException(
            status_code=502,
            detail=f"Only {len(downloaded)} of {search['needed']} selected images were available",
        )

    soup = BeautifulSoup(search["content"], "html.parser")
    fixer = ArticleImageFixer(max_images=search["target_count"])
    inserted = fixer._insert_images(
        soup,
        downloaded,
        include_hero=search["summary"]["image_count"] == 0,
    )
    if inserted != search["needed"]:
        raise HTTPException(status_code=422, detail="Could not place every selected image")

    output_path = resolve_within(proposal_root, search["article_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(soup), encoding="utf-8")
    manifest = {
        "proposal_id": proposal_id,
        "status": "review_required",
        "article_path": search["article_path"],
        "output_path": str(output_path),
        "inserted": inserted,
        "target_count": search["target_count"],
        "images": downloaded,
        "search_candidates": [
            {"query": query, **asdict(result)} for query, result in selected
        ],
    }
    (proposal_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **manifest,
        "preview_url": f"/api/article-images/proposals/{proposal_id}/preview",
        "proposal_dir": str(proposal_root),
    }


@router.get("/api/article-images/proposals/{proposal_id}/preview")
async def preview_article_image_proposal(request: Request, proposal_id: str):
    root = _proposal_root(request.app.state.settings, proposal_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Proposal not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    html_path = resolve_within(root, manifest["article_path"])
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup.find_all(["script", "iframe", "object", "embed"]):
        tag.decompose()
    for tag in soup.find_all("link"):
        rel = {str(value).lower() for value in tag.get("rel", [])}
        if rel & {"stylesheet", "preload", "modulepreload"}:
            tag.decompose()
    for tag in soup.find_all(True):
        for attribute in [name for name in tag.attrs if name.lower().startswith("on")]:
            del tag.attrs[attribute]
    for image in soup.find_all("img"):
        src = image.get("src", "")
        if src.startswith("/images/"):
            filename = Path(src).name
            image["src"] = f"/api/article-images/proposals/{proposal_id}/assets/{filename}"
    return HTMLResponse(
        str(soup),
        headers={
            "Content-Security-Policy": (
                "default-src 'none'; img-src 'self' data:; "
                "style-src 'unsafe-inline'; font-src 'self'"
            )
        },
    )


@router.get("/api/article-images/proposals/{proposal_id}/assets/{filename}")
async def article_image_proposal_asset(request: Request, proposal_id: str, filename: str):
    root = _proposal_root(request.app.state.settings, proposal_id)
    try:
        path = resolve_within(root / "images", filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path.parent != (root / "images").resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)
