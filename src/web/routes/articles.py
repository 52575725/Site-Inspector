from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.web.deps import get_db, templates
from src.sources.base import resolve_within
from src.web.security import validate_github_repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["articles"])

GENERATED_DIR = Path("data/generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR = GENERATED_DIR / "research-plans"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
ARTICLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def _article_orchestrator(settings):
    from src.agents.article_orchestrator import ArticleOrchestratorAgent

    return ArticleOrchestratorAgent(settings.data_dir / "article-agent-runs")


class GenerateRequest(BaseModel):
    topic: str
    keywords: str = ""
    language: str = "en"
    word_count: int = 800
    page_type: str = "auto"  # AI selects unless a supported type is requested
    with_research: bool = True  # search authoritative sources for citations
    topic_area: str = "silver"  # silver, trade, logistics, finance


class AutoGenerateRequest(BaseModel):
    website_url: str = Field(min_length=4, max_length=2048)
    topic: str = Field(default="", max_length=300)
    keywords: str = Field(default="", max_length=1000)
    language: str = "auto"
    word_count: int = Field(default=1200, ge=300, le=3000)
    page_type: str = "auto"
    content_direction: str = Field(default="auto", max_length=40)
    max_reference_articles: int = Field(default=5, ge=1, le=8)


class GenerateFromResearchRequest(BaseModel):
    research_id: str = Field(min_length=32, max_length=32)
    headline: str = Field(default="", max_length=300)
    word_count: int | None = Field(default=None, ge=300, le=5000)
    page_type: str = Field(default="", max_length=40)
    content_direction: str = Field(default="", max_length=40)
    outline: list[str] = Field(default_factory=list, max_length=12)
    target_languages: list[str] | None = Field(default=None, max_length=6)
    auto_translate: bool = True


class TranslateRequest(BaseModel):
    source_path: str  # e.g., "blog/silver-market-q3-2026/index.html"
    target_path: str  # e.g., "jp/blog/silver-market-q3-2026/index.html"
    source_lang: str = "en"
    target_lang: str = "ja"


class BatchGenerateRequest(BaseModel):
    topics: list[str]  # list of topic strings
    keywords: str = ""
    language: str = "en"
    word_count: int = 800
    page_type: str = "auto"


class PushRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    file_path: str = ""  # target path in repo, e.g. "blog/new-article/index.html"


@router.post("/api/articles/batch-generate")
async def batch_generate_articles(request: Request, body: BatchGenerateRequest):
    """Generate multiple articles from a list of topics."""
    settings = request.app.state.settings
    api_key = settings.deepseek_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    results = []
    for topic in body.topics:
        try:
            page_type = await _resolve_article_type(
                settings,
                topic=topic.strip(),
                keywords=body.keywords,
                language=body.language,
                requested=body.page_type,
            )
            prompt = _build_article_prompt(
                topic=topic.strip(),
                keywords=body.keywords,
                language=body.language,
                word_count=body.word_count,
                page_type=page_type,
            )
            import httpx
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": settings.deepseek_model,
                        "messages": [
                            {"role": "system", "content": "You are an expert SEO content writer. Output valid HTML only."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7, "max_tokens": 4096,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]

            html_content = _sanitize_generated_html(_extract_html(raw))
            title = _extract_title(html_content) or topic.strip()
            article_id = __import__("uuid").uuid4().hex[:12]
            article_data = {
                "id": article_id, "topic": topic.strip(), "keywords": body.keywords,
                "language": body.language, "page_type": page_type,
                "title": title, "html": html_content,
                "created_at": __import__("datetime").datetime.utcnow().isoformat(),
                "pushed": False,
            }
            (GENERATED_DIR / f"{article_id}.json").write_text(
                __import__("json").dumps(article_data, ensure_ascii=False, indent=2), encoding="utf-8")
            (GENERATED_DIR / f"{article_id}.html").write_text(html_content, encoding="utf-8")
            results.append({
                "topic": topic.strip(), "id": article_id, "title": title,
                "page_type": page_type, "status": "ok",
            })
        except Exception as e:
            logger.error(f"Batch generate failed for '{topic}': {e}")
            results.append({"topic": topic.strip(), "status": "failed", "error": str(e)[:200]})

    return {"total": len(body.topics), "generated": sum(1 for r in results if r["status"] == "ok"), "results": results}


# ── Translation helpers ──────────────────────────────────────────────

LANG_NAMES = {
    "en": "English", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
    "ar": "Arabic", "vi": "Vietnamese", "th": "Thai",
}


async def _do_translate(
    settings, html_content: str, source_lang: str, target_lang: str,
    lang_names: dict, label: str = "",
) -> str:
    """Call DeepSeek API to translate an HTML fragment.

    Returns the translated HTML string.
    """
    import httpx
    src_name = lang_names.get(source_lang, source_lang.upper())
    tgt_name = lang_names.get(target_lang, target_lang.upper())

    prompt = f"""Translate the following HTML from {src_name} to {tgt_name}.
CRITICAL RULES:
- Keep ALL HTML tags, attributes, and structure exactly as-is
- Translate EVERY visible text between tags — no exceptions
- Keep proper nouns (company names, person names, brand names) in original form
- Output ONLY the translated HTML — no markdown fences, no notes, no explanations

HTML to translate:
{html_content}"""

    api_key = settings.deepseek_api_key
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.deepseek_model,
                "messages": [
                    {"role": "system", "content": f"Professional {src_name}→{tgt_name} translator. Output only translated HTML with identical structure."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3, "max_tokens": 16384,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    result = _extract_html(raw)
    logger.info(f"Translated {len(html_content)}→{len(result)} chars {label}")
    return result


def _set_html_language(html_content: str, language: str) -> str:
    """Set the document language without changing translated article structure."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    if soup.html:
        soup.html["lang"] = language
    return str(soup)


@router.post("/api/articles/translate")
async def translate_article(request: Request, body: TranslateRequest):
    """Translate an existing HTML article to another language using AI."""
    settings = request.app.state.settings
    api_key = settings.deepseek_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    # Read source file from repo — prevent path traversal
    from pathlib import Path
    from src.sources.base import resolve_within
    base_dir = Path("data/site_sources") / settings.target_name
    try:
        source_file = resolve_within(base_dir, body.source_path)
    except ValueError:
        raise HTTPException(status_code=403, detail=f"Invalid source path: {body.source_path}")
    if not source_file.exists():
        raise HTTPException(status_code=404, detail=f"源文件不存在: {body.source_path}")

    source_html = source_file.read_text(encoding="utf-8")

    # ── For large pages, extract body content and translate section-by-section ──
    from bs4 import BeautifulSoup
    source_soup = BeautifulSoup(source_html, "html.parser")

    # Extract main content (skip nav/footer/header/scripts)
    html_body = source_soup.find("body")
    if not html_body:
        raise HTTPException(status_code=400, detail="Source file has no <body>")

    # Remove non-translatable elements
    for tag in html_body.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    # Collect content sections to translate
    sections: list[str] = []
    for tag in html_body.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "li", "blockquote", "table"]):
        text = tag.get_text(strip=True)
        if text and len(text) > 3:
            sections.append(str(tag))

    total_chars = sum(len(s) for s in sections)
    logger.info(f"Translation: {len(sections)} content sections, {total_chars} total chars from {body.source_path}")

    # If total is under 60000 chars, translate all at once
    # Otherwise, chunk into batches of ~40000 chars
    MAX_BATCH_CHARS = 40000
    translated_sections: list[str] = []

    if total_chars <= MAX_BATCH_CHARS * 1.5:
        # Single-pass translation
        content_html = "\n".join(sections)
        translated_html = await _do_translate(
            settings, content_html, body.source_lang, body.target_lang,
            LANG_NAMES, body.source_path,
        )
        translated_sections = [translated_html]
    else:
        # Chunked translation
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_chars = 0
        for s in sections:
            if current_chars + len(s) > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(s)
            current_chars += len(s)
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Chunked translation: {len(batches)} batches")
        for i, batch in enumerate(batches):
            batch_html = "\n".join(batch)
            chunk_translated = await _do_translate(
                settings, batch_html, body.source_lang, body.target_lang,
                LANG_NAMES, f"{body.source_path} (part {i+1}/{len(batches)})",
            )
            translated_sections.append(chunk_translated)

    # Combine translated content back into the original HTML structure
    final_soup = BeautifulSoup(source_html, "html.parser")
    final_body = final_soup.find("body")
    if final_body:
        # Replace body content with translated version
        combined = BeautifulSoup(
            "\n".join(translated_sections), "html.parser",
        )
        final_body.clear()
        for child in list(combined.children):
            final_body.append(child)

    translated_html = str(final_soup)

    # Save
    article_id = __import__("uuid").uuid4().hex[:12]
    title = _extract_title(translated_html) or body.target_path
    article_data = {
        "id": article_id, "topic": f"Translation: {body.source_path}",
        "keywords": "", "language": body.target_lang, "page_type": "blog",
        "title": title, "html": translated_html,
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
        "pushed": False, "source_path": body.source_path, "target_path": body.target_path,
    }
    (GENERATED_DIR / f"{article_id}.json").write_text(
        __import__("json").dumps(article_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (GENERATED_DIR / f"{article_id}.html").write_text(translated_html, encoding="utf-8")

    return {
        "id": article_id, "title": title,
        "target_path": body.target_path, "source_path": body.source_path,
        "html": translated_html,
        "source_html": source_html,  # original for comparison
    }


@router.get("/api/articles/translate-missing")
async def find_missing_translations(request: Request):
    """Find articles that exist in one language but are missing in others.

    Reads target config's language_paths to auto-detect which languages
    are configured and which pages are missing translations.
    Works with any language pair — not hardcoded to EN/JP.
    """
    from pathlib import Path
    from bs4 import BeautifulSoup
    import re

    settings = request.app.state.settings
    base = Path("data/site_sources") / settings.target_name

    if not base.exists():
        return {"missing": [], "count": 0, "message": "Site source directory not found"}

    # Read language config from target
    target_config = settings.__class__.load_target(settings.target_name)
    lang_paths: dict[str, str] = target_config.get("language_paths", {"en": "/"})
    lang_pairs: list[dict] = target_config.get("language_pairs", [])

    if len(lang_paths) <= 1:
        return {"missing": [], "count": 0, "message": "Only one language configured; nothing to translate"}

    # Map: language code → path prefix (e.g., "ja" → "jp")
    lang_prefix: dict[str, str] = {}
    for lang, path in lang_paths.items():
        prefix = path.strip("/")
        lang_prefix[lang] = prefix

    # Content page patterns — only these are considered "translatable content"
    # Excludes: products, about, contact, privacy, terms, home, legal, error pages
    CONTENT_PATTERNS = [
        "/blog/", "/insights/", "/guide/", "/article/", "/news/",
        "/tutorial/", "/market/", "/analysis/", "/report/", "/case-study/",
        "/whitepaper/", "/resources/",
    ]
    EXCLUDE_PATTERNS = [
        "/products/", "/about/", "/contact/", "/privacy/", "/terms/",
        "/legal/", "/cookies/", "/faq/", "/careers/", "/jobs/",
        "/search/", "/login/", "/signup/", "/cart/", "/checkout/",
        "google", "sitemap", "robots",  # exclude verification pages
    ]

    def _is_content_page(rel_path: str) -> bool:
        """Check if a path looks like a translatable content page."""
        path_lower = rel_path.lower()
        # Must match a content pattern
        if not any(p in path_lower for p in CONTENT_PATTERNS):
            return False
        # Must NOT match any exclude pattern
        if any(p in path_lower for p in EXCLUDE_PATTERNS):
            return False
        # Must be an HTML page, not a directory index without content
        if path_lower.endswith("/index.html"):
            # OK — could be blog index or article
            pass
        return True

    # Collect all content HTML files with their detected language
    all_files: list[dict] = []
    for f in base.rglob("*.html"):
        rel = str(f.relative_to(base)).replace("\\", "/")

        # ── Filter: only content pages ──
        if not _is_content_page(rel):
            continue

        # Detect language from html lang attribute
        detected_lang = None
        content = None
        try:
            content = f.read_text(encoding="utf-8")
            m = re.search(r'<html[^>]*lang=["\']([a-z]{2})["\']', content, re.I)
            if m:
                detected_lang = m.group(1).lower()
        except Exception:
            pass

        # Also detect from path prefix
        path_lang = None
        for lang, prefix in lang_prefix.items():
            if prefix and rel.startswith(prefix + "/"):
                path_lang = lang
                break
        if not path_lang and rel.split("/")[0] in lang_prefix.values():
            pfx = rel.split("/")[0]
            for lang, prefix in lang_prefix.items():
                if prefix == pfx:
                    path_lang = lang
                    break

        # Extract title for preview
        page_title = ""
        word_count = 0
        if content:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    page_title = title_tag.get_text(strip=True)[:120]
                for t in soup(["script", "style", "nav", "footer", "header"]):
                    t.decompose()
                body_text = soup.get_text(separator=" ", strip=True)
                word_count = len(body_text.split())
            except Exception:
                pass

        all_files.append({
            "path": rel,
            "lang": detected_lang or path_lang or "unknown",
            "path_lang": path_lang,
            "title": page_title,
            "word_count": word_count,
        })

    # Group by "content slug" — the path minus any language prefix
    # e.g., "blog/my-article/index.html" and "jp/blog/my-article/index.html"
    # share the same content_slug "blog/my-article/index.html"
    def content_slug(rel: str) -> str:
        parts = rel.split("/")
        # Strip known language prefixes from the path
        for prefix in lang_prefix.values():
            if prefix and parts[0] == prefix:
                return "/".join(parts[1:])
        return rel

    # Build existing translations: content_slug → {lang: path}
    existing: dict[str, dict[str, str]] = {}
    for f in all_files:
        slug = content_slug(f["path"])
        if slug not in existing:
            existing[slug] = {}
        existing[slug][f["lang"]] = f["path"]

    # Find missing translations
    primary_lang = list(lang_paths.keys())[0]
    missing = []
    for slug, translations in existing.items():
        # Only consider content that exists in the primary language
        if primary_lang not in translations:
            continue
        for target_lang in lang_paths:
            if target_lang == primary_lang:
                continue
            if target_lang not in translations:
                source_path = translations[primary_lang]
                # Build target path by mapping the primary prefix → target prefix
                target_prefix = lang_prefix.get(target_lang, "")
                target_path = f"{target_prefix}/{slug}" if target_prefix else slug
                missing.append({
                    "source_lang": primary_lang,
                    "target_lang": target_lang,
                    "source_path": source_path,
                    "target_path": target_path,
                })

    # Also check reverse direction (e.g., JP pages missing in EN)
    if lang_pairs:
        for lang in lang_paths:
            if lang == primary_lang:
                continue
            for slug, translations in existing.items():
                if lang in translations and primary_lang not in translations:
                    src = translations[lang]
                    target_prefix = lang_prefix.get(primary_lang, "")
                    tgt = f"{target_prefix}/{slug}" if target_prefix else slug
                    missing.append({
                        "source_lang": lang,
                        "target_lang": primary_lang,
                        "source_path": src,
                        "target_path": tgt,
                    })

    # Deduplicate
    seen = set()
    unique = []
    for m in missing:
        key = (m["source_path"], m["target_lang"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return {
        "missing": sorted(unique, key=lambda x: x["source_path"]),
        "count": len(unique),
        "languages": list(lang_paths.keys()),
        "primary_language": primary_lang,
    }


@router.get("/articles")
async def articles_page(request: Request):
    """Render the article generation page."""
    return templates.TemplateResponse(request, "articles.html")


def _research_plan_path(research_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", research_id):
        raise HTTPException(status_code=400, detail="Invalid research identifier")
    path = (RESEARCH_DIR / f"{research_id}.json").resolve()
    if path.parent != RESEARCH_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid research identifier")
    return path


def _recent_titles_for_website(website_url: str, *, limit: int = 30) -> list[str]:
    """Return recent primary-article titles for the same website."""
    def normalized_host(value: str) -> str:
        host = (urlparse(value).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host

    target_host = normalized_host(website_url)
    if not target_host:
        return []
    titles: list[str] = []
    files = sorted(
        GENERATED_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("source_article_id"):
            continue
        article_host = normalized_host(str(data.get("website_url", "")))
        if article_host != target_host:
            continue
        for value in (data.get("title"), data.get("topic")):
            cleaned = " ".join(str(value or "").split())[:300]
            if cleaned and cleaned.casefold() not in {item.casefold() for item in titles}:
                titles.append(cleaned)
                if len(titles) >= limit:
                    return titles
    return titles


@router.post("/api/articles/research")
async def research_article_plan(request: Request, body: AutoGenerateRequest):
    """Research a site, real-language queries, and competitors without writing an article."""
    settings = request.app.state.settings
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    from src.ai.automatic_article import AutomaticArticleWorkflow

    agent = _article_orchestrator(settings)
    agent_state = agent.start(body.website_url, body.model_dump())
    workflow = AutomaticArticleWorkflow(settings)
    excluded_topics = _recent_titles_for_website(body.website_url)
    try:
        research = await workflow.run(
            body.website_url,
            language=body.language,
            topic_hint=body.topic,
            keyword_hint=body.keywords,
            requested_page_type=body.page_type,
            max_reference_articles=body.max_reference_articles,
            excluded_topics=excluded_topics,
            content_direction=body.content_direction,
        )
        generation_context = workflow.build_generation_context(research)
    except HTTPException as exc:
        agent.fail(agent_state, str(exc.detail))
        raise
    except ValueError as exc:
        agent.fail(agent_state, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        agent.fail(agent_state, str(exc))
        logger.error("Article research failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Website and competitor research failed. Check that the URL is public and try again.",
        ) from exc
    finally:
        await workflow.close()

    research_id = uuid.uuid4().hex
    report = research.to_dict()
    agent_state = agent.complete_research(agent_state, research_id, report)
    plan_data = {
        "id": research_id,
        "status": "awaiting_confirmation",
        "request": body.model_dump(),
        "research_report": report,
        "generation_context": generation_context,
        "agent_run_id": agent_state.run_id,
        "excluded_existing_titles": excluded_topics,
        "created_at": datetime.now(UTC).isoformat(),
    }
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    _research_plan_path(research_id).write_text(
        json.dumps(plan_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "research_id": research_id,
        "status": plan_data["status"],
        "research_report": report,
        "writing_brief": report.get("writing_brief", {}),
        "agent_run_id": agent_state.run_id,
        "agent_stage": agent_state.stage,
    }


@router.post("/api/articles/generate-from-research")
async def generate_article_from_research(
    request: Request,
    body: GenerateFromResearchRequest,
):
    """Generate only after a saved research brief has been reviewed and confirmed."""
    settings = request.app.state.settings
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    plan_path = _research_plan_path(body.research_id)
    if not plan_path.is_file():
        raise HTTPException(status_code=404, detail="Research plan not found")
    try:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Research plan is invalid") from exc
    if plan_data.get("status") == "generated":
        raise HTTPException(
            status_code=409,
            detail="This research plan already generated an article. Start new research for a different topic.",
        )

    agent = _article_orchestrator(settings)
    agent_run_id = str(plan_data.get("agent_run_id", ""))
    try:
        agent_state = agent.load(agent_run_id)
    except (FileNotFoundError, ValueError):
        request_data = plan_data.get("request", {})
        agent_state = agent.start(
            str(request_data.get("website_url", "")),
            request_data,
        )
        agent_state.research_id = body.research_id
    agent_state = agent.begin_writing(agent_state)

    report = plan_data.get("research_report", {})
    brief = dict(report.get("writing_brief", {}))
    request_data = plan_data.get("request", {})
    profile = report.get("profile", {})
    primary_language = profile.get("primary_language") or request_data.get("language", "en")
    if primary_language not in LANG_NAMES:
        primary_language = "en"
    detected_languages = [
        language for language in profile.get("detected_languages", [])
        if language in LANG_NAMES
    ]
    requested_targets = (
        body.target_languages
        if body.target_languages is not None
        else [language for language in detected_languages if language != primary_language]
    )
    target_languages = []
    if body.auto_translate:
        for language in requested_targets:
            if language in LANG_NAMES and language != primary_language and language not in target_languages:
                target_languages.append(language)
    target_languages = target_languages[:5]
    allowed_types = {"blog", "market_analysis", "product_review", "guide", "news", "landing"}
    allowed_directions = {
        "news", "industry_trend", "market_event", "evergreen_guide",
        "buyer_question", "deep_analysis",
    }
    page_type = body.page_type if body.page_type in allowed_types else brief.get("page_type", "blog")
    content_direction = (
        body.content_direction
        if body.content_direction in allowed_directions
        else brief.get("content_direction", "evergreen_guide")
    )
    headline = body.headline.strip() or next(
        iter(brief.get("headline_options", [])),
        brief.get("topic", "Article"),
    )
    word_count = body.word_count or brief.get("recommended_word_count") or request_data.get("word_count", 1200)
    outline = [" ".join(item.split())[:200] for item in body.outline if item.strip()][:12]
    if not outline:
        outline = brief.get("recommended_outline", [])[:12]
    confirmed_brief = {
        **brief,
        "confirmed_headline": headline,
        "confirmed_word_count": word_count,
        "confirmed_page_type": page_type,
        "confirmed_content_direction": content_direction,
        "confirmed_outline": outline,
    }
    prompt = _build_article_prompt(
        topic=headline,
        keywords=", ".join(brief.get("target_keywords", [])),
        language=primary_language,
        word_count=word_count,
        page_type=page_type,
        site_research=(
            plan_data.get("generation_context", "")
            + "\n\n## USER-CONFIRMED BRIEF\n"
            + json.dumps(confirmed_brief, ensure_ascii=False, indent=2)
        ),
    )

    from src.agents.writing_agent import ArticleWritingAgent, WritingTask

    writing_agent = ArticleWritingAgent(settings)
    writing_task = WritingTask(
        prompt=prompt,
        page_type=page_type,
        content_direction=content_direction,
        language=primary_language,
    )
    revision_count = 0
    try:
        raw = await writing_agent.write(writing_task)
        html_content = _sanitize_generated_html(_extract_html(raw))
        quality_preview = agent.quality_agent.inspect_content(
            html_content,
            research_report=report,
            expected_word_count=word_count,
        )
        while not quality_preview.passed and revision_count < 2:
            issues = [
                check.message
                for check in quality_preview.checks
                if not check.passed and check.severity == "error"
            ]
            raw = await writing_agent.revise(writing_task, html_content, issues)
            revision_count += 1
            html_content = _sanitize_generated_html(_extract_html(raw))
            quality_preview = agent.quality_agent.inspect_content(
                html_content,
                research_report=report,
                expected_word_count=word_count,
            )
    except Exception as exc:
        agent.fail(agent_state, str(exc))
        logger.error("Confirmed article generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Research is saved, but article generation failed.") from exc
    finally:
        await writing_agent.close()

    title = _extract_title(html_content) or headline
    article_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(UTC).isoformat()
    article_data = {
        "id": article_id,
        "research_id": body.research_id,
        "website_url": report.get("profile", {}).get("website_url", ""),
        "topic": headline,
        "keywords": ", ".join(brief.get("target_keywords", [])),
        "language": primary_language,
        "page_type": page_type,
        "content_direction": content_direction,
        "title": title,
        "html": html_content,
        "confirmed_brief": confirmed_brief,
        "research_report": report,
        "created_at": created_at,
        "pushed": False,
        "translation_group_id": article_id,
        "translations": [],
        "agent_run_id": agent_state.run_id,
        "revision_count": revision_count,
    }
    (GENERATED_DIR / f"{article_id}.json").write_text(
        json.dumps(article_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (GENERATED_DIR / f"{article_id}.html").write_text(html_content, encoding="utf-8")

    async def create_translation(target_language: str) -> tuple[dict | None, dict | None]:
        try:
            translated = await _do_translate(
                settings,
                html_content,
                primary_language,
                target_language,
                LANG_NAMES,
                f"generated article {article_id}",
            )
            translated_html = _set_html_language(
                _sanitize_generated_html(translated),
                target_language,
            )
            translation_id = uuid.uuid4().hex[:12]
            translation_title = _extract_title(translated_html) or title
            translation_data = {
                "id": translation_id,
                "research_id": body.research_id,
                "website_url": article_data["website_url"],
                "topic": article_data["topic"],
                "keywords": article_data["keywords"],
                "language": target_language,
                "source_language": primary_language,
                "page_type": page_type,
                "content_direction": content_direction,
                "title": translation_title,
                "html": translated_html,
                "source_html": html_content,
                "source_article_id": article_id,
                "translation_group_id": article_id,
                "confirmed_brief": confirmed_brief,
                "research_report": report,
                "created_at": datetime.now(UTC).isoformat(),
                "pushed": False,
            }
            (GENERATED_DIR / f"{translation_id}.json").write_text(
                json.dumps(translation_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (GENERATED_DIR / f"{translation_id}.html").write_text(
                translated_html,
                encoding="utf-8",
            )
            return translation_data, None
        except Exception as exc:
            logger.error(
                "Translation of article %s to %s failed: %s",
                article_id,
                target_language,
                exc,
                exc_info=True,
            )
            return None, {"language": target_language, "error": str(exc)[:200]}

    translation_results = await asyncio.gather(*(
        create_translation(language) for language in target_languages
    )) if quality_preview.passed else []
    translations = [result for result, _ in translation_results if result]
    translation_errors = [error for _, error in translation_results if error]
    article_data["translations"] = [
        {"id": item["id"], "language": item["language"], "title": item["title"]}
        for item in translations
    ]
    (GENERATED_DIR / f"{article_id}.json").write_text(
        json.dumps(article_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    agent_state = agent.complete_writing(
        agent_state,
        article_data,
        revision_count=revision_count,
    )
    article_data["agent_stage"] = agent_state.stage
    article_data["quality_report"] = (
        agent_state.content_quality.model_dump()
        if agent_state.content_quality
        else {}
    )
    (GENERATED_DIR / f"{article_id}.json").write_text(
        json.dumps(article_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plan_data["status"] = "generated"
    plan_data["generated_article_id"] = article_id
    plan_data["generated_article_ids"] = [article_id, *[item["id"] for item in translations]]
    plan_data["translation_errors"] = translation_errors
    plan_data["confirmed_brief"] = confirmed_brief
    plan_data["agent_run_id"] = agent_state.run_id
    plan_path.write_text(json.dumps(plan_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "id": article_id,
        "research_id": body.research_id,
        "title": title,
        "html": html_content,
        "created_at": created_at,
        "topic": article_data["topic"],
        "page_type": page_type,
        "language": primary_language,
        "translations": translations,
        "translation_errors": translation_errors,
        "confirmed_brief": confirmed_brief,
        "research_report": report,
        "agent_run_id": agent_state.run_id,
        "agent_stage": agent_state.stage,
        "quality_report": article_data["quality_report"],
        "revision_count": revision_count,
    }


@router.get("/api/articles/agent-runs/{run_id}")
async def get_article_agent_run(request: Request, run_id: str):
    """Return the persisted decisions and joint quality checks for one article run."""
    try:
        state = _article_orchestrator(request.app.state.settings).load(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.model_dump(mode="json")


@router.post("/api/articles/auto-generate")
async def auto_generate_article(request: Request, body: AutoGenerateRequest):
    """Detect a site's business, research search-result structures, and write a draft."""
    settings = request.app.state.settings
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    from src.ai.automatic_article import AutomaticArticleWorkflow

    workflow = AutomaticArticleWorkflow(settings)
    excluded_topics = _recent_titles_for_website(body.website_url)
    try:
        research = await workflow.run(
            body.website_url,
            language=body.language,
            topic_hint=body.topic,
            keyword_hint=body.keywords,
            requested_page_type=body.page_type,
            max_reference_articles=body.max_reference_articles,
            excluded_topics=excluded_topics,
            content_direction=body.content_direction,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Automatic article research failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="网站研究失败，请检查网址是否公开可访问后重试。",
        ) from exc
    finally:
        await workflow.close()

    profile = research.profile
    topic = research.editorial_decision.topic or body.topic.strip() or profile.recommended_topic
    page_type = research.editorial_decision.page_type
    keywords = profile.keywords
    prompt = _build_article_prompt(
        topic=topic,
        keywords=", ".join(keywords),
        language=body.language,
        word_count=body.word_count,
        page_type=page_type,
        site_research=AutomaticArticleWorkflow.build_generation_context(research),
    )

    from src.agents.quality_agent import ArticleQualityAgent
    from src.agents.writing_agent import ArticleWritingAgent, WritingTask

    writing_agent = ArticleWritingAgent(settings)
    writing_task = WritingTask(
        prompt=prompt,
        page_type=page_type,
        content_direction=research.editorial_decision.content_direction,
        language=profile.primary_language,
    )
    quality_agent = ArticleQualityAgent()
    research_report = research.to_dict()
    revision_count = 0
    try:
        raw = await writing_agent.write(writing_task)
        html_content = _sanitize_generated_html(_extract_html(raw))
        quality_report = quality_agent.inspect_content(
            html_content,
            research_report=research_report,
            expected_word_count=body.word_count,
        )
        while not quality_report.passed and revision_count < 2:
            issues = [
                check.message
                for check in quality_report.checks
                if not check.passed and check.severity == "error"
            ]
            raw = await writing_agent.revise(writing_task, html_content, issues)
            revision_count += 1
            html_content = _sanitize_generated_html(_extract_html(raw))
            quality_report = quality_agent.inspect_content(
                html_content,
                research_report=research_report,
                expected_word_count=body.word_count,
            )
    except Exception as exc:
        logger.error("Automatic article generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="研究已完成，但 AI 文章生成失败，请稍后重试。",
        ) from exc
    finally:
        await writing_agent.close()

    title = _extract_title(html_content) or topic
    article_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(UTC).isoformat()
    article_data = {
        "id": article_id,
        "website_url": profile.website_url,
        "topic": topic,
        "keywords": ", ".join(keywords),
        "language": body.language,
        "page_type": page_type,
        "requested_page_type": body.page_type,
        "title": title,
        "html": html_content,
        "research_report": research_report,
        "quality_report": quality_report.model_dump(),
        "revision_count": revision_count,
        "created_at": created_at,
        "pushed": False,
    }
    (GENERATED_DIR / f"{article_id}.json").write_text(
        json.dumps(article_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (GENERATED_DIR / f"{article_id}.html").write_text(html_content, encoding="utf-8")
    return {
        "id": article_id,
        "title": title,
        "html": html_content,
        "created_at": created_at,
        "topic": topic,
        "page_type": page_type,
        "research_report": research_report,
        "quality_report": quality_report.model_dump(),
        "revision_count": revision_count,
    }


@router.post("/api/articles/generate")
async def generate_article(request: Request, body: GenerateRequest):
    """Generate an SEO-optimized article using AI."""
    settings = request.app.state.settings
    api_key = settings.deepseek_api_key

    if not api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    # ── Research phase: search authoritative sources for citations ──
    research_findings = ""
    citations_used = []
    if body.with_research:
        from src.ai.article_researcher import (
            ResearchResult,
            research_topic,
            build_citation_prompt,
            get_static_citations,
        )
        kw_list = [k.strip() for k in body.keywords.split(",") if k.strip()]
        try:
            research = await research_topic(
                topic=body.topic,
                keywords=kw_list,
                topic_area=body.topic_area,
                max_sources=5,
            )
            if research.findings:
                research_findings = build_citation_prompt(research.findings)
                citations_used = [
                    {"label": f.source_label, "url": f.url, "type": f.source_type}
                    for f in research.findings
                ]
                logger.info(
                    f"Research: found {len(research.findings)} sources for '{body.topic}'"
                )
            else:
                # Fallback to static authoritative citations
                research_findings = get_static_citations(body.topic_area)
                logger.info(f"Research: no live results, using static citations for '{body.topic}'")
        except Exception as e:
            logger.warning(f"Research failed for '{body.topic}': {e}")
            research_findings = get_static_citations(body.topic_area)

    page_type = await _resolve_article_type(
        settings,
        topic=body.topic,
        keywords=body.keywords,
        language=body.language,
        requested=body.page_type,
    )
    prompt = _build_article_prompt(
        topic=body.topic,
        keywords=body.keywords,
        language=body.language,
        word_count=body.word_count,
        page_type=page_type,
        research_findings=research_findings,
    )

    try:
        import httpx
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.deepseek_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert SEO content writer. You write complete, well-structured HTML articles optimized for search engines. Always output valid HTML with proper SEO tags."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Article generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI 生成失败，请稍后重试。如持续失败请检查 API Key 配置。")

    # Extract HTML from the response
    html_content = _sanitize_generated_html(_extract_html(raw))
    title = _extract_title(html_content) or body.topic

    # Store
    article_id = uuid.uuid4().hex[:12]
    article_data = {
        "id": article_id,
        "topic": body.topic,
        "keywords": body.keywords,
        "language": body.language,
        "page_type": page_type,
        "requested_page_type": body.page_type,
        "title": title,
        "html": html_content,
        "citations": citations_used,
        "with_research": body.with_research,
        "created_at": datetime.utcnow().isoformat(),
        "pushed": False,
    }

    file_path = GENERATED_DIR / f"{article_id}.json"
    file_path.write_text(json.dumps(article_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also save the HTML separately for easy viewing
    html_path = GENERATED_DIR / f"{article_id}.html"
    html_path.write_text(html_content, encoding="utf-8")

    return {
        "id": article_id,
        "title": title,
        "html": html_content,
        "created_at": article_data["created_at"],
    }


@router.get("/api/articles/{article_id}")
async def get_article(request: Request, article_id: str):
    """Get a previously generated article."""
    file_path = GENERATED_DIR / f"{article_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Article not found")

    data = json.loads(file_path.read_text(encoding="utf-8-sig"))
    if data.get("images"):
        from src.web.routes.article_images import _display_generated_html
        data["html"] = _display_generated_html(article_id, data["html"])
    translation_versions = []
    for reference in data.get("translations", []):
        translation_id = str(reference.get("id", ""))
        if not ARTICLE_ID_RE.fullmatch(translation_id):
            continue
        translation_path = GENERATED_DIR / f"{translation_id}.json"
        if not translation_path.is_file():
            continue
        translation = json.loads(translation_path.read_text(encoding="utf-8-sig"))
        if translation.get("images"):
            from src.web.routes.article_images import _display_generated_html
            translation["html"] = _display_generated_html(translation_id, translation["html"])
        translation_versions.append(translation)
    if translation_versions:
        data["translations"] = translation_versions
    return data


@router.get("/api/articles")
async def list_articles(request: Request):
    """List all generated articles."""
    articles = []
    for f in sorted(GENERATED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        articles.append({
            "id": data["id"],
            "topic": data["topic"],
            "title": data["title"],
            "created_at": data["created_at"],
            "page_type": data.get("page_type", "blog"),
            "image_count": data.get("image_count", 0),
            "pushed": data.get("pushed", False),
        })
    return articles


@router.post("/api/articles/{article_id}/push")
async def push_article(request: Request, article_id: str, body: PushRequest):
    """Push a generated article to a GitHub repository."""
    if not request.app.state.settings.web_allow_repo_writes:
        raise HTTPException(status_code=403, detail="Repository writes are disabled")
    if not ARTICLE_ID_RE.fullmatch(article_id):
        raise HTTPException(status_code=400, detail="Invalid article identifier")

    file_path = GENERATED_DIR / f"{article_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Article not found")

    data = json.loads(file_path.read_text(encoding="utf-8-sig"))
    html_content = data["html"]

    if not body.repo_url:
        raise HTTPException(status_code=400, detail="请填写 GitHub 仓库地址")

    repo_url, requested_branch = validate_github_repo(body.repo_url, body.branch)
    repo_slug = _parse_repo_slug(repo_url)
    if not repo_slug:
        raise HTTPException(status_code=400, detail=f"无法解析仓库地址: {body.repo_url}，请输入如 https://github.com/owner/repo 的格式")

    # Determine target file path in repo
    target_path = body.file_path.strip() or f"blog/{article_id}/index.html"
    if not target_path.endswith(".html"):
        target_path = target_path.rstrip("/") + "/index.html"

    # Clone repo and push
    import asyncio
    import shutil
    from datetime import datetime

    work_dir = Path("data/site_sources") / f"article_{article_id}_{uuid.uuid4().hex[:8]}"
    branch_name = f"article/{article_id}"
    try:
        article_file = resolve_within(work_dir, target_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Target path must stay inside the repository",
        ) from exc

    orchestrator = None
    agent_state = None
    agent_run_id = str(data.get("agent_run_id", ""))
    if agent_run_id:
        orchestrator = _article_orchestrator(request.app.state.settings)
        try:
            agent_state = orchestrator.begin_publishing(
                orchestrator.load(agent_run_id),
                repo_url,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    errors: list[str] = []

    try:
        actual_branch = requested_branch
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", "-b", requested_branch,
            repo_url, str(work_dir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode().strip() if stderr else "unknown error"
            raise RuntimeError(f"克隆仓库失败 ({requested_branch}): {err}")

        # Write article
        target_path = article_file.relative_to(work_dir.resolve()).as_posix()
        article_file.parent.mkdir(parents=True, exist_ok=True)
        article_file.write_text(html_content, encoding="utf-8")
        staged_paths = [target_path]
        generated_assets = GENERATED_DIR / "assets" / article_id
        if generated_assets.is_dir():
            repo_assets = article_file.parent / "images"
            repo_assets.mkdir(parents=True, exist_ok=True)
            for asset in generated_assets.iterdir():
                if not asset.is_file():
                    continue
                destination = repo_assets / asset.name
                shutil.copy2(asset, destination)
                staged_paths.append(destination.relative_to(work_dir.resolve()).as_posix())

        # Git operations
        async def git(*args):
            p = await asyncio.create_subprocess_exec(
                "git", "-C", str(work_dir), *args,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await p.communicate()
            return p.returncode, out, err

        # Create branch
        code, _, stderr = await git("checkout", "-b", branch_name)
        if code != 0:
            errors.append(f"创建分支失败: {stderr.decode().strip()}")

        code, _, stderr = await git("add", *staged_paths)
        if code != 0:
            errors.append(f"添加文件失败: {stderr.decode().strip()}")

        # Commit
        safe_title = data.get("title", "Untitled")
        commit_msg = (
            f"Add article: {safe_title}\n\n"
            f"Generated by Site Inspector AI\n"
            f"Topic: {data.get('topic', '')}"
        )
        code, _, stderr = await git("commit", "-m", commit_msg)
        if code != 0:
            err_text = stderr.decode().strip() if stderr else ""
            if "nothing to commit" not in err_text.lower():
                errors.append(f"提交失败: {err_text}")

        if errors:
            raise RuntimeError("; ".join(errors))

        # Push
        code, _, stderr = await git("push", "origin", branch_name)
        if code != 0:
            err_text = stderr.decode().strip() if stderr else ""
            raise RuntimeError(f"推送失败: {err_text}")

        # Create PR via gh REST API (GraphQL often times out)
        pr_title = f"新文章: {safe_title}"
        pr_body = (
            f"## AI 生成文章\\n\\n"
            f"**主题:** {data.get('topic', '')}\\n"
            f"**关键词:** {data.get('keywords', '')}\\n"
            f"**语言:** {data.get('language', '')}\\n\\n"
            f"---\\n"
            f"由 Site Inspector AI 自动生成，请审查后合并。"
        )
        proc = await asyncio.create_subprocess_exec(
            "gh", "api", f"repos/{repo_slug}/pulls",
            "-f", f"title={pr_title}",
            "-f", f"head={branch_name}",
            "-f", f"base={actual_branch}",
            "-f", f"body={pr_body}",
            "--jq", ".html_url",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_text = stderr.decode().strip() if stderr else ""
            raise RuntimeError(f"创建 PR 失败: {err_text}")

        pr_url = stdout.decode().strip()

        # Update article metadata
        data["pushed"] = True
        data["pr_url"] = pr_url
        data["repo_url"] = repo_url
        data["pushed_at"] = datetime.utcnow().isoformat()
        if orchestrator and agent_state:
            agent_state = orchestrator.complete_publishing(agent_state, pr_url)
            data["agent_stage"] = agent_state.stage
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "success": True,
            "pr_url": pr_url,
            "agent_run_id": agent_run_id,
            "agent_stage": agent_state.stage if agent_state else "",
        }

    except Exception as e:
        if orchestrator and agent_state:
            orchestrator.fail(agent_state, str(e))
        logger.error(f"Article push failed: {e}", exc_info=True)
        safe_error = re.sub(r"https://[^@\s]+@", "https://***@", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"推送失败: {safe_error[:500]}",
        ) from e
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ── Helpers ───────────────────────────────────────────────────────

def _parse_repo_slug(repo_url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL.

    >>> _parse_repo_slug('https://github.com/owner/repo')
    'owner/repo'
    >>> _parse_repo_slug('https://github.com/owner/repo.git')
    'owner/repo'
    >>> _parse_repo_slug('git@github.com:owner/repo.git')
    'owner/repo'
    """
    import re
    # HTTPS format: https://github.com/owner/repo(.git)?
    m = re.match(r'https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$', repo_url.strip())
    if m:
        return m.group(1)
    # SSH format: git@github.com:owner/repo(.git)?
    m = re.match(r'git@github\.com:([^/]+/[^/]+?)(?:\.git)?$', repo_url.strip())
    if m:
        return m.group(1)
    # Already owner/repo format
    m = re.match(r'^([^/]+/[^/]+?)(?:\.git)?$', repo_url.strip())
    if m:
        return m.group(1)
    return ""


def _extract_html(raw: str) -> str:
    """Extract HTML content from AI response, stripping markdown fences."""
    import re
    # Remove ```html ... ``` fences if present
    m = re.search(r"```html\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: look for <!DOCTYPE or <html
    m = re.search(r"(<!DOCTYPE.*?</html>)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<html.*?</html>)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _extract_title(html_content: str) -> str:
    """Extract <title> from HTML."""
    import re
    m = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<h1>(.*?)</h1>", html_content, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _sanitize_generated_html(html_content: str) -> str:
    """Remove executable content from model-generated article HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup.find_all(["script", "iframe", "object", "embed", "form", "base"]):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.lower().startswith("on"):
                del tag.attrs[attr]
        for attr in ("href", "src", "action"):
            value = tag.get(attr)
            if isinstance(value, str) and value.strip().lower().startswith(
                ("javascript:", "data:text/html")
            ):
                del tag.attrs[attr]
    return str(soup)


async def _resolve_article_type(
    settings,
    *,
    topic: str,
    keywords: str,
    language: str,
    requested: str,
) -> str:
    """Let AI choose a format when the caller requests automatic selection."""
    allowed = {"blog", "market_analysis", "product_review", "guide", "news", "landing"}
    if requested in allowed:
        return requested

    combined = f"{topic} {keywords}".casefold()
    if any(term in combined for term in ("how to", "guide", "教程", "指南", "方法")):
        fallback = "guide"
    elif any(term in combined for term in ("trend", "market", "forecast", "趋势", "市场", "预测")):
        fallback = "market_analysis"
    elif any(term in combined for term in ("compare", "review", "best", "对比", "评测", "推荐")):
        fallback = "product_review"
    elif any(term in combined for term in ("news", "latest", "today", "新闻", "最新", "今日")):
        fallback = "news"
    else:
        fallback = "blog"

    from src.ai.deepseek_client import DeepSeekClient

    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        timeout=settings.deepseek_timeout,
    )
    try:
        result = await client.generate_json(
            f"""Choose the best article format for this topic.
Topic: {topic}
Keywords: {keywords or 'none'}
Language: {language}

Return one field named page_type with exactly one value from:
blog, market_analysis, product_review, guide, news, landing.
Use news only for genuinely time-sensitive topics; use landing only for transactional intent.""",
            system="You are a digital editor choosing the most useful content format.",
            temperature=0.1,
            max_tokens=200,
        )
        selected = result.get("page_type") if isinstance(result, dict) else None
        return selected if selected in allowed else fallback
    except Exception as exc:
        logger.warning("AI article type selection failed, using %s: %s", fallback, exc)
        return fallback
    finally:
        await client.close()


# ── Prompt Builder ────────────────────────────────────────────────

def _build_article_prompt(
    topic: str, keywords: str, language: str, word_count: int, page_type: str,
    research_findings: str = "",
    site_research: str = "",
) -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    lang_name = {
        "en": "English", "ja": "Japanese", "zh": "Chinese",
        "ko": "Korean", "fr": "French", "de": "German",
        "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
    }.get(language, language.capitalize())
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]

    type_labels = {
        "blog": "Blog Article",
        "market_analysis": "Market Analysis Report",
        "product_review": "Product Review",
        "guide": "How-To Guide / Tutorial",
        "news": "News Article",
        "landing": "Landing Page",
    }
    type_label = type_labels.get(page_type, "Blog Article")

    # Type-specific structure hints
    type_structure = {
        "market_analysis": (
            "Structure: Executive Summary → Market Overview → Key Trends → "
            "Competitive Landscape → Data & Statistics → Outlook & Forecast → Conclusion"
        ),
        "product_review": (
            "Structure: Product Overview → Key Features → Performance & Quality → "
            "Pros & Cons (use a comparison table) → Price & Value → "
            "Comparison with Competitors → Verdict & Rating"
        ),
        "guide": (
            "Structure: Introduction → Prerequisites / What You'll Need → "
            "Step-by-Step Instructions (numbered H2 sections) → "
            "Common Mistakes to Avoid → Pro Tips → Conclusion"
        ),
        "news": (
            "Structure: Headline / Lead → Key Facts (who/what/when/where/why) → "
            "Background & Context → Expert Commentary / Quotes → "
            "Impact Analysis → What's Next"
        ),
    }

    structure_hint = type_structure.get(page_type, "")

    citation_section = research_findings if research_findings else ""

    return f"""Write a complete, SEO-optimized HTML article.

{citation_section}

{site_research}

Topic: {topic}
Article Type: {type_label}
Language: {lang_name}
Target Word Count: ~{word_count} words
{f"Primary Keywords: {', '.join(kw_list)}" if kw_list else ""}
{f"Suggested Structure: {structure_hint}" if structure_hint else ""}

CRITICAL RULES:
- Today's date is {today}; use it for publication metadata, not as a reason to make
  unsupported claims sound current
- Do NOT include author name, byline, or author information anywhere
- Do NOT include <meta name="author"> tag
- Do NOT include author in JSON-LD schema
- The article is published by the site/organization, not an individual

Output — return ONLY valid HTML (no markdown fences, no explanations):

```html
<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<title>[SEO title with primary keyword, 50-60 chars]</title>
<meta name="description" content="[Compelling meta description, 120-160 chars, includes primary keyword]">
<meta property="og:title" content="[SEO title]">
<meta property="og:description" content="[Meta description]">
<meta property="og:type" content="article">
<script type="application/ld+json">
{{"@context":"https://schema.org","@type":"Article","headline":"[Title]","publisher":{{"@type":"Organization","name":"Site Name"}},"datePublished":"{today}","dateModified":"{today}"}}
</script>
</head>
<body>
<article>
<header>
<h1>[Compelling H1 with primary keyword]</h1>
<p class="article-date">Published: {today}</p>
</header>

<!-- Open with the reader's concrete question, situation, or verified fact. Follow the
confirmed outline when one is supplied; otherwise create specific, non-generic H2 sections. -->

</article>
</body>
</html>
```

IMPORTANT:
- Replace ALL placeholder brackets [...] with actual content
- Use "{today}" as the publication date in visible metadata and schema
- Write in {lang_name}
- Target ~{word_count} words for body content
- Treat the suggested {type_label} structure as guidance, not a mandatory template
- Naturally include keywords without stuffing
- Use proper heading hierarchy (H1 → H2 → no skips)
- Start directly; do not use generic scene-setting such as "in today's rapidly changing world"
- Make every H2 contribute distinct facts, mechanisms, examples, trade-offs, or actions
- Prefer concrete explanations over claims such as "crucial", "revolutionary", or
  "industry-leading"
- Include data only when the verified research context supplies a source that supports it
- Do not invent facts about the site; use only the verified site context above
- Do not invent personal experience, customer stories, interviews, quotations, statistics,
  prices, typical ranges, rankings, records, or market trends
- Treat reference articles as structural research only and never copy their wording
- Output RAW HTML only — no markdown backticks, no explanations"""
