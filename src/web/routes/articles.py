from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.web.deps import get_db, templates
from src.sources.base import resolve_within
from src.web.security import validate_github_repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["articles"])

GENERATED_DIR = Path("data/generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ARTICLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


class GenerateRequest(BaseModel):
    topic: str
    keywords: str = ""
    language: str = "en"
    word_count: int = 800
    page_type: str = "blog"  # blog, market_analysis, product_review, guide, news, landing


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
    page_type: str = "blog"


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
            prompt = _build_article_prompt(
                topic=topic.strip(),
                keywords=body.keywords,
                language=body.language,
                word_count=body.word_count,
                page_type=body.page_type,
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

            html_content = _extract_html(raw)
            title = _extract_title(html_content) or topic.strip()
            article_id = __import__("uuid").uuid4().hex[:12]
            article_data = {
                "id": article_id, "topic": topic.strip(), "keywords": body.keywords,
                "language": body.language, "page_type": body.page_type,
                "title": title, "html": html_content,
                "created_at": __import__("datetime").datetime.utcnow().isoformat(),
                "pushed": False,
            }
            (GENERATED_DIR / f"{article_id}.json").write_text(
                __import__("json").dumps(article_data, ensure_ascii=False, indent=2), encoding="utf-8")
            (GENERATED_DIR / f"{article_id}.html").write_text(html_content, encoding="utf-8")
            results.append({"topic": topic.strip(), "id": article_id, "title": title, "status": "ok"})
        except Exception as e:
            logger.error(f"Batch generate failed for '{topic}': {e}")
            results.append({"topic": topic.strip(), "status": "failed", "error": str(e)[:200]})

    return {"total": len(body.topics), "generated": sum(1 for r in results if r["status"] == "ok"), "results": results}


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

    lang_names = {
        "en": "English", "ja": "Japanese", "zh": "Chinese",
        "ko": "Korean", "fr": "French", "de": "German",
        "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
        "ar": "Arabic", "vi": "Vietnamese", "th": "Thai",
    }
    src_name = lang_names.get(body.source_lang, body.source_lang.upper())
    tgt_name = lang_names.get(body.target_lang, body.target_lang.upper())

    prompt = f"""Translate EVERY part of the following HTML article from {src_name} to {tgt_name}.
You MUST translate ALL sections, paragraphs, headings, lists, and table content.
Do NOT summarize, skip, or truncate anything.

CRITICAL RULES:
- Keep ALL HTML tags, attributes, and structure exactly as-is
- Translate EVERY visible text between tags — no exceptions
- Translate all headings (h1, h2, h3, h4), all paragraphs, all list items, all table cells
- Keep proper nouns (company names, person names, brand names) in original form
- Translate meta description, OG tags, title, and alt text
- Keep hreflang, canonical URLs, schema/JSON-LD markup unchanged
- Output ONLY the complete translated HTML — no markdown fences, no notes

HTML to translate (FULL — do not skip any part):
{source_html}"""

    try:
        import httpx
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": settings.deepseek_model,
                    "messages": [
                        {"role": "system", "content": f"You are a professional {src_name}→{tgt_name} translator. You translate HTML while preserving all tags and structure exactly. Output only the translated HTML."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3, "max_tokens": 32768,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="翻译失败，请稍后重试")

    translated_html = _extract_html(raw)

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

    # Collect all HTML files with their detected language
    all_files: list[dict] = []
    for f in base.rglob("*.html"):
        rel = str(f.relative_to(base)).replace("\\", "/")

        # Detect language from html lang attribute
        detected_lang = None
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

        all_files.append({
            "path": rel,
            "lang": detected_lang or path_lang or "unknown",
            "path_lang": path_lang,
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


@router.post("/api/articles/generate")
async def generate_article(request: Request, body: GenerateRequest):
    """Generate an SEO-optimized article using AI."""
    settings = request.app.state.settings
    api_key = settings.deepseek_api_key

    if not api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not configured")

    prompt = _build_article_prompt(
        topic=body.topic,
        keywords=body.keywords,
        language=body.language,
        word_count=body.word_count,
        page_type=body.page_type,
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
    html_content = _extract_html(raw)
    title = _extract_title(html_content) or body.topic

    # Store
    article_id = uuid.uuid4().hex[:12]
    article_data = {
        "id": article_id,
        "topic": body.topic,
        "keywords": body.keywords,
        "language": body.language,
        "page_type": body.page_type,
        "title": title,
        "html": html_content,
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

        code, _, stderr = await git("add", target_path)
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
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"success": True, "pr_url": pr_url}

    except Exception as e:
        logger.error(f"Article push failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="推送失败，请检查仓库地址和网络连接后重试。")
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


# ── Prompt Builder ────────────────────────────────────────────────

def _build_article_prompt(
    topic: str, keywords: str, language: str, word_count: int, page_type: str,
) -> str:
    from datetime import datetime
    today = datetime.utcnow().strftime("%Y-%m-%d")

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

    return f"""Write a complete, SEO-optimized HTML article.

Topic: {topic}
Article Type: {type_label}
Language: {lang_name}
Target Word Count: ~{word_count} words
{f"Primary Keywords: {', '.join(kw_list)}" if kw_list else ""}
{f"Suggested Structure: {structure_hint}" if structure_hint else ""}

CRITICAL RULES:
- Today's date is {today} — use this exact date in the content (NOT "[today]" placeholder)
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
<p class="article-type">Category: {type_label}</p>
</header>

<!-- Article body with proper H2 sections, paragraphs, lists, and data as appropriate -->

</article>
</body>
</html>
```

IMPORTANT:
- Replace ALL placeholder brackets [...] with actual content
- Use "{today}" as the publication date everywhere (in text, schema, meta)
- Write in {lang_name}
- Target ~{word_count} words for body content
- Follow the suggested {type_label} structure
- Naturally include keywords without stuffing
- Use proper heading hierarchy (H1 → H2 → no skips)
- Include specific examples, data, or practical insights
- Output RAW HTML only — no markdown backticks, no explanations"""
