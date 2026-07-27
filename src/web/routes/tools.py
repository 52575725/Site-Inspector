"""Tools API: GSC data, competitor tracking, image WebP conversion."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.web.deps import templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])


class CompetitorRequest(BaseModel):
    urls: list[str]


class WebPRequest(BaseModel):
    directory: str


@router.get("/tools")
async def tools_page(request: Request):
    """Render the tools page."""
    return templates.TemplateResponse(request, "tools.html")


# ── Competitor Tracking ─────────────────────────────────────────

@router.post("/api/tools/competitor/snapshot")
async def competitor_snapshot(request: Request, body: CompetitorRequest):
    """Take snapshots of competitor pages and detect changes."""
    from src.inspectors.competitor import CompetitorTracker
    tracker = CompetitorTracker()
    results = await tracker.snapshot_all(body.urls)
    return {"results": results}


@router.get("/api/tools/competitor/history")
async def competitor_history(request: Request, url: str):
    """Get snapshot history for a competitor URL."""
    from src.inspectors.competitor import CompetitorTracker
    tracker = CompetitorTracker()
    return tracker.get_history(url)


# ── Image WebP Conversion ───────────────────────────────────────

@router.post("/api/tools/convert-webp")
async def convert_webp(request: Request, body: WebPRequest):
    """Batch-convert images in a directory to WebP format."""
    from src.integrations.image_webp import batch_convert_directory
    from src.sources.base import resolve_within
    from pathlib import Path

    # Prevent path traversal: only allow directories within data/
    settings = request.app.state.settings
    allowed_root = settings.data_dir.resolve()
    dir_path = Path(body.directory).resolve()
    try:
        dir_path.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Directory must be within {allowed_root}",
        )
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    results = batch_convert_directory(str(dir_path))
    return results


# ── GSC Data ────────────────────────────────────────────────────

@router.get("/api/tools/gsc/summary")
async def gsc_summary(request: Request):
    """Get basic GSC data if credentials are configured."""
    from datetime import date, timedelta
    settings = request.app.state.settings
    gsc_path = settings.google_credentials_path
    gsc_property = settings.gsc_property

    if not gsc_path or not gsc_property:
        return {"available": False, "message": "GSC 未配置。请在 .env 中设置 SI_GOOGLE_CREDENTIALS_PATH 和 SI_GSC_PROPERTY"}

    try:
        from src.integrations.google_search_console import GoogleSearchConsole
        gsc = GoogleSearchConsole(credentials_path=gsc_path, site_url=gsc_property)
        if not gsc.available:
            return {"available": False, "message": "GSC 凭证无效或 API 未安装（需要 google-api-python-client）"}

        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=30)).isoformat()

        positions = await gsc.get_average_position(start_date, end_date)
        ctr = await gsc.get_ctr(start_date, end_date)

        # Merge data
        pages = []
        all_urls = set(positions.keys()) | set(ctr.keys())
        for url in sorted(all_urls)[:20]:
            pages.append({
                "url": url,
                "position": round(positions.get(url, 0), 1),
                "ctr": round(ctr.get(url, 0) * 100, 1),
            })

        return {
            "available": True,
            "period": f"{start_date} ~ {end_date}",
            "pages": pages,
            "total_pages_tracked": len(all_urls),
        }
    except Exception as e:
        logger.error(f"GSC fetch failed: {e}", exc_info=True)
        return {"available": True, "error": str(e)[:200]}
