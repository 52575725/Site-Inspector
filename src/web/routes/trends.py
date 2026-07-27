"""Trend dashboard API — aggregated scan history and metrics."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from src.storage.database import get_session_factory
from src.storage.models import Fix, Issue, Scan
from src.web.deps import templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trends"])


@router.get("/trends")
async def trends_page(request: Request):
    """Render the trends dashboard page."""
    return templates.TemplateResponse(request, "trends.html")


@router.get("/api/trends/overview")
async def trends_overview(request: Request):
    """Get scan history overview for trend charts.

    Returns:
        scans: list of {date, pages_crawled, new_issues, fixes_applied}
        top_categories: most common issue categories over time
        health_score: simplified SEO health score per scan
    """
    settings = request.app.state.settings
    session_factory = get_session_factory(settings)

    async with session_factory() as session:
        # Get scan history (last 30 days, max 50 scans)
        cutoff = datetime.utcnow() - timedelta(days=30)
        result = await session.execute(
            select(Scan)
            .where(Scan.completed_at is not None)
            .where(Scan.completed_at >= cutoff)
            .order_by(Scan.completed_at.desc())
            .limit(50)
        )
        scans = result.scalars().all()

        # Build scan timeline
        scan_data = []
        for scan in reversed(scans):
            completed = scan.completed_at.isoformat() if scan.completed_at else None
            scan_data.append({
                "id": scan.id,
                "date": completed,
                "pages_crawled": scan.pages_crawled or 0,
                "new_issues": scan.total_issues_found or 0,
                "status": scan.status,
                # Health score: more pages with fewer issues = better
                "health_score": _calc_health(scan),
            })

        # Get top issue categories
        top_cats_result = await session.execute(
            select(Issue.category, func.count(Issue.id).label("cnt"))
            .where(Issue.scan_id.in_([s.id for s in scans]))
            .group_by(Issue.category)
            .order_by(func.count(Issue.id).desc())
            .limit(10)
        )
        top_categories = [
            {"category": row[0], "count": row[1]}
            for row in top_cats_result.all()
        ]

        # Get fix success rate
        fix_result = await session.execute(
            select(Fix.status, func.count(Fix.id).label("cnt"))
            .where(Fix.scan_id.in_([s.id for s in scans]))
            .group_by(Fix.status)
        )
        fix_stats = {row[0]: row[1] for row in fix_result.all()}

    return {
        "scans": scan_data,
        "top_categories": top_categories,
        "fix_stats": fix_stats,
        "total_scans": len(scan_data),
        "trend_summary": _make_summary(scan_data, fix_stats),
    }


@router.get("/api/trends/issues-by-category")
async def issues_by_category(request: Request, scan_id: int | None = None):
    """Get issue breakdown by category for a specific scan or latest."""
    settings = request.app.state.settings
    session_factory = get_session_factory(settings)

    async with session_factory() as session:
        if scan_id is None:
            # Get latest scan
            result = await session.execute(
                select(Scan)
                .where(Scan.completed_at is not None)
                .order_by(Scan.completed_at.desc())
                .limit(1)
            )
            scan = result.scalar_one_or_none()
            if scan is None:
                return {"categories": []}
            scan_id = scan.id

        result = await session.execute(
            select(Issue.category, func.count(Issue.id).label("cnt"),
                   Issue.priority_tier)
            .where(Issue.scan_id == scan_id)
            .group_by(Issue.category, Issue.priority_tier)
            .order_by(func.count(Issue.id).desc())
        )
        categories = [
            {"category": row[0], "count": row[1], "priority": row[2] or "P2"}
            for row in result.all()
        ]

    return {"scan_id": scan_id, "categories": categories}


def _calc_health(scan: Scan) -> float:
    """Simplified SEO health score: higher = better."""
    pages = max(scan.pages_crawled or 1, 1)
    issues = scan.total_issues_found or 0
    # Fewer issues per page = better health
    ratio = issues / pages
    if ratio == 0:
        return 100.0
    if ratio >= 10:
        return 0.0
    return max(0.0, 100.0 - ratio * 10)


def _make_summary(scan_data: list, fix_stats: dict) -> str:
    """Human-readable trend summary."""
    if not scan_data:
        return "No scan data available yet."
    latest = scan_data[0]
    applied = fix_stats.get("applied", 0)
    proposed = fix_stats.get("proposed", 0)
    failed = fix_stats.get("validation_failed", 0)

    parts = [
        f"Latest scan: {latest.get('date', 'N/A')[:10]}",
        f"Pages: {latest.get('pages_crawled', 0)}",
        f"Issues: {latest.get('new_issues', 0)}",
        f"Applied fixes: {applied}",
    ]
    if proposed:
        parts.append(f"Pending: {proposed}")
    return " | ".join(parts)
