from __future__ import annotations

import math

from fastapi import APIRouter, Request
from sqlalchemy import select, func

from src.storage.models import Fix, Issue, Scan
from src.storage.repositories import FixRepository, IssueRepository, ScanRepository
from src.web.deps import get_db, templates

router = APIRouter(tags=["dashboard"])

PRIORITY_WEIGHTS = {"P0": 8, "P1": 4, "P2": 2, "P3": 1}

DIMENSION_LABELS = {
    "seo": "SEO",
    "accessibility": "Accessibility",
    "performance": "Performance",
    "mobile": "Mobile",
    "content_quality": "Content Quality",
    "broken_links": "Broken Links",
    "sitemap": "Sitemap",
    "structured_data": "Structured Data",
    "content_gap": "Content Gap",
    "headers": "HTTP Headers",
    "cannibalization": "Cannibalization",
    "js_seo": "JavaScript SEO",
    "eeat": "E-E-A-T",
    "crawl_budget": "Crawl Budget",
    "url_audit": "URL Audit",
    "content_freshness": "Content Freshness",
    "image_seo": "Image SEO",
    "keyword_analyzer": "Keywords",
    "robots_txt": "Robots.txt",
    "platform_seo": "Platform SEO",
}


@router.get("/")
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@router.get("/api/dashboard/summary")
async def dashboard_summary(request: Request):
    factory = get_db(request)
    async with factory() as session:
        scan_repo = ScanRepository(session)
        issue_repo = IssueRepository(session)
        fix_repo = FixRepository(session)

        total_scans = (await session.execute(
            select(func.count(Scan.id))
        )).scalar() or 0

        total_issues = (await session.execute(
            select(func.count(Issue.id))
        )).scalar() or 0

        total_fixes = (await session.execute(
            select(func.count(Fix.id))
        )).scalar() or 0

        p0_count = (await session.execute(
            select(func.count(Issue.id)).where(Issue.priority_tier == "P0")
        )).scalar() or 0

        p1_count = (await session.execute(
            select(func.count(Issue.id)).where(Issue.priority_tier == "P1")
        )).scalar() or 0

        latest_scan = await scan_repo.get_latest_any()
        latest_info = None
        if latest_scan:
            latest_info = {
                "id": latest_scan.id,
                "date": latest_scan.started_at.isoformat() if latest_scan.started_at else None,
                "status": latest_scan.status,
                "pages_crawled": latest_scan.pages_crawled,
                "total_issues_found": latest_scan.total_issues_found,
            }

        return {
            "total_scans": total_scans,
            "total_issues": total_issues,
            "total_fixes": total_fixes,
            "p0_count": p0_count,
            "p1_count": p1_count,
            "latest_scan": latest_info,
        }


@router.get("/api/dashboard/scores")
async def dashboard_scores(request: Request):
    factory = get_db(request)
    async with factory() as session:
        result = await session.execute(
            select(Issue.inspector, Issue.priority_tier, func.count(Issue.id))
            .where(Issue.status == "open")
            .group_by(Issue.inspector, Issue.priority_tier)
        )
        rows = result.all()

        by_inspector: dict[str, dict[str, int]] = {}
        for inspector, tier, count in rows:
            by_inspector.setdefault(inspector, {})[tier] = count

        scores = []
        for inspector, tier_counts in by_inspector.items():
            total = sum(tier_counts.values())
            penalty = sum(
                PRIORITY_WEIGHTS.get(tier, 1) * math.log2(1 + count)
                for tier, count in tier_counts.items()
            )
            score = max(5, round(100 - penalty))

            if score >= 70:
                status = "good"
            elif score >= 40:
                status = "warn"
            else:
                status = "bad"

            scores.append({
                "inspector": inspector,
                "label": DIMENSION_LABELS.get(inspector, inspector),
                "score": score,
                "status": status,
                "issue_count": total,
            })

        # Also include dimensions with zero issues
        for key, label in DIMENSION_LABELS.items():
            if key not in by_inspector:
                scores.append({
                    "inspector": key,
                    "label": label,
                    "score": 98,
                    "status": "good",
                    "issue_count": 0,
                })

        scores.sort(key=lambda s: s["score"])
        return scores


@router.get("/api/dashboard/trend")
async def dashboard_trend(request: Request):
    factory = get_db(request)
    async with factory() as session:
        scans_result = await session.execute(
            select(Scan.id, Scan.started_at, Scan.total_issues_found, Scan.pages_crawled)
            .where(Scan.status == "completed")
            .order_by(Scan.started_at.desc())
            .limit(10)
        )
        scans = list(scans_result.all())

        # Fetch all fix counts in a single grouped query
        scan_ids = [s[0] for s in scans]
        fix_counts = {}
        if scan_ids:
            fix_result = await session.execute(
                select(Fix.scan_id, func.count(Fix.id))
                .where(Fix.scan_id.in_(scan_ids))
                .group_by(Fix.scan_id)
            )
            fix_counts = dict(fix_result.all())

        trend = []
        for scan_id, started_at, total_issues, pages in reversed(scans):
            trend.append({
                "scan_id": scan_id,
                "date": started_at.strftime("%m/%d") if started_at else "",
                "issues": total_issues or 0,
                "fixes": fix_counts.get(scan_id, 0),
                "pages": pages or 0,
            })

        return trend
