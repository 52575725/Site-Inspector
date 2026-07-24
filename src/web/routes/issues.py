from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

from src.storage.models import Fix, Issue, Verification
from src.web.deps import get_db, templates

router = APIRouter(tags=["issues"])

INSPECTOR_LABELS = {
    "seo": "SEO", "accessibility": "Accessibility", "performance": "Performance",
    "mobile": "Mobile", "content_quality": "Content Quality", "broken_links": "Broken Links",
    "sitemap": "Sitemap", "structured_data": "Structured Data", "content_gap": "Content Gap",
    "headers": "HTTP Headers", "cannibalization": "Cannibalization", "js_seo": "JavaScript SEO",
    "eeat": "E-E-A-T", "crawl_budget": "Crawl Budget", "url_audit": "URL Audit",
    "content_freshness": "Content Freshness", "image_seo": "Image SEO",
    "keyword_analyzer": "Keywords", "robots_txt": "Robots.txt", "platform_seo": "Platform SEO",
}


@router.get("/issues")
async def issues_page(request: Request):
    return templates.TemplateResponse(request, "issues.html", {
        "inspector_labels": INSPECTOR_LABELS,
    })


@router.get("/issues/{issue_id}")
async def issue_detail_page(request: Request, issue_id: int):
    factory = get_db(request)
    async with factory() as session:
        result = await session.execute(
            select(Issue).options(
                joinedload(Issue.fixes).joinedload(Fix.verifications)
            ).where(Issue.id == issue_id)
        )
        issue = result.unique().scalar_one_or_none()

        if not issue:
            return templates.TemplateResponse(request, "issue_detail.html", {
                "error": "Issue not found",
            })

        return templates.TemplateResponse(request, "issue_detail.html", {
            "issue": issue,
            "fixes": issue.fixes,
            "inspector_labels": INSPECTOR_LABELS,
        })


@router.get("/api/issues")
async def api_issues(
    request: Request,
    scan_id: int | None = Query(None),
    inspector: str | None = Query(None),
    priority: str | None = Query(None),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    factory = get_db(request)
    async with factory() as session:
        stmt = select(Issue).options(joinedload(Issue.fixes))

        if scan_id:
            stmt = stmt.where(Issue.scan_id == scan_id)
        if inspector:
            stmt = stmt.where(Issue.inspector == inspector)
        if priority:
            stmt = stmt.where(Issue.priority_tier == priority)
        if status:
            stmt = stmt.where(Issue.status == status)

        # Count total
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await session.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(Issue.priority_score.desc()).offset((page - 1) * limit).limit(limit)
        result = await session.execute(stmt)
        issues = result.unique().scalars().all()

        return {
            "items": [
                {
                    "id": i.id,
                    "url": i.url,
                    "inspector": i.inspector,
                    "category": i.category,
                    "priority_tier": i.priority_tier,
                    "priority_score": round(i.priority_score, 2) if i.priority_score else 0,
                    "title": i.title,
                    "status": i.status,
                    "first_seen_at": i.first_seen_at.isoformat() if i.first_seen_at else None,
                    "fix_count": len(i.fixes),
                }
                for i in issues
            ],
            "total": total,
            "page": page,
            "pages": max(1, (total + limit - 1) // limit),
        }


@router.get("/api/issues/{issue_id}")
async def api_issue_detail(request: Request, issue_id: int):
    factory = get_db(request)
    async with factory() as session:
        result = await session.execute(
            select(Issue).options(
                joinedload(Issue.fixes).joinedload(Fix.verifications)
            ).where(Issue.id == issue_id)
        )
        issue = result.unique().scalar_one_or_none()

        if not issue:
            return {"error": "Issue not found"}

        fixes_data = []
        for fix in issue.fixes:
            verifications = [
                {
                    "id": v.id,
                    "metric_name": v.metric_name,
                    "value_before": v.value_before,
                    "value_after": v.value_after,
                    "status": v.status,
                }
                for v in fix.verifications
            ]
            fixes_data.append({
                "id": fix.id,
                "fixer": fix.fixer,
                "fix_type": fix.fix_type,
                "file_path": fix.file_path,
                "diff": fix.diff,
                "git_pr_url": fix.git_pr_url,
                "applied_at": fix.applied_at.isoformat() if fix.applied_at else None,
                "verifications": verifications,
            })

        return {
            "id": issue.id,
            "url": issue.url,
            "inspector": issue.inspector,
            "category": issue.category,
            "priority_tier": issue.priority_tier,
            "priority_score": round(issue.priority_score, 2) if issue.priority_score else 0,
            "title": issue.title,
            "description": issue.description,
            "element": issue.element,
            "current_value": issue.current_value,
            "suggested_value": issue.suggested_value,
            "status": issue.status,
            "first_seen_at": issue.first_seen_at.isoformat() if issue.first_seen_at else None,
            "fix_applied_at": issue.fix_applied_at.isoformat() if issue.fix_applied_at else None,
            "fixes": fixes_data,
        }
