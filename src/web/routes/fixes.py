from __future__ import annotations

from fastapi import APIRouter, Query, Request
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

from src.storage.models import Fix, Verification
from src.presentation.issue_explainer import build_fix_preview
from src.web.deps import get_db, templates

router = APIRouter(tags=["fixes"])


@router.get("/fixes")
async def fixes_page(request: Request):
    return templates.TemplateResponse(request, "fixes.html")


@router.get("/api/fixes")
async def api_fixes(
    request: Request,
    scan_id: int | None = Query(None),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    factory = get_db(request)
    async with factory() as session:
        stmt = select(Fix).options(
            joinedload(Fix.issue), joinedload(Fix.verifications)
        )

        if scan_id:
            stmt = stmt.where(Fix.scan_id == scan_id)
        if status:
            stmt = stmt.where(Fix.status == status)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await session.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(Fix.applied_at.desc()).offset((page - 1) * limit).limit(limit)
        result = await session.execute(stmt)
        fixes = result.unique().scalars().all()

        return {
            "items": [build_fix_preview(f, include_developer=False) for f in fixes],
            "total": total,
            "page": page,
            "pages": max(1, (total + limit - 1) // limit),
        }


@router.get("/api/verifications")
async def api_verifications(request: Request):
    factory = get_db(request)
    async with factory() as session:
        result = await session.execute(
            select(Verification).order_by(Verification.window_start.desc()).limit(100)
        )
        verifications = result.scalars().all()

        return [
            {
                "id": v.id,
                "fix_id": v.fix_id,
                "metric_name": v.metric_name,
                "value_before": v.value_before,
                "value_after": v.value_after,
                "status": v.status,
                "window_start": v.window_start.isoformat() if v.window_start else None,
            }
            for v in verifications
        ]


def _agg_verification_status(verifications: list) -> str:
    if not verifications:
        return "none"
    statuses = {v.status for v in verifications}
    if "degraded" in statuses:
        return "degraded"
    if "improved" in statuses:
        return "improved"
    if "pending" in statuses:
        return "pending"
    return "unchanged"
