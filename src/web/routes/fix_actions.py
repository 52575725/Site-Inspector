from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.presentation.issue_explainer import STATUS_LABELS, build_fix_preview
from src.sources.base import resolve_within
from src.storage.models import Fix
from src.web.deps import get_db, templates

router = APIRouter(tags=["fix-actions"])


@router.get("/fixes")
async def fixes_page(request: Request):
    return RedirectResponse(url="/", status_code=307)


@router.get("/fixes/{fix_id}")
async def fix_detail_page(request: Request, fix_id: int):
    factory = get_db(request)
    async with factory() as session:
        fix = await _get_fix(session, fix_id)
        if not fix:
            return templates.TemplateResponse(
                request, "fix_detail.html", {"error": "没有找到这条修复建议"}, status_code=404,
            )
        return templates.TemplateResponse(
            request, "fix_detail.html", {"fix": fix, "preview": build_fix_preview(fix)},
        )


@router.get("/api/fixes/{fix_id}/preview")
async def api_fix_preview(request: Request, fix_id: int):
    factory = get_db(request)
    async with factory() as session:
        fix = await _get_fix(session, fix_id)
        if not fix:
            raise HTTPException(status_code=404, detail="Fix not found")
        return build_fix_preview(fix)


@router.post("/api/fixes/{fix_id}/approve")
async def approve_fix(request: Request, fix_id: int):
    factory = get_db(request)
    async with factory() as session:
        fix = await _get_fix(session, fix_id)
        if not fix:
            raise HTTPException(status_code=404, detail="Fix not found")
        if fix.status not in {"proposed", "rejected"}:
            raise HTTPException(status_code=409, detail="This fix can no longer be approved")
        fix.status = "approved"
        fix.approved_at = datetime.utcnow()
        fix.rejected_at = None
        fix.issue.status = "approved"
        await session.commit()
        return {"status": fix.status, "status_label": STATUS_LABELS[fix.status]}


@router.post("/api/fixes/{fix_id}/reject")
async def reject_fix(request: Request, fix_id: int):
    factory = get_db(request)
    async with factory() as session:
        fix = await _get_fix(session, fix_id)
        if not fix:
            raise HTTPException(status_code=404, detail="Fix not found")
        if fix.status not in {"proposed", "approved"}:
            raise HTTPException(status_code=409, detail="This fix can no longer be rejected")
        fix.status = "rejected"
        fix.rejected_at = datetime.utcnow()
        fix.approved_at = None
        fix.issue.status = "rejected"
        await session.commit()
        return {"status": fix.status, "status_label": STATUS_LABELS[fix.status]}


@router.post("/api/fixes/{fix_id}/apply")
async def apply_fix(request: Request, fix_id: int):
    factory = get_db(request)
    settings = request.app.state.settings
    async with factory() as session:
        fix = await _get_fix(session, fix_id)
        if not fix:
            raise HTTPException(status_code=404, detail="Fix not found")
        if fix.status != "approved":
            raise HTTPException(status_code=409, detail="Approve this fix before applying it")
        if not fix.file_path or fix.after_content is None:
            raise HTTPException(status_code=409, detail="This fix has no writable content")

        dependency_result = await session.execute(
            select(Fix).options(joinedload(Fix.issue)).where(
                Fix.scan_id == fix.scan_id,
                Fix.file_path == fix.file_path,
                Fix.id < fix.id,
            ).order_by(Fix.id)
        )
        dependencies = list(dependency_result.scalars().all())
        blockers = [item for item in dependencies if item.status not in {"approved", "applied"}]
        if blockers:
            raise HTTPException(
                status_code=409,
                detail="同一文件有更早的修复建议尚未批准，请先处理这些建议",
            )

        output_dir = (settings.data_dir / "fixed" / str(fix.scan_id)).resolve()
        output_path = resolve_within(output_dir, fix.file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(fix.after_content, encoding="utf-8")

        fix.status = "applied"
        fix.applied_at = datetime.utcnow()
        fix.issue.status = "applied"
        fix.issue.fix_applied_at = fix.applied_at
        for dependency in dependencies:
            if dependency.status == "approved":
                dependency.status = "applied"
                dependency.applied_at = fix.applied_at
                dependency.issue.status = "applied"
                dependency.issue.fix_applied_at = fix.applied_at
        await session.commit()
        return {
            "status": fix.status,
            "status_label": STATUS_LABELS[fix.status],
            "output_path": str(output_path),
        }


async def _get_fix(session, fix_id: int) -> Fix | None:
    result = await session.execute(
        select(Fix)
        .options(joinedload(Fix.issue), joinedload(Fix.verifications))
        .where(Fix.id == fix_id)
    )
    return result.unique().scalar_one_or_none()
