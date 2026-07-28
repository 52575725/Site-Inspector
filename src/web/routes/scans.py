from __future__ import annotations

import logging
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from src.storage.models import Scan, Target
from src.storage.repositories import ScanRepository, TargetRepository
from src.web.deps import get_db, templates
from src.web.security import validate_github_repo, validate_public_http_url

logger = logging.getLogger(__name__)
router = APIRouter(tags=["scans"])


class QuickScanRequest(BaseModel):
    url: str
    name: str | None = None
    repo_url: str | None = None
    repo_branch: str = "main"
    push_changes: bool = False


def resolve_quick_scan_target_name(url: str, explicit_name: str | None, settings) -> str:
    if explicit_name:
        return explicit_name
    requested_host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    configured = settings.__class__.load_target(settings.target_name)
    configured_url = configured.get("base_url") or settings.target_base_url
    configured_host = (
        (urlparse(configured_url).hostname or "").lower().removeprefix("www.")
    )
    if requested_host and requested_host == configured_host:
        return settings.target_name
    return requested_host.replace(".", "-") or "quick-scan"


@router.get("/scans")
async def scans_page(request: Request):
    return templates.TemplateResponse(request, "scans.html")


@router.get("/api/scans")
async def api_scans(request: Request):
    factory = get_db(request)
    async with factory() as session:
        result = await session.execute(
            select(Scan).order_by(Scan.started_at.desc()).limit(20)
        )
        scans = result.scalars().all()

        return [
            {
                "id": s.id,
                "scan_type": s.scan_type,
                "status": s.status,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "pages_crawled": s.pages_crawled,
                "total_issues_found": s.total_issues_found,
            }
            for s in scans
        ]


@router.post("/api/scan/trigger")
async def trigger_scan(request: Request, background_tasks: BackgroundTasks):
    """Trigger a new scan. Runs in background to avoid blocking."""
    settings = request.app.state.settings

    async def run_scan():
        from src.core.engine import Engine
        factory = request.app.state.session_factory
        engine = Engine(settings, factory)
        try:
            await engine.run_daily_scan()
        finally:
            await engine.close()

    background_tasks.add_task(run_scan)
    return {"status": "started", "message": "Scan started in background"}


@router.get("/api/scans/{scan_id}")
async def get_scan_detail(request: Request, scan_id: int):
    """Get detailed status of a single scan for progress polling."""
    factory = get_db(request)
    async with factory() as session:
        result = await session.execute(
            select(Scan).where(Scan.id == scan_id)
        )
        scan = result.scalar_one_or_none()
        if not scan:
            return {"error": "Scan not found"}

        t_result = await session.execute(
            select(Target).where(Target.id == scan.target_id)
        )
        target_obj = t_result.scalar_one_or_none()

        return {
            "id": scan.id,
            "scan_type": scan.scan_type,
            "status": scan.status,
            "phase": scan.phase,
            "error_message": scan.error_message,
            "fix_error": scan.fix_error,
            "pr_url": scan.pr_url,
            "pages_crawled": scan.pages_crawled,
            "total_issues_found": scan.total_issues_found,
            "started_at": scan.started_at.isoformat() if scan.started_at else None,
            "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
            "target_url": target_obj.base_url if target_obj else None,
        }


@router.post("/api/scans/{scan_id}/apply-fixes")
async def apply_fixes_for_scan(request: Request, scan_id: int):
    """Apply suggested fixes by writing the fixed HTML files to disk."""
    from datetime import datetime

    from src.sources.base import resolve_within
    from src.storage.repositories import FixRepository

    factory = request.app.state.session_factory
    settings = request.app.state.settings

    async with factory() as session:
        scan_repo = ScanRepository(session)
        scan = await scan_repo.get_by_id(scan_id)
        if not scan:
            return {"error": "Scan not found", "fixes_applied": 0, "files_written": 0, "output_dir": ""}

        fix_repo = FixRepository(session)
        all_fixes = await fix_repo.get_by_scan(scan_id)
        fixes = [fix for fix in all_fixes if fix.status == "approved"]
        if not fixes:
            return {"message": "请先在修复建议页面批准要应用的内容", "fixes_applied": 0, "files_written": 0, "output_dir": ""}
        approved_paths = {fix.file_path for fix in fixes}
        blockers = [
            fix for fix in all_fixes
            if fix.file_path in approved_paths and fix.status not in {"approved", "applied"}
        ]
        if blockers:
            raise HTTPException(
                status_code=409,
                detail="同一文件仍有未批准的修复建议，请先逐条处理",
            )

        # Group fixes by file_path; last fix per file has cumulative after_content
        fixed_files: dict[str, str] = {}
        for f in fixes:
            if f.after_content:
                fixed_files[f.file_path] = f.after_content

        if not fixed_files:
            return {"message": "No fix content available", "fixes_applied": len(fixes), "files_written": 0, "output_dir": ""}

        output_dir = (settings.data_dir / "fixed" / str(scan_id)).resolve()
        for file_path, content in fixed_files.items():
            full = resolve_within(output_dir, file_path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")

        applied_at = datetime.utcnow()
        for fix in fixes:
            fix.status = "applied"
            fix.applied_at = applied_at
            fix.issue.status = "applied"
            fix.issue.fix_applied_at = applied_at
        await session.commit()

    return {
        "message": f"Wrote {len(fixed_files)} fixed files",
        "fixes_applied": len(fixes),
        "files_written": len(fixed_files),
        "output_dir": str(output_dir),
    }


@router.post("/api/quick-scan")
async def trigger_quick_scan(request: Request,
                             body: QuickScanRequest,
                             background_tasks: BackgroundTasks):
    """Trigger an on-demand quick scan for an arbitrary URL."""
    url = await validate_public_http_url(body.url)

    factory = request.app.state.session_factory
    settings = request.app.state.settings
    target_name = resolve_quick_scan_target_name(url, body.name, settings)

    repo_url = None
    repo_branch = body.repo_branch
    if body.repo_url:
        if not settings.web_allow_repo_writes:
            raise HTTPException(status_code=403, detail="Repository writes are disabled")
        repo_url, repo_branch = validate_github_repo(body.repo_url, body.repo_branch)

    # Create target + scan synchronously so we can return scan_id
    async with factory() as session:
        active = (
            await session.execute(
                select(func.count()).select_from(Scan).where(Scan.status == "running")
            )
        ).scalar_one()
        if active >= settings.web_max_active_scans:
            raise HTTPException(status_code=429, detail="The scan concurrency limit was reached")

        target_repo = TargetRepository(session)
        target = await target_repo.get_or_create(
            name=target_name, base_url=url, source_type="http",
            languages=["en"],
        )
        scan_repo = ScanRepository(session)
        scan = await scan_repo.create(target_id=target.id, scan_type="quick")
        await scan_repo.set_phase(scan.id, "starting")
        await session.commit()
        scan_id = scan.id

    async def run_scan():
        from src.core.engine import Engine
        engine = Engine(settings, factory)
        try:
            await engine.run_quick_scan(
                url, name=target_name, scan_id=scan_id,
                repo_url=repo_url,
                repo_branch=repo_branch,
                push_changes=body.push_changes and settings.web_allow_repo_writes,
            )
        except Exception as e:
            logger.error(f"Quick scan failed for {url}: {e}")
            try:
                async with factory() as s:
                    sr = ScanRepository(s)
                    await sr.fail_with_error(scan_id, str(e)[:500])
                    await s.commit()
            except Exception:
                pass
        finally:
            await engine.close()

    background_tasks.add_task(run_scan)

    return {
        "status": "started",
        "scan_id": scan_id,
        "message": f"Quick scan started for {url}",
    }
