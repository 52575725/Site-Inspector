from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import Settings
from src.presentation.issue_explainer import build_fix_preview
from src.storage.database import _upgrade_sqlite_schema
from src.storage.models import Fix, Issue, PageScan, Scan, Target
from src.web.routes.fix_actions import router


@pytest.mark.asyncio
async def test_fix_requires_approval_before_apply(test_db, tmp_path):
    factory = async_sessionmaker(test_db, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        target = Target(name="approval-test", base_url="https://example.com")
        session.add(target)
        await session.flush()
        scan = Scan(target_id=target.id, scan_type="quick")
        session.add(scan)
        await session.flush()
        page = PageScan(scan_id=scan.id, url="https://example.com/", http_status=200)
        session.add(page)
        await session.flush()
        issue = Issue(
            scan_id=scan.id,
            page_scan_id=page.id,
            url=page.url,
            inspector="seo",
            category="missing_meta_description",
            title="Missing description",
            status="proposed",
        )
        session.add(issue)
        await session.flush()
        fix = Fix(
            issue_id=issue.id,
            scan_id=scan.id,
            fixer="meta_fixer",
            fix_type="fully_auto",
            status="proposed",
            risk_level="low",
            file_path="index.html",
            before_content="<html><head></head><body>Home</body></html>",
            after_content='<html><head><meta name="description" content="Clear summary"></head><body>Home</body></html>',
        )
        session.add(fix)
        await session.commit()
        fix_id = fix.id

    app = FastAPI()
    app.include_router(router)
    app.state.session_factory = factory
    app.state.settings = Settings(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/fixes/{fix_id}/apply")
        assert response.status_code == 409

        response = await client.post(f"/api/fixes/{fix_id}/approve")
        assert response.status_code == 200
        assert response.json()["status"] == "approved"

        response = await client.post(f"/api/fixes/{fix_id}/apply")
        assert response.status_code == 200
        assert response.json()["status"] == "applied"

    assert (tmp_path / "fixed" / str(scan.id) / "index.html").read_text(
        encoding="utf-8"
    ).find("Clear summary") >= 0


def test_preview_translates_meta_change():
    issue = Issue(
        id=1,
        scan_id=1,
        page_scan_id=1,
        url="https://example.com/",
        inspector="seo",
        category="missing_meta_description",
        title="Missing description",
    )
    fix = Fix(
        id=1,
        issue_id=1,
        scan_id=1,
        fixer="meta_fixer",
        fix_type="fully_auto",
        status="proposed",
        risk_level="low",
        file_path="index.html",
        before_content="<html><head></head></html>",
        after_content='<html><head><meta name="description" content="Clear summary"></head></html>',
    )
    fix.issue = issue
    preview = build_fix_preview(fix)
    assert preview["before"] == "没有搜索结果描述"
    assert preview["after"] == "Clear summary"
    assert preview["status_label"] == "等待批准"


def test_preview_escalates_large_diff_to_high_risk():
    issue = Issue(
        id=2, scan_id=1, page_scan_id=1, url="https://example.com/",
        inspector="seo", category="missing_h1", title="Missing heading",
    )
    changed_lines = "\n".join([f"-old {index}\n+new {index}" for index in range(25)])
    fix = Fix(
        id=2, issue_id=2, scan_id=1, fixer="heading", fix_type="semi_auto",
        status="proposed", risk_level="medium", file_path="index.html",
        before_content="<h2>Old</h2>", after_content="<h1>New</h1>", diff=changed_lines,
    )
    fix.issue = issue
    preview = build_fix_preview(fix, include_developer=False)
    assert preview["risk"] == "high"
    assert preview["warning"]
    assert "diff" not in preview


@pytest.mark.asyncio
async def test_upgrade_adds_approval_columns(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'old.db'}")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE issues (id INTEGER PRIMARY KEY, status VARCHAR(30))"))
        await conn.execute(text(
            "CREATE TABLE fixes (id INTEGER PRIMARY KEY, issue_id INTEGER, scan_id INTEGER, "
            "fixer VARCHAR(50), fix_type VARCHAR(20), applied_at DATETIME)"
        ))
        await _upgrade_sqlite_schema(conn)
        result = await conn.exec_driver_sql("PRAGMA table_info(fixes)")
        columns = {row[1] for row in result.fetchall()}
    await engine.dispose()
    assert {"status", "plain_summary", "risk_level", "approved_at", "rejected_at"} <= columns
