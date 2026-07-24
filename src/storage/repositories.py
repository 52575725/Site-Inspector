from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import AuditLog, Fix, Issue, PageScan, Scan, Target, Verification


class TargetRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, name: str, base_url: str, **kwargs) -> Target:
        result = await self.session.execute(select(Target).where(Target.name == name))
        target = result.scalar_one_or_none()
        if target is None:
            target = Target(name=name, base_url=base_url, **kwargs)
            self.session.add(target)
            await self.session.flush()
        return target

    async def get_by_name(self, name: str) -> Optional[Target]:
        result = await self.session.execute(select(Target).where(Target.name == name))
        return result.scalar_one_or_none()


class ScanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, target_id: int, scan_type: str = "daily") -> Scan:
        scan = Scan(target_id=target_id, scan_type=scan_type)
        self.session.add(scan)
        await self.session.flush()
        return scan

    async def complete(self, scan_id: int, pages_crawled: int, total_issues: int) -> None:
        await self.session.execute(
            update(Scan)
            .where(Scan.id == scan_id)
            .values(
                status="completed",
                completed_at=datetime.utcnow(),
                pages_crawled=pages_crawled,
                total_issues_found=total_issues,
            )
        )

    async def fail(self, scan_id: int) -> None:
        await self.session.execute(
            update(Scan)
            .where(Scan.id == scan_id)
            .values(status="failed", completed_at=datetime.utcnow())
        )

    async def set_phase(self, scan_id: int, phase: str) -> None:
        await self.session.execute(
            update(Scan).where(Scan.id == scan_id).values(phase=phase)
        )

    async def fail_with_error(self, scan_id: int, error_message: str) -> None:
        await self.session.execute(
            update(Scan)
            .where(Scan.id == scan_id)
            .values(status="failed", completed_at=datetime.utcnow(),
                    error_message=error_message)
        )

    async def get_by_id(self, scan_id: int) -> Optional[Scan]:
        result = await self.session.execute(select(Scan).where(Scan.id == scan_id))
        return result.scalar_one_or_none()

    async def get_latest(self, target_id: int) -> Optional[Scan]:
        result = await self.session.execute(
            select(Scan)
            .where(Scan.target_id == target_id)
            .order_by(Scan.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def set_fix_error(self, scan_id: int, error: str) -> None:
        await self.session.execute(
            update(Scan).where(Scan.id == scan_id).values(fix_error=error)
        )

    async def set_pr_url(self, scan_id: int, pr_url: str) -> None:
        await self.session.execute(
            update(Scan).where(Scan.id == scan_id).values(pr_url=pr_url)
        )

    async def get_latest_any(self) -> Optional[Scan]:
        result = await self.session.execute(
            select(Scan)
            .order_by(Scan.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


class PageScanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, scan_id: int, **kwargs) -> PageScan:
        ps = PageScan(scan_id=scan_id, **kwargs)
        self.session.add(ps)
        await self.session.flush()
        return ps

    async def bulk_create(self, scan_id: int, pages: list[dict]) -> list[PageScan]:
        instances = [PageScan(scan_id=scan_id, **p) for p in pages]
        self.session.add_all(instances)
        await self.session.flush()
        return instances


class IssueRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Issue:
        issue = Issue(**kwargs)
        self.session.add(issue)
        await self.session.flush()
        return issue

    async def bulk_create(self, issues: list[dict]) -> list[Issue]:
        instances = [Issue(**i) for i in issues]
        self.session.add_all(instances)
        await self.session.flush()
        return instances

    async def get_by_scan(self, scan_id: int) -> Sequence[Issue]:
        result = await self.session.execute(
            select(Issue).where(Issue.scan_id == scan_id)
        )
        return result.scalars().all()

    async def get_open_by_url(self, url: str) -> Sequence[Issue]:
        result = await self.session.execute(
            select(Issue).where(Issue.url == url, Issue.status == "open")
        )
        return result.scalars().all()

    async def update_status(self, issue_id: int, status: str) -> None:
        await self.session.execute(
            update(Issue).where(Issue.id == issue_id).values(status=status)
        )


class FixRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Fix:
        fix = Fix(**kwargs)
        self.session.add(fix)
        await self.session.flush()
        return fix

    async def get_by_scan(self, scan_id: int) -> Sequence[Fix]:
        result = await self.session.execute(select(Fix).where(Fix.scan_id == scan_id))
        return result.scalars().all()

    async def mark_pr_created(self, fix_id: int, pr_url: str) -> None:
        await self.session.execute(
            update(Fix)
            .where(Fix.id == fix_id)
            .values(git_pr_url=pr_url, status="pr_created")
        )


class VerificationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Verification:
        v = Verification(**kwargs)
        self.session.add(v)
        await self.session.flush()
        return v

    async def get_pending(self) -> Sequence[Verification]:
        result = await self.session.execute(
            select(Verification).where(Verification.status == "pending")
        )
        return result.scalars().all()

    async def update_status(self, verification_id: int, status: str, value_after: float) -> None:
        await self.session.execute(
            update(Verification)
            .where(Verification.id == verification_id)
            .values(status=status, value_after=value_after)
        )


class AuditLogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(self, action: str, entity_type: str | None = None,
                  entity_id: int | None = None, details: dict | None = None) -> None:
        entry = AuditLog(
            action=action, entity_type=entity_type,
            entity_id=entity_id, details=details,
        )
        self.session.add(entry)
        await self.session.flush()
