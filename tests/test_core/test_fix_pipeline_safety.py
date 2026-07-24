from __future__ import annotations

import pytest

from config.settings import Settings
from src.core.fix_orchestrator import FixOrchestrator
from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource
from src.storage.models import Issue, PageScan, Scan, Target


ORIGINAL = """<html><head><title>About</title></head><body><h1>About</h1><p>Original company content that must remain intact after validation.</p></body></html>"""
CORRUPTED = """<html><head><title>About</title></head><body><h1>About</h1><h1>About</h1><p>Original company content that must remain intact after validation.</p></body></html>"""


class MemorySource(BaseSource):
    source_type = "memory"

    def __init__(self, content: str):
        self.files = {"about/index.html": content}

    async def connect(self):
        return None

    async def sync(self):
        return None

    async def read_file(self, relative_path: str) -> str:
        return self.files[relative_path]

    async def write_file(self, relative_path: str, content: str) -> None:
        self.files[relative_path] = content

    async def list_files(self, pattern: str = "*") -> list[str]:
        return list(self.files)

    async def disconnect(self):
        return None


class CorruptingFixer(BaseFixer):
    fixer_name = "corrupting_fixer"
    fix_type = "fully_auto"
    supported_categories = ["test_corruption"]

    async def generate_fix(self, issue, source, page_content):
        return FixResult(
            success=True,
            issue_id=issue["id"],
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue["file_path"],
            before_content=page_content,
            after_content=CORRUPTED,
            diff="duplicate h1",
        )


class ReviewOnlyFixer(CorruptingFixer):
    fixer_name = "review_only_fixer"
    fix_type = "semi_auto"


async def _issue(test_session) -> Issue:
    target = Target(name="pipeline-test", base_url="https://example.com")
    test_session.add(target)
    await test_session.flush()
    scan = Scan(target_id=target.id, scan_type="quick")
    test_session.add(scan)
    await test_session.flush()
    page = PageScan(scan_id=scan.id, url="https://example.com/about/", http_status=200)
    test_session.add(page)
    await test_session.flush()
    issue = Issue(
        scan_id=scan.id,
        page_scan_id=page.id,
        url=page.url,
        inspector="seo",
        category="test_corruption",
        title="Pipeline safety test",
        priority_tier="P0",
    )
    test_session.add(issue)
    await test_session.flush()
    return issue


@pytest.mark.asyncio
async def test_source_override_validates_and_rolls_back(test_session):
    issue = await _issue(test_session)
    source = MemorySource(ORIGINAL)
    orchestrator = FixOrchestrator(Settings(), test_session)
    orchestrator.fixers = [CorruptingFixer()]

    fixes = await orchestrator.run_fixes([issue], source_override=source)

    assert fixes == []
    assert source.files["about/index.html"] == ORIGINAL
    assert issue.status == "open"


@pytest.mark.asyncio
async def test_repository_mode_skips_review_only_fixer(test_session):
    issue = await _issue(test_session)
    source = MemorySource(ORIGINAL)
    orchestrator = FixOrchestrator(Settings(), test_session)
    orchestrator.fixers = [ReviewOnlyFixer()]

    fixes = await orchestrator.run_fixes([issue], source_override=source)

    assert fixes == []
    assert source.files["about/index.html"] == ORIGINAL
