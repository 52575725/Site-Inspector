from types import SimpleNamespace

import pytest

from src.core.scan_orchestrator import ScanOrchestrator
from src.inspectors.base import RawFinding


@pytest.mark.asyncio
async def test_quick_scan_loads_target_config_inside_run_full_scan(monkeypatch):
    loaded_targets: list[str] = []

    class FakeSettings:
        target_name = "configured"
        target_base_url = "https://configured.example"
        target_languages = ["en"]
        source_type = "http"

        @classmethod
        def load_target(cls, name):
            loaded_targets.append(name)
            return {}

    class FakeTargetRepository:
        async def get_or_create(self, **kwargs):
            return SimpleNamespace(id=1)

    class FakeScanRepository:
        async def get_by_id(self, scan_id):
            return SimpleNamespace(id=scan_id, status="running")

        async def set_phase(self, scan_id, phase):
            return None

        async def fail(self, scan_id):
            return None

    class FakeAuditRepository:
        async def log(self, *args):
            return None

    class FakeSession:
        async def commit(self):
            return None

    class FakeCrawler:
        def __init__(self, settings, *, base_url, use_browser):
            assert base_url == "https://quick.example"
            assert use_browser is False

        async def discover_pages(self):
            return []

        async def close(self):
            return None

    monkeypatch.setattr("src.core.scan_orchestrator.Crawler", FakeCrawler)
    orchestrator = object.__new__(ScanOrchestrator)
    orchestrator.settings = FakeSettings()
    orchestrator.session = FakeSession()
    orchestrator.target_repo = FakeTargetRepository()
    orchestrator.scan_repo = FakeScanRepository()
    orchestrator.audit_repo = FakeAuditRepository()
    orchestrator._last_crawled_pages = []

    scan = await orchestrator.run_full_scan(
        target_name="quick-example",
        target_base_url="https://quick.example",
        target_languages=["en"],
        existing_scan_id=7,
    )

    assert scan.id == 7
    assert loaded_targets == ["quick-example"]


def test_site_findings_are_grouped_with_affected_url_evidence():
    findings = [
        RawFinding(
            url="https://example.com/a",
            inspector="headers",
            category="missing_content_security_policy",
            description="Missing CSP",
            scope="site",
            group_key="missing_content_security_policy",
        ),
        RawFinding(
            url="https://example.com/b",
            inspector="headers",
            category="missing_content_security_policy",
            description="Missing CSP",
            scope="site",
            group_key="missing_content_security_policy",
        ),
        RawFinding(
            url="https://example.com/a",
            inspector="seo",
            category="missing_title",
            description="Missing title",
        ),
    ]

    grouped = ScanOrchestrator._aggregate_findings(findings)

    assert len(grouped) == 2
    header_finding = next(item for item in grouped if item.inspector == "headers")
    assert header_finding.raw_metadata["affected_url_count"] == 2
    assert header_finding.raw_metadata["affected_urls"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert "Affects 2 scanned URLs" in header_finding.description


def test_inspection_errors_are_grouped_outside_seo_findings():
    errors = [
        {"inspector": "competitor_gap", "stage": "inspect", "message": "boom",
         "url": "https://example.com/a"},
        {"inspector": "competitor_gap", "stage": "inspect", "message": "boom",
         "url": "https://example.com/b"},
    ]

    summary = ScanOrchestrator._summarize_inspection_errors(errors)

    assert summary == [{
        "inspector": "competitor_gap",
        "stage": "inspect",
        "message": "boom",
        "count": 2,
        "sample_urls": ["https://example.com/a", "https://example.com/b"],
    }]


def test_non_html_resources_are_excluded_from_page_inspectors():
    html_page = SimpleNamespace(headers={"Content-Type": "text/html; charset=utf-8"})
    feed = SimpleNamespace(headers={"content-type": "application/rss+xml"})
    unknown = SimpleNamespace(headers={})

    assert ScanOrchestrator._is_html_page(html_page)
    assert not ScanOrchestrator._is_html_page(feed)
    assert ScanOrchestrator._is_html_page(unknown)
