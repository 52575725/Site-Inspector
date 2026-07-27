from types import SimpleNamespace

import pytest

from src.core.scan_orchestrator import ScanOrchestrator


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
