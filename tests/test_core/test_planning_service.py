from types import SimpleNamespace

import pytest

from src.agents.planning_context import PlanningContext
from src.core.planning_service import PlanningService


class FakeFixer:
    fixer_name = "canonical_fixer"
    fix_type = "fully_auto"
    supported_categories = ["missing_canonical"]


@pytest.mark.asyncio
async def test_planning_service_is_the_decision_gateway(tmp_path, monkeypatch):
    captured: dict[str, str] = {}

    class FakeCollector:
        def __init__(self, settings, session, *, target_name=None):
            captured["target_name"] = target_name

        async def collect(self, issues):
            return PlanningContext()

    monkeypatch.setattr(
        "src.core.planning_service.PlanningContextCollector",
        FakeCollector,
    )
    settings = SimpleNamespace(data_dir=tmp_path, target_name="default")
    issue = SimpleNamespace(
        id=1,
        url="https://example.com/products/bar/",
        category="missing_canonical",
        title="Missing canonical",
        description="No canonical link was found",
        priority_tier="P0",
        priority_score=0.95,
        severity=0.95,
        status="open",
    )

    decision = await PlanningService(
        settings,
        object(),
        [FakeFixer()],
    ).decide(
        [issue],
        scan_id=42,
        target_name="example",
    )

    assert captured["target_name"] == "example"
    assert decision.path.exists()
    assert decision.plan.actions[0].decision == "execute_automatically"
    assert [item.id for item in decision.select_issues([issue], preview=False)] == [1]
