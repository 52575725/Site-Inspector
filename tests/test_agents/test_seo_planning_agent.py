from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from src.agents.seo_planning_agent import PlanningPolicy, SEOPlanningAgent
from src.cli.main import app


class FakeFixer:
    def __init__(self, name: str, mode: str, categories: list[str]):
        self.fixer_name = name
        self.fix_type = mode
        self.supported_categories = categories


class FakeReasoner:
    async def generate_text(self, prompt: str, system: str = "", **kwargs) -> str:
        assert "Do not invent" in system
        return json.dumps({
            "portfolio_note": "Start with the indexability action and validate the canary page.",
            "action_notes": [
                {"action_id": "A01", "note": "This note is grounded in supplied evidence."},
                {"action_id": "A99", "note": "Injected action must be ignored."},
            ],
        })


def issue(
    issue_id: int,
    category: str,
    score: float,
    *,
    url: str = "https://example.com/products/bar/",
    tier: str = "P1",
):
    return SimpleNamespace(
        id=issue_id,
        url=url,
        category=category,
        title=category.replace("_", " ").title(),
        description=f"Observed {category} in the scan",
        priority_tier=tier,
        priority_score=score,
        severity=score,
        status="open",
    )


@pytest.mark.asyncio
async def test_plan_orders_dependencies_and_separates_approval():
    planner = SEOPlanningAgent([
        FakeFixer("canonical_fixer", "fully_auto", ["missing_canonical"]),
        FakeFixer("meta_fixer", "fully_auto", ["missing_meta_description"]),
    ])
    plan = await planner.create_plan(
        [
            issue(1, "missing_canonical", 0.92, tier="P0"),
            issue(2, "thin_content", 0.86),
            issue(3, "missing_meta_description", 0.81),
            issue(4, "missing_title", 0.72),
        ],
        scan_id=7,
        target_name="example",
    )

    assert [action.phase for action in plan.actions] == [1, 3, 3]
    canonical = plan.actions[0]
    assert not canonical.approval_required
    assert canonical.issue_ids == [1]
    assert all(canonical.action_id in action.dependencies for action in plan.actions[1:])

    content_action = next(action for action in plan.actions if "thin_content" in action.categories)
    assert content_action.risk == "high"
    assert content_action.execution_mode == "manual_required"
    assert content_action.approval_required
    assert content_action.issue_ids[0] not in plan.executable_issue_ids()

    assert any(item.issue_id == 4 and "per-page" in item.reason for item in plan.deferred)

    dry_run_issues = plan.select_issues(
        [issue(1, "missing_canonical", 0.92), issue(2, "thin_content", 0.86)],
        dry_run=True,
    )
    apply_issues = plan.select_issues(
        [issue(1, "missing_canonical", 0.92), issue(2, "thin_content", 0.86)],
        dry_run=False,
    )
    assert [item.id for item in dry_run_issues] == [1, 2]
    assert [item.id for item in apply_issues] == [1]


def test_executable_allowlist_rechecks_all_safety_invariants():
    planner = SEOPlanningAgent([
        FakeFixer("canonical_fixer", "fully_auto", ["missing_canonical"]),
    ])
    plan = planner._build_plan(
        [issue(1, "missing_canonical", 0.92, tier="P0")],
        scan_id=11,
        target_name="example",
        objective="test",
    )

    action = plan.actions[0]
    assert plan.executable_issue_ids() == [1]

    action.execution_mode = "semi_auto"
    assert plan.executable_issue_ids() == []
    action.execution_mode = "fully_auto"
    action.risk = "high"
    assert plan.executable_issue_ids() == []
    action.risk = "low"
    action.confidence = 0.5
    assert plan.executable_issue_ids() == []
    action.confidence = 0.9
    action.approval_required = True
    assert plan.executable_issue_ids() == []


@pytest.mark.asyncio
async def test_ai_can_only_annotate_known_actions():
    planner = SEOPlanningAgent(
        [FakeFixer("canonical_fixer", "fully_auto", ["missing_canonical"])],
        reasoner=FakeReasoner(),
    )
    plan = await planner.create_plan(
        [issue(1, "missing_canonical", 0.9, tier="P0")],
        scan_id=8,
        target_name="example",
        use_ai=True,
    )

    assert plan.ai_assisted
    assert len(plan.actions) == 1
    assert plan.actions[0].action_id == "A01"
    assert plan.actions[0].strategic_note == "This note is grounded in supplied evidence."
    assert not plan.actions[0].approval_required
    assert "A99" not in plan.model_dump_json()


@pytest.mark.asyncio
async def test_plan_capacity_defers_excess_actions():
    planner = SEOPlanningAgent(policy=PlanningPolicy(max_actions=1, max_issues_per_page=3))
    plan = await planner.create_plan(
        [
            issue(1, "missing_canonical", 0.9, url="https://example.com/a/"),
            issue(2, "thin_content", 0.8, url="https://example.com/b/"),
        ],
        scan_id=9,
        target_name="example",
    )

    assert len(plan.actions) == 1
    assert len(plan.deferred) == 1
    assert plan.deferred[0].reason == "plan action capacity reached"


@pytest.mark.asyncio
async def test_plan_json_round_trip(tmp_path):
    planner = SEOPlanningAgent()
    plan = await planner.create_plan(
        [issue(1, "thin_content", 0.7)],
        scan_id=10,
        target_name="example",
    )
    path = plan.write_json(tmp_path / "plan.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["scan_id"] == 10
    assert payload["actions"][0]["evidence"][0]["issue_id"] == 1


def test_plan_cli_is_registered():
    result = CliRunner().invoke(app, ["plan", "generate", "--help"])
    assert result.exit_code == 0
    assert "read-only optimization plan" in result.output
    assert "--ai" in result.output
