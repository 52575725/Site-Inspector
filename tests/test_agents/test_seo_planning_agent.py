from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from src.agents.planning_context import (
    BusinessObjective,
    FeedbackSummary,
    PageSignals,
    PlanningContext,
)
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
async def test_business_goal_and_live_signals_change_action_order():
    planner = SEOPlanningAgent([
        FakeFixer("meta_fixer", "fully_auto", ["missing_title", "missing_meta_description"]),
    ])
    priority_url = "https://example.com/products/high-value/"
    context = PlanningContext(
        objective=BusinessObjective(
            goal="qualified_inquiries",
            priority_url_patterns=["/products/"],
        ),
        page_signals={
            priority_url: PageSignals(gsc_position=8, gsc_ctr=0.01, ga_pageviews=500),
        },
    )
    plan = await planner.create_plan(
        [
            issue(1, "missing_title", 0.6, url=priority_url),
            issue(2, "missing_meta_description", 0.9, url="https://example.com/blog/post/"),
        ],
        scan_id=12,
        target_name="example",
        context=context,
    )

    assert plan.actions[0].categories == ["missing_title"]
    assert plan.actions[0].business_value == 0.95
    assert plan.actions[0].opportunity_score > 0.7
    assert "priority business page" in plan.actions[0].decision_factors
    assert plan.facts["signal_pages"] == 1


@pytest.mark.asyncio
async def test_plan_explains_problem_solution_and_autonomous_decision():
    planner = SEOPlanningAgent([
        FakeFixer("canonical_fixer", "fully_auto", ["missing_canonical"]),
    ])
    plan = await planner.create_plan(
        [issue(1, "missing_canonical", 0.95, tier="P0")],
        scan_id=14,
        target_name="example",
    )

    action = plan.actions[0]
    assert "missing canonical" in action.problem_statement
    assert "canonical" in action.proposed_solution.lower()
    assert action.decision == "execute_automatically"
    assert action.solution_steps
    assert plan.facts["autonomous_actions"] == 1


@pytest.mark.asyncio
async def test_security_headers_are_not_classified_as_content_seo():
    planner = SEOPlanningAgent([
        FakeFixer(
            "headers_fixer",
            "semi_auto",
            ["missing_content_security_policy"],
        ),
    ])
    plan = await planner.create_plan(
        [issue(1, "missing_content_security_policy", 0.9, tier="P0")],
        scan_id=16,
        target_name="example",
    )

    action = plan.actions[0]
    assert action.phase == 4
    assert "web server or CDN" in action.proposed_solution
    assert "valid security headers" in action.expected_metrics
    assert "business facts remain unchanged" not in action.validation_checks
    assert "critical scripts" in action.rollback_condition


@pytest.mark.asyncio
async def test_capacity_selects_business_value_before_execution_phase():
    planner = SEOPlanningAgent(
        [FakeFixer("meta_fixer", "fully_auto", ["missing_title"])],
        policy=PlanningPolicy(max_actions=1),
    )
    product_url = "https://example.com/products/high-value/"
    context = PlanningContext(
        objective=BusinessObjective(
            goal="qualified_inquiries",
            priority_url_patterns=["/products/"],
        ),
        page_signals={
            product_url: PageSignals(gsc_position=8, gsc_ctr=0.01, ga_pageviews=500),
        },
    )
    plan = await planner.create_plan(
        [
            issue(1, "missing_canonical", 0.35, url="https://example.com/blog/old/"),
            issue(2, "missing_title", 0.8, url=product_url),
        ],
        scan_id=15,
        target_name="example",
        context=context,
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].issue_ids == [2]
    assert plan.actions[0].business_value == 0.95


@pytest.mark.asyncio
async def test_protected_pages_and_degraded_history_require_approval():
    planner = SEOPlanningAgent([
        FakeFixer("canonical_fixer", "fully_auto", ["missing_canonical"]),
        FakeFixer("meta_fixer", "fully_auto", ["missing_title"]),
    ])
    context = PlanningContext(
        objective=BusinessObjective(protected_url_patterns=["/legal/"]),
        category_feedback={
            "missing_title": FeedbackSummary(improved=1, degraded=2),
        },
    )
    plan = await planner.create_plan(
        [
            issue(1, "missing_canonical", 0.95, url="https://example.com/legal/terms/"),
            issue(2, "missing_title", 0.95, url="https://example.com/products/a/"),
        ],
        scan_id=13,
        target_name="example",
        context=context,
    )

    protected = next(action for action in plan.actions if action.issue_ids == [1])
    degraded = next(action for action in plan.actions if action.issue_ids == [2])
    assert protected.approval_required
    assert "protected page requires approval" in protected.decision_factors
    assert degraded.approval_required
    assert "historical degradation rate requires approval" in degraded.decision_factors
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
    assert plan.ai_strategy_note.startswith("Start with the indexability action")
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

    assert payload["schema_version"] == "1.3"
    assert payload["scan_id"] == 10
    assert payload["actions"][0]["evidence"][0]["issue_id"] == 1


def test_plan_cli_is_registered():
    result = CliRunner().invoke(app, ["plan", "generate", "--help"])
    assert result.exit_code == 0
    assert "read-only optimization plan" in result.output
    assert "--ai" in result.output
    assert "--goal" in result.output


def test_plan_cli_rejects_unknown_goal_before_loading_storage():
    result = CliRunner().invoke(app, ["plan", "generate", "--goal", "traffic_at_all_costs"])

    assert result.exit_code == 0
    assert "--goal must be" in result.output
