from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.agents.planning_context import PlanningContextCollector
from src.agents.seo_planning_agent import AuditPlan, SEOPlanningAgent, TextReasoner
from src.fixers.base import BaseFixer


@dataclass(frozen=True)
class PlanningDecision:
    plan: AuditPlan
    path: Path

    def select_issues(self, issues: Sequence[Any], *, preview: bool) -> list[Any]:
        return self.plan.select_issues(issues, dry_run=preview)


class PlanningService:
    """Single decision gateway between analyzed issues and every fixer workflow."""

    def __init__(
        self,
        settings: Settings,
        session: AsyncSession,
        fixers: Sequence[BaseFixer],
        *,
        reasoner: TextReasoner | None = None,
    ):
        self.settings = settings
        self.session = session
        self.fixers = fixers
        self.reasoner = reasoner

    async def decide(
        self,
        issues: Sequence[Any],
        *,
        scan_id: int,
        target_name: str,
        objective: str = (
            "Improve qualified organic visibility without unreviewed business changes"
        ),
    ) -> PlanningDecision:
        context = await PlanningContextCollector(
            self.settings,
            self.session,
            target_name=target_name,
        ).collect(issues)
        planner = SEOPlanningAgent(self.fixers, reasoner=self.reasoner)
        plan = await planner.create_plan(
            issues,
            scan_id=scan_id,
            target_name=target_name,
            objective=objective,
            context=context,
            use_ai=self.reasoner is not None,
        )
        path = (
            self.settings.data_dir
            / "reports"
            / "plans"
            / f"scan-{scan_id}-plan.json"
        )
        plan.write_json(path)
        return PlanningDecision(plan=plan, path=path)
