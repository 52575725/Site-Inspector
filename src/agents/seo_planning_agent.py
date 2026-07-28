from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from src.agents.planning_context import PlanningContext
from src.fixers.base import BaseFixer

logger = logging.getLogger(__name__)

RiskLevel = Literal["low", "medium", "high"]
ExecutionMode = Literal["fully_auto", "semi_auto", "manual_required"]
DecisionType = Literal[
    "execute_automatically",
    "request_approval",
    "manual_implementation",
]


class TextReasoner(Protocol):
    async def generate_text(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> str: ...


class PlanningPolicy(BaseModel):
    max_actions: int = Field(default=20, ge=1, le=100)
    max_issues_per_page: int = Field(default=3, ge=1, le=20)
    min_auto_confidence: float = Field(default=0.78, ge=0.0, le=1.0)
    feedback_min_samples: int = Field(default=3, ge=1, le=100)
    feedback_degradation_limit: float = Field(default=0.5, ge=0.0, le=1.0)


class PlanEvidence(BaseModel):
    issue_id: int
    url: str
    category: str
    title: str
    description: str = ""
    priority_tier: str
    priority_score: float


class PlannedAction(BaseModel):
    action_id: str
    phase: int
    title: str
    rationale: str
    problem_statement: str
    proposed_solution: str
    solution_steps: list[str] = Field(default_factory=list)
    decision: DecisionType
    strategic_note: str = ""
    issue_ids: list[int]
    urls: list[str]
    categories: list[str]
    fixer: str | None = None
    execution_mode: ExecutionMode
    risk: RiskLevel
    confidence: float
    impact_score: float
    business_value: float
    opportunity_score: float
    feedback_score: float
    effort_score: float
    decision_score: float
    estimated_hours: float
    decision_factors: list[str] = Field(default_factory=list)
    approval_required: bool
    dependencies: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)
    rollback_condition: str
    evidence: list[PlanEvidence]


class DeferredIssue(BaseModel):
    issue_id: int
    url: str
    category: str
    reason: str


class AuditPlan(BaseModel):
    schema_version: str = "1.3"
    scan_id: int
    target_name: str
    objective: str
    generated_at: datetime
    executive_summary: str
    facts: dict[str, Any]
    actions: list[PlannedAction]
    deferred: list[DeferredIssue]
    stop_conditions: list[str]
    warnings: list[str]
    min_auto_confidence: float = 0.78
    ai_assisted: bool = False
    ai_strategy_note: str = ""

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    def executable_issue_ids(self) -> list[int]:
        """Return only low-risk, high-confidence actions that need no approval."""
        return [
            issue_id
            for action in self.actions
            if (
                action.execution_mode == "fully_auto"
                and action.risk == "low"
                and action.confidence >= self.min_auto_confidence
                and not action.approval_required
            )
            for issue_id in action.issue_ids
        ]

    def planned_issue_ids(self) -> list[int]:
        return [issue_id for action in self.actions for issue_id in action.issue_ids]

    def select_issues(self, issues: Sequence[Any], *, dry_run: bool) -> list[Any]:
        """Bind a plan back to its source issues without expanding its scope."""
        allowed_ids = set(
            self.planned_issue_ids() if dry_run else self.executable_issue_ids()
        )
        return [issue for issue in issues if getattr(issue, "id", None) in allowed_ids]


class SEOPlanningAgent:
    """Build an evidence-bound optimization plan before any site write.

    The deterministic plan is authoritative. An optional language model may
    add narrative notes to known action IDs, but cannot add actions, change
    risk, lower approval requirements, or modify execution modes.
    """

    PHASE_LABELS = {
        1: "Restore crawlability and indexability",
        2: "Repair site architecture and URL signals",
        3: "Improve page relevance and search snippets",
        4: "Improve rich results, media, accessibility, and performance",
        5: "Strengthen authority, freshness, and competitive coverage",
    }

    LOW_RISK = {
        "missing_title", "title_too_short", "title_too_long",
        "missing_meta_description", "meta_description_too_short",
        "meta_description_too_long", "missing_viewport_meta",
        "missing_canonical", "image_no_lazy_loading",
        "image_missing_dimensions", "image_no_async_decoding",
    }
    HIGH_RISK_PREFIXES = (
        "content_", "cannibalization_", "url_", "freshness_", "eeat_",
    )
    HIGH_RISK = {
        "thin_content", "duplicate_content", "low_content_quality_ai",
        "robots_txt_disallow_all", "invalid_jsonld", "schema_invalid_value",
        "crawl_budget_faceted_url", "crawl_budget_pagination_self_canonical",
    }

    def __init__(
        self,
        fixers: Sequence[BaseFixer] = (),
        *,
        policy: PlanningPolicy | None = None,
        reasoner: TextReasoner | None = None,
    ):
        self.policy = policy or PlanningPolicy()
        self.reasoner = reasoner
        self._capabilities: dict[str, tuple[ExecutionMode, str]] = {}
        for fixer in fixers:
            mode: ExecutionMode = (
                fixer.fix_type
                if fixer.fix_type in {"fully_auto", "semi_auto"}
                else "manual_required"
            )
            for category in fixer.supported_categories:
                self._capabilities[category] = (mode, fixer.fixer_name)

    async def create_plan(
        self,
        issues: Sequence[Any],
        *,
        scan_id: int,
        target_name: str,
        objective: str = "Improve qualified organic visibility without unreviewed business changes",
        context: PlanningContext | None = None,
        use_ai: bool = False,
    ) -> AuditPlan:
        plan = self._build_plan(
            issues,
            scan_id=scan_id,
            target_name=target_name,
            objective=objective,
            context=context or PlanningContext(),
        )
        if use_ai and self.reasoner:
            await self._add_ai_notes(plan)
        elif use_ai:
            plan.warnings.append("AI reasoning was requested but no reasoner was configured")
        return plan

    def _build_plan(
        self,
        issues: Sequence[Any],
        *,
        scan_id: int,
        target_name: str,
        objective: str,
        context: PlanningContext | None = None,
    ) -> AuditPlan:
        context = context or PlanningContext()
        open_issues = [issue for issue in issues if getattr(issue, "status", "open") == "open"]
        ordered = sorted(
            open_issues,
            key=lambda issue: self._issue_sort_key(issue, context),
        )

        selected: list[Any] = []
        deferred: list[DeferredIssue] = []
        per_page: Counter[str] = Counter()
        for issue in ordered:
            url = str(getattr(issue, "url", ""))
            if per_page[url] >= self.policy.max_issues_per_page:
                deferred.append(self._defer(issue, "per-page planning limit reached"))
                continue
            per_page[url] += 1
            selected.append(issue)

        groups: dict[tuple[int, str, ExecutionMode, str | None, RiskLevel], list[Any]] = defaultdict(list)
        for issue in selected:
            category = str(getattr(issue, "category", "unknown"))
            mode, fixer = self._capability(category)
            phase = self._phase(category)
            risk = self._risk(category)
            groups[(phase, category, mode, fixer, risk)].append(issue)

        ranked_groups = sorted(
            groups.items(),
            key=lambda item: (
                -max(self._decision_score(issue, context)[0] for issue in item[1]),
                item[0][0],
                item[0][1],
            ),
        )

        mandatory = [
            item for item in ranked_groups
            if any(str(getattr(issue, "priority_tier", "P2")) == "P0" for issue in item[1])
        ]
        mandatory_keys = {item[0] for item in mandatory}
        optional = [item for item in ranked_groups if item[0] not in mandatory_keys]
        selected_by_value = (mandatory + optional)[: self.policy.max_actions]
        kept_groups = sorted(
            selected_by_value,
            key=lambda item: (
                item[0][0],
                -max(self._decision_score(issue, context)[0] for issue in item[1]),
                item[0][1],
            ),
        )
        selected_keys = {item[0] for item in selected_by_value}
        for group_key, group_issues in ranked_groups:
            if group_key in selected_keys:
                continue
            deferred.extend(self._defer(issue, "plan action capacity reached") for issue in group_issues)

        actions: list[PlannedAction] = []
        for index, (group_key, group_issues) in enumerate(kept_groups, start=1):
            phase, category, mode, fixer, risk = group_key
            action_id = f"A{index:02d}"
            evidence = [self._evidence(issue) for issue in group_issues]
            confidence = self._confidence(group_issues, mode, category, context)
            component_scores = [
                self._decision_score(issue, context) for issue in group_issues
            ]
            decision_score = self._average([score[0] for score in component_scores])
            impact_score = self._average([score[1] for score in component_scores])
            business_value = self._average([score[2] for score in component_scores])
            opportunity_score = self._average([score[3] for score in component_scores])
            feedback_score = self._average([score[4] for score in component_scores])
            effort_score = self._effort_score(mode, risk, len(group_issues))
            feedback = context.feedback_for(category)
            protected = any(
                context.objective.is_protected(item.url) for item in evidence
            )
            degraded_history = (
                feedback.total >= self.policy.feedback_min_samples
                and feedback.degradation_rate >= self.policy.feedback_degradation_limit
            )
            approval_required = not (
                mode == "fully_auto"
                and risk == "low"
                and confidence >= self.policy.min_auto_confidence
                and not protected
                and not degraded_history
            )
            decision = self._decision(mode, approval_required)
            actions.append(PlannedAction(
                action_id=action_id,
                phase=phase,
                title=self._action_title(category, len(group_issues)),
                rationale=self._rationale(category, group_issues),
                problem_statement=self._problem_statement(category, group_issues),
                proposed_solution=self._proposed_solution(category, fixer),
                solution_steps=self._solution_steps(category, fixer),
                decision=decision,
                issue_ids=[item.issue_id for item in evidence],
                urls=sorted({item.url for item in evidence}),
                categories=[category],
                fixer=fixer,
                execution_mode=mode,
                risk=risk,
                confidence=confidence,
                impact_score=impact_score,
                business_value=business_value,
                opportunity_score=opportunity_score,
                feedback_score=feedback_score,
                effort_score=effort_score,
                decision_score=decision_score,
                estimated_hours=self._estimated_hours(mode, risk, len(group_issues)),
                decision_factors=self._decision_factors(
                    group_issues, category, context, protected, degraded_history,
                ),
                approval_required=approval_required,
                expected_metrics=self._expected_metrics(category),
                validation_checks=self._validation_checks(category),
                rollback_condition=self._rollback_condition(category),
                evidence=evidence,
            ))

        self._attach_dependencies(actions)
        mode_counts = Counter(action.execution_mode for action in actions)
        tier_counts = Counter(str(getattr(issue, "priority_tier", "P2")) for issue in open_issues)
        warnings = [
            "Expected SEO impact is an estimate until Search Console checkpoints confirm it",
            "High-risk and non-deterministic actions require human approval",
        ]
        if not self._capabilities:
            warnings.append("No fixer catalog was supplied; every action is manual")
        warnings.extend(context.warnings)

        return AuditPlan(
            scan_id=scan_id,
            target_name=target_name,
            objective=objective,
            generated_at=datetime.now(UTC),
            executive_summary=(
                f"Plan {len(actions)} actions across {len({i.url for i in selected})} pages; "
                f"{sum(not action.approval_required for action in actions)} actions meet the "
                "low-risk unattended threshold."
            ),
            facts={
                "open_issues": len(open_issues),
                "planned_issues": sum(len(action.issue_ids) for action in actions),
                "affected_pages": len({str(getattr(issue, "url", "")) for issue in open_issues}),
                "priority_tiers": dict(sorted(tier_counts.items())),
                "execution_modes": dict(sorted(mode_counts.items())),
                "planning_goal": context.objective.goal,
                "signal_pages": len(context.page_signals),
                "feedback_observations": sum(
                    feedback.total for feedback in context.category_feedback.values()
                ),
                "autonomous_actions": sum(
                    action.decision == "execute_automatically" for action in actions
                ),
                "approval_actions": sum(
                    action.decision == "request_approval" for action in actions
                ),
                "manual_actions": sum(
                    action.decision == "manual_implementation" for action in actions
                ),
            },
            actions=actions,
            deferred=deferred,
            stop_conditions=[
                "Stop if HTML, JSON-LD, YAML, or application tests fail",
                "Stop if rendered layout or a conversion path regresses",
                "Stop if a source snapshot is unavailable for rollback",
                "Pause expansion when a canary page fails validation",
            ],
            warnings=warnings,
            min_auto_confidence=self.policy.min_auto_confidence,
        )

    async def _add_ai_notes(self, plan: AuditPlan) -> None:
        allowed = {action.action_id: action for action in plan.actions}
        prompt_actions = [
            {
                "action_id": action.action_id,
                "title": action.title,
                "risk": action.risk,
                "confidence": action.confidence,
                "decision": action.decision,
                "execution_mode": action.execution_mode,
                "approval_required": action.approval_required,
                "decision_score": action.decision_score,
                "decision_factors": action.decision_factors,
                "urls": action.urls,
                "evidence": [item.model_dump() for item in action.evidence],
            }
            for action in plan.actions
        ]
        prompt = json.dumps({
            "objective": plan.objective,
            "facts": plan.facts,
            "actions": prompt_actions,
            "response_schema": {
                "portfolio_note": "string",
                "action_notes": [{"action_id": "A01", "note": "string"}],
            },
        }, ensure_ascii=False)
        system = (
            "You are an SEO planning reviewer. Use only supplied evidence. "
            "Do not invent metrics, pages, actions, or company facts. Return JSON only."
        )
        try:
            raw = await self.reasoner.generate_text(
                prompt,
                system=system,
                temperature=0.1,
                max_tokens=1800,
            )
            payload = self._parse_json(raw)
            for item in payload.get("action_notes", []):
                action = allowed.get(str(item.get("action_id", "")))
                note = self._clean_note(item.get("note", ""))
                if action and note:
                    action.strategic_note = note
            portfolio_note = self._clean_note(payload.get("portfolio_note", ""), limit=800)
            if portfolio_note:
                plan.ai_strategy_note = portfolio_note
            plan.ai_assisted = True
            plan.warnings.append("AI notes are advisory and cannot change execution controls")
        except Exception as exc:
            logger.warning("Planning AI refinement failed: %s", exc)
            plan.warnings.append("AI refinement failed; deterministic plan retained")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("AI response must be a JSON object")
        return payload

    @staticmethod
    def _clean_note(value: Any, *, limit: int = 500) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:limit]

    def _issue_sort_key(
        self, issue: Any, context: PlanningContext,
    ) -> tuple[int, float, float, int]:
        tier_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return (
            tier_order.get(str(getattr(issue, "priority_tier", "P2")), 2),
            -self._decision_score(issue, context)[0],
            -float(getattr(issue, "severity", 0.0) or 0.0),
            int(getattr(issue, "id", 0) or 0),
        )

    def _capability(self, category: str) -> tuple[ExecutionMode, str | None]:
        return self._capabilities.get(category, ("manual_required", None))

    @classmethod
    def _risk(cls, category: str) -> RiskLevel:
        if category in cls.LOW_RISK:
            return "low"
        if category in cls.HIGH_RISK or category.startswith(cls.HIGH_RISK_PREFIXES):
            return "high"
        return "medium"

    @staticmethod
    def _is_security_header(category: str) -> bool:
        return any(token in category for token in (
            "content_security_policy",
            "strict_transport_security",
            "security_header",
            "x_content_type_options",
            "x_frame_options",
            "referrer_policy",
            "permissions_policy",
        ))

    @staticmethod
    def _phase(category: str) -> int:
        if SEOPlanningAgent._is_security_header(category):
            return 4
        if any(token in category for token in (
            "robots", "sitemap", "http_", "x_robots", "content_type", "canonical",
        )):
            return 1
        if any(token in category for token in (
            "hreflang", "redirect", "orphan", "crawl_budget", "url_", "breadcrumb",
        )):
            return 2
        if any(token in category for token in (
            "title", "meta_description", "h1", "h_tag", "keyword", "content",
            "cannibalization",
        )):
            return 3
        if any(token in category for token in (
            "schema", "structured", "image", "mobile", "viewport", "lcp", "cls",
            "performance", "wcag", "accessibility", "font", "touch", "scroll",
        )):
            return 4
        return 5

    def _confidence(
        self,
        issues: Sequence[Any],
        mode: ExecutionMode,
        category: str,
        context: PlanningContext,
    ) -> float:
        evidence = self._average([
            sum(bool(getattr(issue, field, None)) for field in ("url", "title", "description")) / 3
            for issue in issues
        ])
        diagnosis = self._average([
            0.55 + 0.4 * float(getattr(issue, "priority_score", 0.0) or 0.0)
            for issue in issues
        ])
        feedback = context.feedback_for(category)
        reliability = (
            feedback.score
            if feedback.total >= self.policy.feedback_min_samples
            else {"fully_auto": 0.9, "semi_auto": 0.65, "manual_required": 0.45}[mode]
        )
        signal_coverage = self._average([
            context.signals_for(str(getattr(issue, "url", ""))).observed_fields / 5
            for issue in issues
        ])
        impact_confidence = 0.55 + signal_coverage * 0.4
        return round(min(0.98, evidence * 0.3 + diagnosis * 0.3 + reliability * 0.25 + impact_confidence * 0.15), 2)

    def _decision_score(
        self, issue: Any, context: PlanningContext,
    ) -> tuple[float, float, float, float, float]:
        url = str(getattr(issue, "url", ""))
        category = str(getattr(issue, "category", "unknown"))
        impact = self._average([
            float(getattr(issue, "priority_score", 0.0) or 0.0),
            float(getattr(issue, "severity", 0.0) or 0.0),
            float(getattr(issue, "impact_scope", 0.0) or 0.0),
        ])
        if not getattr(issue, "impact_scope", None):
            impact = self._average([
                float(getattr(issue, "priority_score", 0.0) or 0.0),
                float(getattr(issue, "severity", 0.0) or 0.0),
            ])
        business = context.objective.business_value(url)
        opportunity = self._opportunity_score(context.signals_for(url))
        feedback = context.feedback_for(category).score
        mode, _ = self._capability(category)
        risk = self._risk(category)
        effort = self._effort_score(mode, risk, 1)
        weights = {
            "organic_visibility": (0.3, 0.2, 0.3, 0.1, 0.1),
            "qualified_inquiries": (0.25, 0.35, 0.2, 0.1, 0.1),
            "technical_health": (0.4, 0.1, 0.15, 0.15, 0.2),
        }[context.objective.goal]
        decision = (
            impact * weights[0]
            + business * weights[1]
            + opportunity * weights[2]
            + feedback * weights[3]
            + (1 - effort) * weights[4]
        )
        return (
            round(decision, 3), round(impact, 3), round(business, 3),
            round(opportunity, 3), round(feedback, 3),
        )

    @staticmethod
    def _opportunity_score(signals: Any) -> float:
        observed: list[float] = []
        if signals.indexed is False:
            observed.append(1.0)
        elif signals.indexed is True:
            observed.append(0.55)
        if signals.gsc_position is not None:
            position = signals.gsc_position
            if 4 <= position <= 20:
                observed.append(max(0.55, 1 - abs(position - 10) / 20))
            elif position <= 3:
                observed.append(0.35)
            else:
                observed.append(max(0.3, 0.75 - (position - 20) / 100))
        if signals.gsc_ctr is not None:
            observed.append(max(0.25, min(1.0, 1 - signals.gsc_ctr * 8)))
        if signals.ga_pageviews is not None:
            views = signals.ga_pageviews
            observed.append(0.3 if views == 0 else min(0.95, 0.4 + views / 2000))
        return SEOPlanningAgent._average(observed) if observed else 0.5

    @staticmethod
    def _effort_score(mode: ExecutionMode, risk: RiskLevel, count: int) -> float:
        mode_cost = {"fully_auto": 0.2, "semi_auto": 0.5, "manual_required": 0.8}[mode]
        risk_cost = {"low": 0.15, "medium": 0.45, "high": 0.75}[risk]
        scale_cost = min(0.2, max(0, count - 1) * 0.02)
        return round(min(1.0, mode_cost * 0.55 + risk_cost * 0.35 + scale_cost), 3)

    @staticmethod
    def _estimated_hours(mode: ExecutionMode, risk: RiskLevel, count: int) -> float:
        base = {"fully_auto": 0.25, "semi_auto": 0.75, "manual_required": 1.5}[mode]
        multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0}[risk]
        return round(base * multiplier + max(0, count - 1) * 0.2, 2)

    def _decision_factors(
        self,
        issues: Sequence[Any],
        category: str,
        context: PlanningContext,
        protected: bool,
        degraded_history: bool,
    ) -> list[str]:
        factors = [f"planning goal: {context.objective.goal}"]
        if any(context.objective.business_value(str(getattr(issue, "url", ""))) >= 0.9 for issue in issues):
            factors.append("priority business page")
        if any(context.signals_for(str(getattr(issue, "url", ""))).observed_fields for issue in issues):
            factors.append("traffic or search opportunity signal available")
        feedback = context.feedback_for(category)
        if feedback.total:
            factors.append(
                f"historical outcomes: {feedback.improved} improved, "
                f"{feedback.degraded} degraded, {feedback.unchanged} unchanged"
            )
        if protected:
            factors.append("protected page requires approval")
        if degraded_history:
            factors.append("historical degradation rate requires approval")
        return factors

    @staticmethod
    def _average(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    @staticmethod
    def _evidence(issue: Any) -> PlanEvidence:
        return PlanEvidence(
            issue_id=int(getattr(issue, "id", 0) or 0),
            url=str(getattr(issue, "url", "")),
            category=str(getattr(issue, "category", "unknown")),
            title=str(getattr(issue, "title", "")),
            description=str(getattr(issue, "description", "") or "")[:500],
            priority_tier=str(getattr(issue, "priority_tier", "P2")),
            priority_score=round(float(getattr(issue, "priority_score", 0.0) or 0.0), 3),
        )

    @staticmethod
    def _defer(issue: Any, reason: str) -> DeferredIssue:
        return DeferredIssue(
            issue_id=int(getattr(issue, "id", 0) or 0),
            url=str(getattr(issue, "url", "")),
            category=str(getattr(issue, "category", "unknown")),
            reason=reason,
        )

    @classmethod
    def _action_title(cls, category: str, count: int) -> str:
        readable = category.replace("_", " ")
        return f"{cls.PHASE_LABELS[cls._phase(category)]}: {readable} ({count})"

    @staticmethod
    def _rationale(category: str, issues: Sequence[Any]) -> str:
        tiers = sorted({str(getattr(issue, "priority_tier", "P2")) for issue in issues})
        pages = len({str(getattr(issue, "url", "")) for issue in issues})
        return (
            f"The scan found {len(issues)} evidence-backed {category} issue(s) "
            f"across {pages} page(s), prioritized as {', '.join(tiers)}."
        )

    @staticmethod
    def _decision(mode: ExecutionMode, approval_required: bool) -> DecisionType:
        if not approval_required:
            return "execute_automatically"
        if mode == "manual_required":
            return "manual_implementation"
        return "request_approval"

    @staticmethod
    def _problem_statement(category: str, issues: Sequence[Any]) -> str:
        pages = len({str(getattr(issue, "url", "")) for issue in issues})
        examples = [
            str(getattr(issue, "description", "") or "").strip()
            for issue in issues[:2]
            if str(getattr(issue, "description", "") or "").strip()
        ]
        observed = f" Examples: {'; '.join(examples)}" if examples else ""
        return (
            f"Detected {len(issues)} {category.replace('_', ' ')} issue(s) "
            f"affecting {pages} page(s).{observed}"
        )[:1000]

    @staticmethod
    def _proposed_solution(category: str, fixer: str | None) -> str:
        if SEOPlanningAgent._is_security_header(category):
            solution = (
                "Configure the missing response security header at the web server or CDN, "
                "starting with report-only or a canary page where applicable"
            )
        elif "robots" in category or "sitemap" in category:
            solution = "Correct crawl directives and regenerate a resolvable, canonical sitemap"
        elif "canonical" in category or "hreflang" in category:
            solution = "Align canonical and language annotations with the final indexable URL set"
        elif "title" in category or "meta_description" in category:
            solution = "Rewrite the search snippet from verified page content and target intent"
        elif any(token in category for token in ("content", "keyword", "cannibalization")):
            solution = "Consolidate search intent and improve evidence-backed page content without inventing facts"
        elif "schema" in category or "structured" in category:
            solution = "Repair JSON-LD using only entities and properties verified on the page"
        elif any(token in category for token in ("image", "lcp", "cls", "performance")):
            solution = "Optimize media delivery and layout stability while preserving visual quality"
        elif any(token in category for token in ("link", "orphan", "breadcrumb")):
            solution = "Restore contextual internal navigation and crawlable page relationships"
        elif any(token in category for token in ("mobile", "viewport", "accessibility", "wcag")):
            solution = "Correct responsive and accessibility markup, then verify desktop and mobile rendering"
        elif any(token in category for token in ("eeat", "freshness")):
            solution = "Update trust and freshness signals from verifiable business and editorial evidence"
        else:
            solution = "Apply the matching remediation and validate that the detected condition is removed"
        return f"{solution}. Recommended handler: {fixer}." if fixer else f"{solution}."

    @staticmethod
    def _solution_steps(category: str, fixer: str | None) -> list[str]:
        steps = [
            "Confirm the finding against the current source and rendered page",
            "Capture a rollback snapshot before changing the source",
        ]
        if fixer:
            steps.append(f"Apply the scoped change with {fixer}")
        else:
            steps.append("Prepare a reviewed manual change because no safe fixer is available")
        steps.extend([
            "Run category-specific validation and rendering checks",
            "Measure the expected search or engagement metric after release",
        ])
        if SEOPlanningAgent._is_security_header(category):
            steps.insert(
                2,
                "Inventory scripts, forms, embeds, and third-party origins before enforcing the header",
            )
        elif any(token in category for token in ("content", "eeat", "freshness")):
            steps.insert(2, "Verify every business claim against an approved source")
        return steps

    @staticmethod
    def _expected_metrics(category: str) -> list[str]:
        if SEOPlanningAgent._is_security_header(category):
            return ["valid security headers", "policy violations", "affected page count"]
        if any(token in category for token in ("robots", "sitemap", "canonical", "crawl", "url_")):
            return ["valid indexed pages", "crawl errors", "index coverage"]
        if any(token in category for token in ("title", "meta_description", "schema")):
            return ["search impressions", "organic CTR", "rich result validity"]
        if any(token in category for token in ("content", "keyword", "cannibalization", "eeat")):
            return ["query position", "qualified clicks", "engaged sessions"]
        if any(token in category for token in ("image", "lcp", "cls", "performance", "mobile")):
            return ["Core Web Vitals", "page load performance", "mobile usability"]
        return ["affected issue count", "organic landing-page performance"]

    @staticmethod
    def _validation_checks(category: str) -> list[str]:
        checks = ["source snapshot exists", "syntax and content-retention validation"]
        if SEOPlanningAgent._is_security_header(category):
            checks.extend([
                "response header is present on representative pages",
                "critical scripts, forms, images, and embeds still work",
                "browser console has no new policy violations",
            ])
        if "schema" in category or "structured" in category:
            checks.extend(["all JSON-LD parses", "no duplicate or fabricated schema fields"])
        if any(token in category for token in ("robots", "sitemap", "canonical", "hreflang")):
            checks.extend(["crawl directives remain valid", "canonical and language URLs resolve"])
        if any(token in category for token in ("image", "mobile", "lcp", "cls", "performance")):
            checks.extend(["desktop and mobile render comparison", "asset URLs resolve"])
        if (
            not SEOPlanningAgent._is_security_header(category)
            and any(token in category for token in ("content", "title", "meta_description", "eeat"))
        ):
            checks.extend(["business facts remain unchanged", "copy is not truncated"])
        return list(dict.fromkeys(checks))

    @staticmethod
    def _rollback_condition(category: str) -> str:
        if SEOPlanningAgent._is_security_header(category):
            return (
                "Rollback if the policy blocks critical scripts, forms, images, "
                "or third-party integrations"
            )
        if (
            not SEOPlanningAgent._is_security_header(category)
            and any(token in category for token in ("content", "title", "meta_description", "eeat"))
        ):
            return "Rollback if facts change, text is truncated, or the target metric degrades"
        return "Rollback if validation fails, rendering regresses, or the target metric degrades"

    @staticmethod
    def _attach_dependencies(actions: list[PlannedAction]) -> None:
        for action in actions:
            earlier = [
                candidate.action_id
                for candidate in actions
                if candidate.phase < action.phase and set(candidate.urls) & set(action.urls)
            ]
            action.dependencies = earlier[-3:]
