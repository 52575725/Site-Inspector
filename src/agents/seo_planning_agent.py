from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from src.fixers.base import BaseFixer

logger = logging.getLogger(__name__)

RiskLevel = Literal["low", "medium", "high"]
ExecutionMode = Literal["fully_auto", "semi_auto", "manual_required"]


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
    strategic_note: str = ""
    issue_ids: list[int]
    urls: list[str]
    categories: list[str]
    fixer: str | None = None
    execution_mode: ExecutionMode
    risk: RiskLevel
    confidence: float
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
    schema_version: str = "1.0"
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
    ai_assisted: bool = False

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
                and action.confidence >= 0.78
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
        use_ai: bool = False,
    ) -> AuditPlan:
        plan = self._build_plan(
            issues,
            scan_id=scan_id,
            target_name=target_name,
            objective=objective,
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
    ) -> AuditPlan:
        open_issues = [issue for issue in issues if getattr(issue, "status", "open") == "open"]
        ordered = sorted(open_issues, key=self._issue_sort_key)

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
                item[0][0],
                -max(float(getattr(issue, "priority_score", 0.0) or 0.0) for issue in item[1]),
                item[0][1],
            ),
        )

        kept_groups = ranked_groups[: self.policy.max_actions]
        for _, group_issues in ranked_groups[self.policy.max_actions :]:
            deferred.extend(self._defer(issue, "plan action capacity reached") for issue in group_issues)

        actions: list[PlannedAction] = []
        for index, (group_key, group_issues) in enumerate(kept_groups, start=1):
            phase, category, mode, fixer, risk = group_key
            action_id = f"A{index:02d}"
            evidence = [self._evidence(issue) for issue in group_issues]
            confidence = self._confidence(group_issues, mode)
            approval_required = not (
                mode == "fully_auto"
                and risk == "low"
                and confidence >= self.policy.min_auto_confidence
            )
            actions.append(PlannedAction(
                action_id=action_id,
                phase=phase,
                title=self._action_title(category, len(group_issues)),
                rationale=self._rationale(category, group_issues),
                issue_ids=[item.issue_id for item in evidence],
                urls=sorted({item.url for item in evidence}),
                categories=[category],
                fixer=fixer,
                execution_mode=mode,
                risk=risk,
                confidence=confidence,
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
        )

    async def _add_ai_notes(self, plan: AuditPlan) -> None:
        allowed = {action.action_id: action for action in plan.actions}
        prompt_actions = [
            {
                "action_id": action.action_id,
                "title": action.title,
                "risk": action.risk,
                "confidence": action.confidence,
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
                plan.executive_summary = f"{plan.executive_summary} {portfolio_note}"
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

    @staticmethod
    def _issue_sort_key(issue: Any) -> tuple[int, float, float, int]:
        tier_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return (
            tier_order.get(str(getattr(issue, "priority_tier", "P2")), 2),
            -float(getattr(issue, "priority_score", 0.0) or 0.0),
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
    def _phase(category: str) -> int:
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

    @staticmethod
    def _confidence(issues: Sequence[Any], mode: ExecutionMode) -> float:
        scores = [float(getattr(issue, "priority_score", 0.0) or 0.0) for issue in issues]
        average = sum(scores) / max(1, len(scores))
        evidence_bonus = 0.05 if all(getattr(issue, "description", None) for issue in issues) else 0.0
        mode_bonus = {"fully_auto": 0.18, "semi_auto": 0.08, "manual_required": 0.0}[mode]
        return round(min(0.96, 0.5 + average * 0.25 + evidence_bonus + mode_bonus), 2)

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
    def _expected_metrics(category: str) -> list[str]:
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
        if "schema" in category or "structured" in category:
            checks.extend(["all JSON-LD parses", "no duplicate or fabricated schema fields"])
        if any(token in category for token in ("robots", "sitemap", "canonical", "hreflang")):
            checks.extend(["crawl directives remain valid", "canonical and language URLs resolve"])
        if any(token in category for token in ("image", "mobile", "lcp", "cls", "performance")):
            checks.extend(["desktop and mobile render comparison", "asset URLs resolve"])
        if any(token in category for token in ("content", "title", "meta_description", "eeat")):
            checks.extend(["business facts remain unchanged", "copy is not truncated"])
        return list(dict.fromkeys(checks))

    @staticmethod
    def _rollback_condition(category: str) -> str:
        if any(token in category for token in ("content", "title", "meta_description", "eeat")):
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
