from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.agents.image_agent import ArticleImageAgent
from src.agents.models import AgentEvent, ArticleAgentStage, ArticleAgentState
from src.agents.quality_agent import ArticleQualityAgent


class ArticleOrchestratorAgent:
    """Persist and coordinate research, writing, image, QA, and publishing stages."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.image_agent = ArticleImageAgent()
        self.quality_agent = ArticleQualityAgent()

    def start(self, website_url: str, request: dict) -> ArticleAgentState:
        state = ArticleAgentState(
            run_id=uuid.uuid4().hex,
            website_url=website_url,
            request=request,
            stage=ArticleAgentStage.RESEARCHING,
            events=[AgentEvent(stage=ArticleAgentStage.RESEARCHING, summary="Website research started")],
        )
        return self._save(state)

    def load(self, run_id: str) -> ArticleAgentState:
        if not run_id or any(char not in "0123456789abcdef" for char in run_id) or len(run_id) != 32:
            raise ValueError("Invalid article agent run identifier")
        path = self.state_dir / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError("Article agent run not found")
        return ArticleAgentState.model_validate_json(path.read_text(encoding="utf-8"))

    def complete_research(self, state: ArticleAgentState, research_id: str, report: dict) -> ArticleAgentState:
        profile = report.get("profile") or {}
        state.research_id = research_id
        state.research_summary = {
            "site_name": profile.get("site_name", ""),
            "business_summary": profile.get("business_summary", ""),
            "primary_language": profile.get("primary_language", ""),
            "detected_languages": profile.get("detected_languages", []),
            "topic": (report.get("editorial_decision") or {}).get("topic", ""),
        }
        return self._transition(state, ArticleAgentStage.AWAITING_CONFIRMATION, "Research plan is ready for review")

    def begin_writing(self, state: ArticleAgentState) -> ArticleAgentState:
        return self._transition(state, ArticleAgentStage.WRITING, "Confirmed article plan is being written")

    def complete_writing(
        self,
        state: ArticleAgentState,
        article: dict,
        *,
        revision_count: int = 0,
    ) -> ArticleAgentState:
        report = self.quality_agent.inspect_content(
            str(article.get("html", "")),
            research_report=article.get("research_report") or {},
            expected_word_count=(article.get("confirmed_brief") or {}).get("confirmed_word_count"),
        )
        state.article_id = str(article.get("id", ""))
        state.writing_summary = {
            "title": article.get("title", ""),
            "language": article.get("language", ""),
            "page_type": article.get("page_type", ""),
            "content_direction": article.get("content_direction", ""),
        }
        state.content_quality = report
        state.revision_count = revision_count
        stage = (
            ArticleAgentStage.ARTICLE_READY
            if report.passed
            else ArticleAgentStage.CONTENT_REVIEW
        )
        return self._transition(
            state,
            stage,
            (
                "Article passed the content quality gate"
                if report.passed
                else "Article requires review after automatic revisions"
            ),
            {"quality_passed": report.passed, "revision_count": revision_count},
        )

    def plan_images(self, state: ArticleAgentState, article: dict, requested_target: int | None) -> ArticleAgentState:
        if state.stage not in {
            ArticleAgentStage.ARTICLE_READY,
            ArticleAgentStage.READY_TO_PUBLISH,
            ArticleAgentStage.IMAGE_REVIEW,
        }:
            raise ValueError("Article must pass the content quality gate before image planning")
        state = self._transition(state, ArticleAgentStage.IMAGE_PLANNING, "Image agent is analyzing the article")
        state.image_plan = self.image_agent.plan(
            str(article.get("html", "")),
            research_report=article.get("research_report") or {},
            requested_target=requested_target,
        )
        return self._transition(
            state,
            ArticleAgentStage.AWAITING_IMAGE_SELECTION,
            "Image plan and candidate requirements are ready",
            {"target_count": state.image_plan.target_count, "needed_count": state.image_plan.needed_count},
        )

    def image_proposal_ready(self, state: ArticleAgentState, proposal_id: str) -> ArticleAgentState:
        return self._transition(
            state,
            ArticleAgentStage.IMAGE_REVIEW,
            "Image proposal is ready for review",
            {"proposal_id": proposal_id},
        )

    def complete_images(
        self,
        state: ArticleAgentState,
        html: str,
        *,
        research_report: dict | None = None,
        expected_word_count: int | None = None,
    ) -> ArticleAgentState:
        if state.image_plan is None:
            raise ValueError("Image plan is missing from the article agent run")
        state.final_quality = self.quality_agent.inspect_article_with_images(
            html,
            state.image_plan,
            research_report=research_report,
            expected_word_count=expected_word_count,
        )
        stage = ArticleAgentStage.READY_TO_PUBLISH if state.final_quality.passed else ArticleAgentStage.IMAGE_REVIEW
        summary = "Joint article-image checks passed" if state.final_quality.passed else "Joint checks found image issues that require review"
        return self._transition(state, stage, summary, {"quality_passed": state.final_quality.passed})

    def begin_publishing(self, state: ArticleAgentState, repo_url: str) -> ArticleAgentState:
        retrying_publish = (
            state.stage == ArticleAgentStage.FAILED
            and any(event.stage == ArticleAgentStage.PUBLISHING for event in state.events[-2:])
            and state.final_quality is not None
            and state.final_quality.passed
        )
        if state.stage != ArticleAgentStage.READY_TO_PUBLISH and not retrying_publish:
            raise ValueError("Article must pass joint quality checks before publishing")
        state.publication = {"repo_url": repo_url, "status": "publishing"}
        return self._transition(state, ArticleAgentStage.PUBLISHING, "GitHub publication started")

    def complete_publishing(self, state: ArticleAgentState, pr_url: str) -> ArticleAgentState:
        state.publication = {**state.publication, "status": "completed", "pr_url": pr_url}
        return self._transition(state, ArticleAgentStage.COMPLETED, "GitHub pull request created")

    def fail(self, state: ArticleAgentState, error: str) -> ArticleAgentState:
        if state.stage != ArticleAgentStage.FAILED:
            state.last_successful_stage = state.stage
        state.retry_count += 1
        state.error = " ".join(str(error).split())[:500]
        return self._transition(state, ArticleAgentStage.FAILED, "Agent stage failed", {"error": state.error})

    def _transition(
        self,
        state: ArticleAgentState,
        stage: ArticleAgentStage,
        summary: str,
        metadata: dict | None = None,
    ) -> ArticleAgentState:
        if stage != ArticleAgentStage.FAILED:
            state.last_successful_stage = stage
            state.error = ""
        state.stage = stage
        state.updated_at = datetime.now(UTC).isoformat()
        state.events.append(AgentEvent(stage=stage, summary=summary, metadata=metadata or {}))
        return self._save(state)

    def _save(self, state: ArticleAgentState) -> ArticleAgentState:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        target = self.state_dir / f"{state.run_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return state
