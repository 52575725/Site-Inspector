from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ArticleAgentStage(StrEnum):
    RESEARCHING = "researching"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    WRITING = "writing"
    CONTENT_REVIEW = "content_review"
    ARTICLE_READY = "article_ready"
    IMAGE_PLANNING = "image_planning"
    AWAITING_IMAGE_SELECTION = "awaiting_image_selection"
    IMAGE_REVIEW = "image_review"
    READY_TO_PUBLISH = "ready_to_publish"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentEvent(BaseModel):
    stage: ArticleAgentStage
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityCheck(BaseModel):
    name: str
    passed: bool
    severity: str = "warning"
    message: str


class QualityReport(BaseModel):
    passed: bool
    checks: list[QualityCheck] = Field(default_factory=list)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)


class ImagePlacementSlot(BaseModel):
    kind: str
    heading: str
    visual_brief: str
    slot_id: str = ""
    image_type: str = "photo"
    section_excerpt: str = ""
    search_query: str = ""
    insertion_reason: str = ""
    chart_spec: dict[str, Any] = Field(default_factory=dict)
    section_index: int = -1


class ImagePlan(BaseModel):
    target_count: int = Field(ge=3, le=10)
    existing_count: int = Field(ge=0)
    needed_count: int = Field(ge=0)
    article_title: str
    section_count: int = Field(ge=0)
    placement_slots: list[ImagePlacementSlot] = Field(default_factory=list)
    rationale: str


class ArticleAgentState(BaseModel):
    run_id: str
    website_url: str = ""
    research_id: str = ""
    article_id: str = ""
    stage: ArticleAgentStage
    last_successful_stage: ArticleAgentStage | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    research_summary: dict[str, Any] = Field(default_factory=dict)
    writing_summary: dict[str, Any] = Field(default_factory=dict)
    revision_count: int = 0
    image_plan: ImagePlan | None = None
    content_quality: QualityReport | None = None
    final_quality: QualityReport | None = None
    publication: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    retry_count: int = 0
    events: list[AgentEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
