from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.integrations.google_analytics import GoogleAnalytics
from src.integrations.google_search_console import GoogleSearchConsole
from src.storage.repositories import VerificationRepository

logger = logging.getLogger(__name__)

PlanningGoal = Literal[
    "organic_visibility",
    "qualified_inquiries",
    "technical_health",
]


class SearchConsoleReader(Protocol):
    @property
    def available(self) -> bool: ...

    async def get_average_position(
        self, start_date: str, end_date: str, urls: list[str] | None = None,
    ) -> dict[str, float]: ...

    async def get_ctr(
        self, start_date: str, end_date: str, urls: list[str] | None = None,
    ) -> dict[str, float]: ...

    async def get_index_status(self, urls: list[str]) -> dict[str, bool]: ...


class AnalyticsReader(Protocol):
    @property
    def available(self) -> bool: ...

    async def get_pageviews(
        self, start_date: str, end_date: str, url_filter: str | None = None,
    ) -> dict[str, int]: ...

    async def get_avg_engagement_time(
        self, start_date: str, end_date: str, url_filter: str | None = None,
    ) -> dict[str, float]: ...

    async def close(self) -> None: ...


class BusinessObjective(BaseModel):
    goal: PlanningGoal = "organic_visibility"
    priority_url_patterns: list[str] = Field(default_factory=list)
    protected_url_patterns: list[str] = Field(default_factory=list)

    def business_value(self, url: str) -> float:
        if self.is_protected(url):
            return 0.35
        if self._matches(url, self.priority_url_patterns):
            return 0.95
        return 0.55

    def is_protected(self, url: str) -> bool:
        return self._matches(url, self.protected_url_patterns)

    @staticmethod
    def _matches(url: str, patterns: Sequence[str]) -> bool:
        path = urlparse(url).path or "/"
        return any(pattern and pattern in path for pattern in patterns)


class PageSignals(BaseModel):
    gsc_position: float | None = None
    gsc_ctr: float | None = None
    indexed: bool | None = None
    ga_pageviews: int | None = None
    ga_engagement_seconds: float | None = None

    @property
    def observed_fields(self) -> int:
        return sum(value is not None for value in self.model_dump().values())


class FeedbackSummary(BaseModel):
    improved: int = 0
    degraded: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.improved + self.degraded + self.unchanged

    @property
    def score(self) -> float:
        if not self.total:
            return 0.5
        weighted = self.improved + self.unchanged * 0.5
        return round(weighted / self.total, 3)

    @property
    def degradation_rate(self) -> float:
        return round(self.degraded / self.total, 3) if self.total else 0.0


class PlanningContext(BaseModel):
    objective: BusinessObjective = Field(default_factory=BusinessObjective)
    page_signals: dict[str, PageSignals] = Field(default_factory=dict)
    category_feedback: dict[str, FeedbackSummary] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def signals_for(self, url: str) -> PageSignals:
        return self.page_signals.get(url, PageSignals())

    def feedback_for(self, category: str) -> FeedbackSummary:
        return self.category_feedback.get(category, FeedbackSummary())


class PlanningContextCollector:
    """Collect optional outcome and traffic signals without changing the site."""

    def __init__(
        self,
        settings: Settings,
        session: AsyncSession,
        *,
        gsc: SearchConsoleReader | None = None,
        ga: AnalyticsReader | None = None,
    ):
        self.settings = settings
        self.verify_repo = VerificationRepository(session)
        self.gsc = gsc or GoogleSearchConsole(
            credentials_path=settings.google_credentials_path,
            site_url=settings.gsc_property,
        )
        self.ga = ga or GoogleAnalytics(
            credentials_path=settings.google_credentials_path,
            property_id=settings.ga4_property_id,
        )

    async def collect(self, issues: Sequence[Any]) -> PlanningContext:
        urls = sorted({
            str(getattr(issue, "url", ""))
            for issue in issues
            if getattr(issue, "url", None)
        })
        context = PlanningContext(objective=self._load_objective())
        context.category_feedback = await self._load_feedback()

        if not urls:
            if self.ga.available:
                await self.ga.close()
            return context

        today = date.today()
        start = (today - timedelta(days=28)).isoformat()
        end = today.isoformat()
        raw_signals: dict[str, dict[str, Any]] = defaultdict(dict)

        if self.gsc.available:
            try:
                positions = await self.gsc.get_average_position(start, end, urls)
                ctr = await self.gsc.get_ctr(start, end, urls)
                indexed = await self.gsc.get_index_status(urls)
                for url in urls:
                    if url in positions:
                        raw_signals[url]["gsc_position"] = positions[url]
                    if url in ctr:
                        raw_signals[url]["gsc_ctr"] = ctr[url]
                    if url in indexed:
                        raw_signals[url]["indexed"] = indexed[url]
            except Exception as exc:
                logger.warning("Planning GSC signals failed: %s", exc)
                context.warnings.append("Search Console signals were unavailable")
        else:
            context.warnings.append(
                "Search Console is not configured; opportunity uses scan evidence"
            )

        if self.ga.available:
            try:
                pageviews = await self.ga.get_pageviews(start, end)
                engagement = await self.ga.get_avg_engagement_time(start, end)
                by_path = {
                    self._normalise_path(key): value for key, value in pageviews.items()
                }
                engagement_by_path = {
                    self._normalise_path(key): value for key, value in engagement.items()
                }
                for url in urls:
                    path = self._normalise_path(urlparse(url).path)
                    if path in by_path:
                        raw_signals[url]["ga_pageviews"] = by_path[path]
                    if path in engagement_by_path:
                        raw_signals[url]["ga_engagement_seconds"] = engagement_by_path[path]
            except Exception as exc:
                logger.warning("Planning GA signals failed: %s", exc)
                context.warnings.append("Analytics signals were unavailable")
            finally:
                await self.ga.close()
        else:
            context.warnings.append(
                "Analytics is not configured; business value uses target policy"
            )

        context.page_signals = {
            url: PageSignals(**raw_signals.get(url, {}))
            for url in urls
            if raw_signals.get(url)
        }
        return context

    async def _load_feedback(self) -> dict[str, FeedbackSummary]:
        rows = await self.verify_repo.get_outcome_counts_by_category(
            self.settings.target_name,
        )
        grouped: dict[str, dict[str, int]] = defaultdict(dict)
        for category, status, count in rows:
            grouped[category][status] = count
        return {
            category: FeedbackSummary(**counts)
            for category, counts in grouped.items()
        }

    def _load_objective(self) -> BusinessObjective:
        target = self.settings.load_target(self.settings.target_name)
        raw = target.get("planning", {})
        if not isinstance(raw, dict):
            return BusinessObjective()
        try:
            return BusinessObjective.model_validate(raw)
        except Exception as exc:
            logger.warning("Invalid target planning policy: %s", exc)
            return BusinessObjective()

    @staticmethod
    def _normalise_path(value: str) -> str:
        path = urlparse(value).path if "://" in value else value
        path = "/" + str(PurePosixPath(path or "/")).strip("/")
        return "/" if path == "/." else path.rstrip("/") or "/"
