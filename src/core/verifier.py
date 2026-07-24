from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.integrations.google_analytics import GoogleAnalytics
from src.integrations.google_search_console import GoogleSearchConsole
from src.integrations.lighthouse_ci import LighthouseCI
from src.storage.models import Fix, Issue, Verification
from src.storage.repositories import (
    AuditLogRepository,
    FixRepository,
    VerificationRepository,
)

logger = logging.getLogger(__name__)


class Verifier:
    """Post-fix verification: track metrics, detect regressions, trigger rollbacks."""

    def __init__(self, settings: Settings, session: AsyncSession):
        self.settings = settings
        self.session = session
        self.fix_repo = FixRepository(session)
        self.verify_repo = VerificationRepository(session)
        self.audit_repo = AuditLogRepository(session)

        # Initialize integrations
        self.gsc = GoogleSearchConsole(
            credentials_path=settings.google_credentials_path,
            site_url=settings.gsc_property,
        )
        self.ga = GoogleAnalytics(
            credentials_path=settings.google_credentials_path,
            property_id=settings.ga4_property_id,
        )
        self.lighthouse = LighthouseCI(
            lighthouse_path=settings.lighthouse_path,
        )

    async def create_verifications(self, fixes: Sequence[Fix]) -> list[Verification]:
        """Create verification records for newly applied fixes."""
        created = []

        for fix in fixes:
            issue = fix.issue
            if not issue:
                continue

            url = issue.url
            now = datetime.utcnow()
            window_end = now + timedelta(days=self.settings.verification_observation_days)

            # Determine which metrics to track based on issue category
            metrics_to_track = self._metrics_for_category(issue.inspector, issue.category)

            for metric in metrics_to_track:
                baseline = await self._get_baseline(metric, url)

                v = await self.verify_repo.create(
                    fix_id=fix.id,
                    metric_name=metric,
                    value_before=baseline,
                    window_start=now,
                    window_end=window_end,
                    status="pending",
                )
                created.append(v)

        await self.session.commit()
        logger.info(f"Created {len(created)} verification records for {len(fixes)} fixes")
        return created

    async def check_pending_verifications(self) -> dict:
        """Check all pending verifications and update statuses.

        Returns rollback and manual-intervention counts separately.
        """
        pending = await self.verify_repo.get_pending()
        today = date.today()
        checkpoint_days = self.settings.verification_checkpoints

        degraded_count = 0
        improved_count = 0
        rollback_count = 0
        manual_rollback_count = 0

        for v in pending:
            days_elapsed = (today - v.window_start.date()).days

            # Check at specified checkpoints
            if days_elapsed not in checkpoint_days and days_elapsed < max(checkpoint_days):
                continue

            after_value = await self._measure_metric(v.metric_name, v.fix.issue.url)

            if after_value is None:
                continue

            change = (after_value - v.value_before) / max(v.value_before, 0.001)

            # Degradation check
            threshold = self.settings.verification_degradation_threshold
            if self._is_degraded(v.metric_name, change, threshold):
                await self.verify_repo.update_status(v.id, "degraded", after_value)
                degraded_count += 1
                await self.audit_repo.log(
                    "verification_degraded", "verification", v.id,
                    {"metric": v.metric_name, "before": v.value_before,
                     "after": after_value, "change_pct": change},
                )

                # Trigger rollback
                if await self._rollback_fix(v.fix):
                    rollback_count += 1
                else:
                    manual_rollback_count += 1

            elif self._is_improved(v.metric_name, change, threshold):
                await self.verify_repo.update_status(v.id, "improved", after_value)
                improved_count += 1
            else:
                await self.verify_repo.update_status(v.id, "unchanged", after_value)

            # If we're past the full observation window, close out
            if days_elapsed >= max(checkpoint_days):
                pass  # already updated above

        await self.session.commit()
        logger.info(f"Verification check: {improved_count} improved, {degraded_count} degraded")
        return {
            "rollback_count": rollback_count,
            "manual_rollback_count": manual_rollback_count,
            "improved_count": improved_count,
            "degraded_count": degraded_count,
        }

    @staticmethod
    def _is_degraded(metric_name: str, change: float, threshold: float) -> bool:
        """Check if metric change indicates degradation.

        For LCP/TTFB/position: lower is better (positive change = worse).
        For CTR/pageviews: higher is better (negative change = worse).
        """
        lower_is_better = {"lcp_ms", "cls", "ttfb_ms", "gsc_position", "inp_ms"}
        if metric_name in lower_is_better:
            return change > threshold  # increased = worse
        return change < -threshold  # decreased = worse

    @staticmethod
    def _is_improved(metric_name: str, change: float, threshold: float) -> bool:
        """Check if metric change indicates improvement (symmetric to _is_degraded)."""
        lower_is_better = {"lcp_ms", "cls", "ttfb_ms", "gsc_position", "inp_ms"}
        if metric_name in lower_is_better:
            return change < -threshold  # decreased = better
        return change > threshold  # increased = better

    def _metrics_for_category(self, inspector: str, category: str) -> list[str]:
        mapping = {
            "performance": ["lcp_ms", "cls"],
            "seo": ["gsc_position", "gsc_ctr"],
            "content_quality": ["ga_pageviews", "ga_engagement"],
            "mobile": ["ga_pageviews"],
            "accessibility": ["ga_pageviews"],
            "broken_links": ["ga_pageviews"],
        }
        return mapping.get(inspector, [])

    async def _get_baseline(self, metric: str, url: str) -> float:
        """Get baseline metric value before fix."""
        today = date.today()
        start = (today - timedelta(days=14)).isoformat()
        end = today.isoformat()

        if metric == "gsc_position":
            data = await self.gsc.get_average_position(start, end, [url])
            return data.get(url, 0)
        if metric == "gsc_ctr":
            data = await self.gsc.get_ctr(start, end, [url])
            return data.get(url, 0)
        if metric == "ga_pageviews":
            data = await self.ga.get_pageviews(start, end, url)
            return sum(data.values())
        if metric == "ga_engagement":
            data = await self.ga.get_avg_engagement_time(start, end, url)
            vals = list(data.values())
            return sum(vals) / len(vals) if vals else 0
        if metric in ("lcp_ms", "cls", "ttfb_ms"):
            cwv = await self.lighthouse.measure_cwv(url)
            return cwv.get(metric, 0)

        return 0

    async def _measure_metric(self, metric: str, url: str) -> float | None:
        """Measure current metric value."""
        today = date.today()
        start = (today - timedelta(days=7)).isoformat()
        end = today.isoformat()
        return await self._get_baseline(metric, url)

    async def _rollback_fix(self, fix: Fix) -> bool:
        """Request manual rollback when no restorable deployment artifact exists."""
        logger.warning(f"Rollback required for fix #{fix.id} on {fix.issue.url}")

        # Update issue status
        await self.session.execute(
            update(Issue).where(Issue.id == fix.issue_id).values(status="rollback_required")
        )

        # Log rollback
        await self.audit_repo.log(
            "rollback_required", "fix", fix.id,
            {
                "issue_id": fix.issue_id,
                "url": fix.issue.url,
                "reason": "No commit hash or deployment adapter is attached to this fix",
            },
        )
        return False
