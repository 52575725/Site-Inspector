from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.core.engine import Engine

logger = logging.getLogger(__name__)


class ScheduledJobs:
    """APScheduler job functions."""

    def __init__(self, engine: Engine):
        self.engine = engine

    async def daily_scan_job(self) -> None:
        """Job: run daily full inspection scan."""
        logger.info(f"Scheduled daily scan starting at {datetime.utcnow()}")
        try:
            result = await self.engine.run_daily_scan()
            logger.info(f"Daily scan completed: {result}")
            from src.notifications.notifier import Notifier
            notifier = Notifier(self.engine.settings)
            await notifier.notify_scan_complete(
                target_name=self.engine.settings.target_name,
                pages=result.get("pages", 0),
                issues=result.get("new_issues", 0),
                fixes=result.get("fixes_applied", 0),
                report_path=result.get("report_path", ""),
            )
        except Exception as e:
            logger.error(f"Daily scan job failed: {e}", exc_info=True)
            from src.notifications.notifier import Notifier
            notifier = Notifier(self.engine.settings)
            await notifier.notify_error(self.engine.settings.target_name, str(e))

    async def weekly_report_job(self) -> None:
        """Job: generate weekly trend report."""
        logger.info(f"Scheduled weekly report at {datetime.utcnow()}")
        try:
            result = await self.engine.run_weekly_report()
            logger.info(f"Weekly report completed: {result}")
        except Exception as e:
            logger.error(f"Weekly report job failed: {e}", exc_info=True)

    async def sitemap_submit_job(self) -> None:
        """Job: submit sitemap to Google and Bing for faster indexing."""
        logger.info(f"Scheduled sitemap submission at {datetime.utcnow()}")
        try:
            result = await self.engine.run_sitemap_submit()
            logger.info(f"Sitemap submission completed: {result}")
        except Exception as e:
            logger.error(f"Sitemap submission job failed: {e}", exc_info=True)

    async def verification_check_job(self) -> None:
        """Job: check pending verifications and trigger rollbacks."""
        logger.info(f"Verification check at {datetime.utcnow()}")
        try:
            result = await self.engine.run_verification_check()
            if result.get("rollback_count", 0) > 0:
                logger.warning(f"Verification check: {result['rollback_count']} rollbacks triggered")
            else:
                logger.info(f"Verification check: {result}")
        except Exception as e:
            logger.error(f"Verification check job failed: {e}", exc_info=True)
