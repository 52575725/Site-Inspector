from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import Settings
from src.core.engine import Engine
from src.scheduler.jobs import ScheduledJobs

logger = logging.getLogger(__name__)


class SchedulerRunner:
    """Manage APScheduler lifecycle for daily/weekly jobs."""

    def __init__(self, settings: Settings, engine: Engine):
        self.settings = settings
        self.engine = engine
        self.scheduler = AsyncIOScheduler()
        self.jobs = ScheduledJobs(engine)

    def start(self) -> None:
        """Configure and start the scheduler with daily + weekly jobs."""
        settings = self.settings

        # Daily scan
        hour, minute = settings.daily_scan_time.split(":")
        self.scheduler.add_job(
            self.jobs.daily_scan_job,
            trigger=CronTrigger(hour=int(hour), minute=int(minute)),
            id="daily_scan",
            name="Daily site inspection scan",
            replace_existing=True,
        )

        # Weekly report
        day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        day = day_map.get(settings.weekly_report_day.lower(), 0)
        hour, minute = settings.weekly_report_time.split(":")
        self.scheduler.add_job(
            self.jobs.weekly_report_job,
            trigger=CronTrigger(day_of_week=day, hour=int(hour), minute=int(minute)),
            id="weekly_report",
            name="Weekly trend report",
            replace_existing=True,
        )

        # Sitemap submission (run weekly after report, e.g., Monday morning)
        sm_day = day_map.get(settings.weekly_report_day.lower(), 0)
        sm_hour = (int(hour) + 3) % 24
        self.scheduler.add_job(
            self.jobs.sitemap_submit_job,
            trigger=CronTrigger(day_of_week=sm_day, hour=sm_hour, minute=0),
            id="sitemap_submit",
            name="Submit sitemap to search engines",
            replace_existing=True,
        )

        # Verification check (run daily at a different time)
        v_hour = (int(hour) + 2) % 24
        self.scheduler.add_job(
            self.jobs.verification_check_job,
            trigger=CronTrigger(hour=v_hour, minute=0),
            id="verification_check",
            name="Post-fix verification check",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(
            f"Scheduler started: daily scan at {settings.daily_scan_time}, "
            f"weekly report on {settings.weekly_report_day} at {settings.weekly_report_time}"
        )

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    def list_jobs(self) -> list[dict]:
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time),
            }
            for job in self.scheduler.get_jobs()
        ]
