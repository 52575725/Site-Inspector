from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from config.settings import Settings
from src.agents.planning_context import PlanningContextCollector
from src.storage.models import Fix, Issue, PageScan, Scan, Target, Verification
from src.storage.repositories import VerificationRepository


class FakeGSC:
    available = True

    async def get_average_position(self, start_date, end_date, urls=None):
        return {urls[0]: 8.0}

    async def get_ctr(self, start_date, end_date, urls=None):
        return {urls[0]: 0.015}

    async def get_index_status(self, urls):
        return {urls[0]: True}


class FakeGA:
    available = True

    def __init__(self):
        self.closed = False

    async def get_pageviews(self, start_date, end_date, url_filter=None):
        return {"/products/bar/": 240}

    async def get_avg_engagement_time(self, start_date, end_date, url_filter=None):
        return {"/products/bar": 42.5}

    async def close(self):
        self.closed = True


class FakeVerificationRepository:
    async def get_outcome_counts_by_category(self, target_name):
        assert target_name == "helinsilver"
        return [
            ("missing_title", "improved", 3),
            ("missing_title", "degraded", 1),
        ]


@pytest.mark.asyncio
async def test_collector_combines_business_traffic_and_outcome_signals(test_session):
    ga = FakeGA()
    collector = PlanningContextCollector(
        Settings(),
        test_session,
        gsc=FakeGSC(),
        ga=ga,
    )
    collector.verify_repo = FakeVerificationRepository()
    url = "https://www.helinsilver.com/products/bar/"

    context = await collector.collect([
        SimpleNamespace(url=url, category="missing_title"),
    ])

    signals = context.signals_for(url)
    assert signals.gsc_position == 8.0
    assert signals.gsc_ctr == 0.015
    assert signals.indexed is True
    assert signals.ga_pageviews == 240
    assert signals.ga_engagement_seconds == 42.5
    assert context.feedback_for("missing_title").score == 0.75
    assert context.objective.goal == "qualified_inquiries"
    assert context.objective.business_value(url) == 0.95
    assert ga.closed


@pytest.mark.asyncio
async def test_collector_closes_analytics_when_no_issue_urls(test_session):
    ga = FakeGA()
    collector = PlanningContextCollector(
        Settings(),
        test_session,
        gsc=FakeGSC(),
        ga=ga,
    )
    collector.verify_repo = FakeVerificationRepository()

    context = await collector.collect([])

    assert context.page_signals == {}
    assert ga.closed


@pytest.mark.asyncio
async def test_feedback_repository_groups_completed_outcomes_by_category(test_session):
    target = Target(name="feedback-test", base_url="https://example.com")
    test_session.add(target)
    await test_session.flush()
    scan = Scan(target_id=target.id, status="completed")
    test_session.add(scan)
    await test_session.flush()
    page = PageScan(scan_id=scan.id, url="https://example.com/a/")
    test_session.add(page)
    await test_session.flush()
    issue = Issue(
        scan_id=scan.id,
        page_scan_id=page.id,
        url=page.url,
        inspector="seo",
        category="missing_title",
        title="Missing title",
    )
    test_session.add(issue)
    await test_session.flush()
    fix = Fix(
        issue_id=issue.id,
        scan_id=scan.id,
        fixer="meta_fixer",
        fix_type="fully_auto",
        status="applied",
    )
    test_session.add(fix)
    await test_session.flush()
    now = datetime.now()
    test_session.add_all([
        Verification(
            fix_id=fix.id,
            metric_name="gsc_ctr",
            value_before=0.01,
            value_after=0.02,
            window_start=now,
            window_end=now + timedelta(days=14),
            status="improved",
        ),
        Verification(
            fix_id=fix.id,
            metric_name="gsc_position",
            value_before=8,
            value_after=8,
            window_start=now,
            window_end=now + timedelta(days=14),
            status="unchanged",
        ),
        Verification(
            fix_id=fix.id,
            metric_name="pending_metric",
            value_before=1,
            window_start=now,
            window_end=now + timedelta(days=14),
            status="pending",
        ),
    ])
    await test_session.flush()

    rows = await VerificationRepository(
        test_session,
    ).get_outcome_counts_by_category("feedback-test")

    assert sorted(rows) == [
        ("missing_title", "improved", 1),
        ("missing_title", "unchanged", 1),
    ]
