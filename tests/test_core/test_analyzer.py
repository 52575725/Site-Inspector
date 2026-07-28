from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.core.analyzer import Analyzer
from src.storage.models import Issue, PageScan, Scan, Target


@pytest.fixture
def analyzer_settings():
    return Settings.load()


@pytest.mark.asyncio
async def test_p0_classification(test_session: AsyncSession, analyzer_settings):
    """Critical issues on important pages should be P0."""
    analyzer = Analyzer(analyzer_settings, test_session)

    # Create a missing_title on the homepage
    issue = Issue(
        scan_id=1, page_scan_id=1,
        url="https://www.helinsilver.com/",
        inspector="seo", category="missing_title",
        description="No title tag",
    )

    impact = analyzer._calculate_impact(issue, 29, 29)
    severity = analyzer._calculate_severity(issue)
    roi = analyzer._calculate_roi(issue)

    score = (
        impact * 0.40 + severity * 0.35 + roi * 0.25
    )

    # Should be P0 (missing_title is 0.95 severity, fully auto ROI 0.9, landing page 0.7)
    assert score >= 0.70
    assert analyzer._classify_tier(score, issue.category) == "P0"


def test_high_score_non_blocker_is_not_p0(analyzer_settings, test_session):
    analyzer = Analyzer(analyzer_settings, test_session)

    assert analyzer._classify_tier(0.90, "missing_content_security_policy") == "P1"


@pytest.mark.asyncio
async def test_p3_classification(test_session: AsyncSession, analyzer_settings):
    """Minor issues on single pages should be P3."""
    analyzer = Analyzer(analyzer_settings, test_session)

    issue = Issue(
        scan_id=1, page_scan_id=1,
        url="https://www.helinsilver.com/blog/old-post",
        inspector="performance", category="optimize_unused_css",
        description="Unused CSS",
    )

    impact = analyzer._calculate_impact(issue, 1, 29)
    severity = analyzer._calculate_severity(issue)
    roi = analyzer._calculate_roi(issue)

    score = (
        impact * 0.40 + severity * 0.35 + roi * 0.25
    )

    assert score < 0.70  # Not P0/P1
