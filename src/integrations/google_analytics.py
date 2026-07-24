from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GoogleAnalytics:
    """Async-friendly wrapper around Google Analytics Data API (GA4)."""

    SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]

    def __init__(self, credentials_path: Optional[Path] = None,
                 property_id: Optional[str] = None):
        self.credentials_path = credentials_path
        self.property_id = property_id
        self._client = None
        self._available = False

        if credentials_path and property_id:
            self._init_client()

    def _init_client(self) -> bool:
        try:
            from google.oauth2 import service_account
            from google.analytics.data_v1beta import BetaAnalyticsDataAsyncClient

            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=self.SCOPES,
            )
            self._client = BetaAnalyticsDataAsyncClient(credentials=credentials)
            self._available = True
            return True
        except ImportError:
            logger.warning("google-analytics-data not installed, GA4 disabled")
        except FileNotFoundError:
            logger.warning(f"GA credentials not found: {self.credentials_path}")
        except Exception as e:
            logger.warning(f"GA4 init failed: {e}")
        return False

    @property
    def available(self) -> bool:
        return self._available

    async def get_pageviews(self, start_date: str, end_date: str,
                            url_filter: str | None = None) -> dict[str, int]:
        """Get pageviews per URL in date range."""
        if not self._available:
            return {}

        try:
            from google.analytics.data_v1beta.types import (
                DateRange,
                Dimension,
                Filter,
                FilterExpression,
                Metric,
                RunReportRequest,
            )

            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[Dimension(name="pagePath")],
                metrics=[Metric(name="screenPageViews")],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            )

            if url_filter:
                request.dimension_filter = FilterExpression(
                    filter=Filter(
                        field_name="pagePath",
                        string_filter=Filter.StringFilter(
                            match_type=Filter.StringFilter.MatchType.CONTAINS,
                            value=url_filter,
                        ),
                    ),
                )

            response = await self._client.run_report(request)
            return {
                row.dimension_values[0].value: int(row.metric_values[0].value)
                for row in response.rows
            }
        except Exception as e:
            logger.error(f"GA4 pageviews query failed: {e}")
            return {}

    async def get_avg_engagement_time(self, start_date: str, end_date: str,
                                      url_filter: str | None = None) -> dict[str, float]:
        """Get average engagement time per URL."""
        if not self._available:
            return {}

        try:
            from google.analytics.data_v1beta.types import (
                DateRange,
                Dimension,
                Metric,
                RunReportRequest,
            )

            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[Dimension(name="pagePath")],
                metrics=[Metric(name="averageSessionDuration")],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            )

            if url_filter:
                from google.analytics.data_v1beta.types import Filter, FilterExpression
                request.dimension_filter = FilterExpression(
                    filter=Filter(
                        field_name="pagePath",
                        string_filter=Filter.StringFilter(
                            match_type=Filter.StringFilter.MatchType.CONTAINS,
                            value=url_filter,
                        ),
                    ),
                )

            response = await self._client.run_report(request)
            return {
                row.dimension_values[0].value: float(row.metric_values[0].value)
                for row in response.rows
            }
        except Exception as e:
            logger.error(f"GA4 engagement time query failed: {e}")
            return {}

    async def close(self) -> None:
        if self._client:
            await self._client.close()
