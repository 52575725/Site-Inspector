from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GoogleSearchConsole:
    """Async-friendly wrapper around Google Search Console API."""

    SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

    def __init__(self, credentials_path: Optional[Path] = None,
                 site_url: Optional[str] = None):
        self.credentials_path = credentials_path
        self.site_url = site_url
        self._service = None
        self._available = False

        if credentials_path and site_url:
            self._init_service()

    def _init_service(self) -> bool:
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=self.SCOPES,
            )
            self._service = build("searchconsole", "v1", credentials=credentials)
            self._available = True
            return True
        except ImportError:
            logger.warning("google-api-python-client not installed, GSC disabled")
        except FileNotFoundError:
            logger.warning(f"GSC credentials not found: {self.credentials_path}")
        except Exception as e:
            logger.warning(f"GSC init failed: {e}")
        return False

    @property
    def available(self) -> bool:
        return self._available

    async def get_average_position(
        self, start_date: str, end_date: str, urls: list[str] | None = None,
    ) -> dict[str, float]:
        """Get average position per URL in date range.

        Returns {url: position} mapping.
        """
        if not self._available:
            return {}

        return await asyncio.to_thread(
            self._query_position, start_date, end_date, urls,
        )

    def _query_position(self, start_date: str, end_date: str,
                        urls: list[str] | None) -> dict[str, float]:
        try:
            body = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": ["page"],
                "rowLimit": 500,
            }
            if urls:
                body["dimensionFilterGroups"] = [{
                    "filters": [{
                        "dimension": "page",
                        "operator": "includingRegex",
                        "expression": "|".join(re.escape(u) for u in urls),
                    }],
                }]

            response = self._service.searchanalytics().query(
                siteUrl=self.site_url, body=body,
            ).execute()

            return {
                row["keys"][0]: row.get("position", 0)
                for row in response.get("rows", [])
            }
        except Exception as e:
            logger.error(f"GSC query failed: {e}")
            return {}

    async def get_ctr(self, start_date: str, end_date: str,
                      urls: list[str] | None = None) -> dict[str, float]:
        """Get click-through rate per URL."""
        if not self._available:
            return {}
        return await asyncio.to_thread(
            self._query_ctr, start_date, end_date, urls,
        )

    def _query_ctr(self, start_date: str, end_date: str,
                   urls: list[str] | None) -> dict[str, float]:
        try:
            body = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": ["page"],
                "rowLimit": 500,
            }
            if urls:
                body["dimensionFilterGroups"] = [{
                    "filters": [{
                        "dimension": "page",
                        "operator": "includingRegex",
                        "expression": "|".join(re.escape(u) for u in urls),
                    }],
                }]

            response = self._service.searchanalytics().query(
                siteUrl=self.site_url, body=body,
            ).execute()

            return {
                row["keys"][0]: row.get("ctr", 0)
                for row in response.get("rows", [])
            }
        except Exception as e:
            logger.error(f"GSC CTR query failed: {e}")
            return {}

    async def get_index_status(self, urls: list[str]) -> dict[str, bool]:
        """Check if URLs are indexed."""
        if not self._available:
            return {}
        return await asyncio.to_thread(self._check_indexed, urls)

    def _check_indexed(self, urls: list[str]) -> dict[str, bool]:
        try:
            result = {}
            body = {
                "startDate": (date.today() - timedelta(days=30)).isoformat(),
                "endDate": date.today().isoformat(),
                "dimensions": ["page"],
                "dimensionFilterGroups": [{
                    "filters": [{
                        "dimension": "page",
                        "operator": "includingRegex",
                        "expression": "|".join(re.escape(u) for u in urls),
                    }],
                }],
                "rowLimit": 500,
            }
            response = self._service.searchanalytics().query(
                siteUrl=self.site_url, body=body,
            ).execute()

            indexed = {row["keys"][0] for row in response.get("rows", [])
                       if row.get("impressions", 0) > 0}
            return {url: url in indexed for url in urls}
        except Exception as e:
            logger.error(f"GSC index check failed: {e}")
            return {url: False for url in urls}
