from __future__ import annotations

import asyncio

import pytest

from src.inspectors.competitor_gap import CompetitorGapInspector


@pytest.mark.asyncio
async def test_competitor_profiles_are_fetched_once_under_concurrency(monkeypatch):
    inspector = CompetitorGapInspector(competitor_urls=["https://competitor.example"])
    calls = 0

    async def fake_fetch_all():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)

    monkeypatch.setattr(inspector, "_fetch_all_competitors", fake_fetch_all)

    await asyncio.gather(*[
        inspector.inspect(f"https://example.com/{index}", "<html><body>Page</body></html>")
        for index in range(5)
    ])

    assert calls == 1
