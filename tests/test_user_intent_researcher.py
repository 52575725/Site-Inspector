from __future__ import annotations

import pytest

from config.settings import Settings
from src.ai.user_intent_researcher import (
    KeywordIntent,
    TrendCandidate,
    UserIntentResearcher,
    UserQuery,
)


@pytest.mark.asyncio
async def test_simulated_queries_require_live_search_validation(monkeypatch):
    researcher = UserIntentResearcher(
        Settings(),
        http_client=object(),
        ai_client=object(),
    )
    validated = UserQuery(
        query="how to improve warehouse inventory accuracy",
        keyword="warehouse inventory software",
        intent="informational",
        format="how-to",
        source="AI-simulated",
    )
    unsupported = UserQuery(
        query="unsupported invented warehouse query",
        keyword="warehouse inventory software",
        intent="informational",
        format="question",
        source="AI-simulated",
    )

    async def fake_search(query):
        if query == validated.query:
            return [TrendCandidate(
                query=query,
                title="Warehouse Accuracy Guide",
                url="https://reference.example/accuracy",
            )]
        return []

    monkeypatch.setattr(researcher, "_search_web", fake_search)
    intents = [KeywordIntent(
        keyword="warehouse inventory software",
        user_queries=[validated, unsupported],
    )]

    await researcher._validate_simulated_queries(intents, limit=10)

    assert validated.validation_status == "search-validated"
    assert validated.evidence_urls == ["https://reference.example/accuracy"]
    assert validated.estimated_volume == ""
    assert unsupported.validation_status == "unverified"
    assert unsupported.evidence_urls == []
