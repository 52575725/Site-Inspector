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
async def test_run_researches_first_eight_diversified_keywords(monkeypatch):
    researcher = UserIntentResearcher(
        Settings(),
        http_client=object(),
        ai_client=object(),
    )
    researched: list[str] = []

    async def fake_research(keyword, **kwargs):
        researched.append(keyword)
        return KeywordIntent(keyword=keyword)

    async def fake_generate_queries(*args, **kwargs):
        return {}

    async def fake_generate_ideas(*args, **kwargs):
        return []

    async def fake_validate(*args, **kwargs):
        return None

    async def fake_validate_url(url):
        return url

    monkeypatch.setattr("src.ai.user_intent_researcher.validate_public_http_url", fake_validate_url)
    monkeypatch.setattr(researcher, "_research_keyword_intent", fake_research)
    monkeypatch.setattr(researcher, "_generate_ai_user_queries", fake_generate_queries)
    monkeypatch.setattr(researcher, "_validate_simulated_queries", fake_validate)
    monkeypatch.setattr(researcher, "_generate_article_ideas", fake_generate_ideas)

    await researcher.run(
        "https://example.com",
        [f"keyword {index}" for index in range(12)],
    )

    assert researched == [f"keyword {index}" for index in range(8)]


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


def test_autocomplete_matching_tolerates_numeric_and_plural_variants():
    researcher = UserIntentResearcher(
        Settings(),
        http_client=object(),
        ai_client=object(),
    )

    query = researcher._infer_user_query(
        "how much can i sell a 100 oz silver bar for",
        "99.99% silver bars for sale",
        source="google-autocomplete",
    )

    assert query is not None
    assert query.validation_status == "search-observed"
    assert query.source == "google-autocomplete"
