from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI

from config.settings import Settings
from src.ai.article_researcher import ResearchFinding, build_citation_prompt
from src.ai.automatic_article import (
    AutomaticArticleWorkflow,
    AutomaticResearchResult,
    EditorialDecision,
    ReferenceArticle,
    SiteProfile,
    TrendCandidate,
)
from src.web.routes import articles


def test_detects_primary_and_alternate_site_languages_from_html_evidence():
    homepage = AutomaticArticleWorkflow._extract_site_page(
        "https://example.com/",
        """<html lang="en"><head>
        <link rel="alternate" hreflang="en" href="https://example.com/">
        <link rel="alternate" hreflang="ja-JP" href="https://example.com/jp/">
        </head><body><h1>Silver market guidance</h1></body></html>""",
    )
    japanese = AutomaticArticleWorkflow._extract_site_page(
        "https://example.com/jp/",
        "<html lang='ja'><body><h1>日本語の記事と市場情報</h1></body></html>",
    )

    primary, detected, evidence = AutomaticArticleWorkflow._detect_site_languages(
        [homepage, japanese],
        requested_language="auto",
    )

    assert primary == "en"
    assert detected[:2] == ["en", "ja"]
    assert any("hreflang=ja" in item for item in evidence)


def test_rejects_exact_and_lightly_rephrased_existing_titles():
    existing = ["How to Verify LBMA Compliance When Sourcing Silver Bars from Hong Kong"]

    assert AutomaticArticleWorkflow._titles_too_similar(existing[0], existing)
    assert AutomaticArticleWorkflow._titles_too_similar(
        "LBMA Compliance Checklist for Silver Bars Sourced from Hong Kong Exporters",
        existing,
    )
    assert not AutomaticArticleWorkflow._titles_too_similar(
        "Why Electronics Manufacturing Is Increasing Industrial Silver Demand",
        existing,
    )


def test_auto_editorial_search_fallback_is_broader_than_commercial_topics():
    queries = AutomaticArticleWorkflow._build_editorial_search_queries(
        "precious metals",
        ["industrial silver demand", "silver supply"],
        content_direction="auto",
    )

    assert any("latest news" in query for query in queries)
    assert any("industry trends" in query for query in queries)
    assert any("market update" in query for query in queries)
    assert any("data analysis" in query for query in queries)
    assert any("origin history" in query for query in queries)
    assert any("culture symbolism" in query for query in queries)
    assert any("unexpected uses" in query for query in queries)


def test_profile_keywords_keep_user_hints_and_limit_trade_location_cluster():
    narrow = [
        "Hong Kong silver supplier",
        "Hong Kong silver buying guide",
        "silver customs clearance",
        "silver import rules",
        "silver export compliance",
        "silver wholesale sourcing",
        "silver freight shipping",
        "silver tariff guide",
    ]
    broad = [
        "how silver tarnish forms",
        "silver jewelry origin stories",
        "traditional silver craftsmanship",
        "silver in contemporary design",
        "unexpected uses of silver",
        "silver conservation science",
        "famous silver objects in history",
        "silver recycling innovations",
        "silver symbolism across cultures",
        "future of wearable silver",
    ]

    result = AutomaticArticleWorkflow._diversify_profile_keywords(
        narrow + broad + ["silver in historical mysteries"],
        explicit_keywords=["custom silver keepsakes"],
        limit=30,
    )

    assert result[0] == "custom silver keepsakes"
    assert "silver in historical mysteries" in result
    assert sum(item in narrow for item in result) <= 6
    assert sum(item in narrow for item in result[:8]) <= 2


@pytest.mark.asyncio
async def test_build_profile_keeps_up_to_thirty_ai_keywords():
    generated_keywords = [f"distinct silver editorial angle {index}" for index in range(25)]

    class FakeAI:
        async def generate_json(self, prompt, **kwargs):
            assert "20-30" in prompt
            assert "non-exhaustive" in prompt
            return {
                "site_name": "Example Silver",
                "business_summary": "A silver jewelry studio.",
                "niche": "silver jewelry",
                "target_audience": ["collectors"],
                "offerings": ["silver jewelry"],
                "keywords": generated_keywords,
                "recommended_topic": "The changing meanings of silver jewelry",
            }

    workflow = AutomaticArticleWorkflow(
        Settings(deepseek_api_key="test"),
        http_client=object(),
        ai_client=FakeAI(),
    )
    profile = await workflow._build_profile(
        "https://example.com",
        [{
            "url": "https://example.com",
            "title": "Example Silver",
            "description": "A silver jewelry studio.",
            "headings": [],
        }],
        language="en",
        topic_hint="",
        keyword_hint="handmade silver gifts, silver family traditions",
    )

    assert profile.keywords[:2] == ["handmade silver gifts", "silver family traditions"]
    assert len(profile.keywords) == 27
    assert generated_keywords[-1] in profile.keywords


@pytest.mark.asyncio
async def test_ai_can_invent_open_ended_editorial_discovery_queries():
    class FakeAI:
        async def generate_json(self, prompt, **kwargs):
            assert "non-exhaustive" in prompt
            assert "explore freely" in prompt
            return {
                "queries": [
                    "silver jewelry in maritime wedding rituals oral histories",
                    "how museum conservators diagnose tarnished silver",
                    "silver objects as clues in historical crime investigations",
                ]
            }

    workflow = AutomaticArticleWorkflow(
        Settings(deepseek_api_key="test"),
        http_client=object(),
        ai_client=FakeAI(),
    )
    queries = await workflow._discover_editorial_queries(
        SiteProfile(
            website_url="https://example.com",
            site_name="Example Silver",
            business_summary="A silver jewelry studio.",
            niche="silver jewelry",
            keywords=["silver jewelry"],
        ),
        content_direction="auto",
    )

    assert "how museum conservators diagnose tarnished silver" in queries
    assert any("crime investigations" in query for query in queries)


def test_open_topic_candidates_keep_novel_lenses_and_limit_procurement():
    candidates = AutomaticArticleWorkflow._clean_topic_candidates(
        [
            {
                "topic": "How to buy silver jewelry in Hong Kong",
                "headline": "A Hong Kong Silver Jewelry Buying Guide",
                "editorial_lens": "local buyer journey",
                "content_direction": "buyer_question",
            },
            {
                "topic": "How to choose a Hong Kong silver supplier",
                "headline": "Choosing a Hong Kong Silver Supplier",
                "editorial_lens": "supplier selection",
                "content_direction": "buyer_question",
            },
            {
                "topic": "What tarnish reveals to museum conservators",
                "headline": "Reading Silver Tarnish Like a Conservator",
                "editorial_lens": "forensic conservation",
                "content_direction": "deep_analysis",
            },
            {
                "topic": "Silver jewelry carried along old sea routes",
                "headline": "The Sea Routes Hidden Inside Silver Jewelry",
                "editorial_lens": "objects as maps of migration",
                "content_direction": "evergreen_guide",
            },
        ],
        excluded_topics=[],
        allowed_urls=set(),
        allowed_directions={
            "news", "industry_trend", "market_event", "evergreen_guide",
            "buyer_question", "deep_analysis",
        },
        allowed_types={"blog", "market_analysis", "product_review", "guide", "news", "landing"},
        enforce_open_diversity=True,
    )

    assert len(candidates) == 3
    assert [item["editorial_lens"] for item in candidates] == [
        "local buyer journey",
        "forensic conservation",
        "objects as maps of migration",
    ]


def test_fixed_direction_is_applied_to_every_candidate_without_commercial_limit():
    candidates = AutomaticArticleWorkflow._clean_topic_candidates(
        [
            {
                "topic": "Silver ring sizing questions",
                "editorial_lens": "fit and measurement",
                "content_direction": "buyer_question",
            },
            {
                "topic": "Silver hallmark questions",
                "editorial_lens": "authenticity checks",
                "content_direction": "deep_analysis",
            },
        ],
        excluded_topics=[],
        allowed_urls=set(),
        allowed_directions={"buyer_question"},
        allowed_types={"blog", "guide"},
        enforce_open_diversity=False,
        forced_direction="buyer_question",
    )

    assert len(candidates) == 2
    assert {item["content_direction"] for item in candidates} == {"buyer_question"}


def test_topic_candidates_are_distinct_and_exclude_existing_coverage():
    candidates = AutomaticArticleWorkflow._clean_topic_candidates(
        [
            {"topic": "Existing LBMA sourcing guide", "headline": "Existing LBMA sourcing guide"},
            {
                "topic": "Industrial silver demand from solar manufacturing",
                "headline": "How Solar Manufacturing Is Changing Silver Demand",
                "content_direction": "industry_trend",
                "page_type": "market_analysis",
                "recommended_outline": ["Demand signal", "Supply impact"],
            },
            {
                "topic": "Solar manufacturing and industrial silver demand",
                "headline": "Silver Demand From Solar",
                "content_direction": "industry_trend",
                "page_type": "blog",
            },
            {
                "topic": "New customs reporting rules for precious-metal imports",
                "headline": "What New Customs Reporting Rules Mean for Silver Buyers",
                "content_direction": "news",
                "page_type": "news",
                "source_urls": ["https://authority.example/update"],
            },
        ],
        excluded_topics=["Existing LBMA sourcing guide"],
        allowed_urls={"https://authority.example/update"},
        allowed_directions={
            "news", "industry_trend", "market_event", "evergreen_guide",
            "buyer_question", "deep_analysis",
        },
        allowed_types={"blog", "market_analysis", "product_review", "guide", "news", "landing"},
    )

    assert len(candidates) == 2
    assert {item["content_direction"] for item in candidates} == {"industry_trend", "news"}
    assert candidates[1]["source_urls"] == ["https://authority.example/update"]


def test_old_regime_effective_year_is_removed_from_evergreen_headline():
    candidates = AutomaticArticleWorkflow._clean_topic_candidates(
        [{
            "topic": "Understanding Hong Kong's Precious Metals Dealer Registration Regime",
            "headline": (
                "Hong Kong Silver Exporter Compliance: Navigating the 2023 "
                "Precious Metals Dealer Registration"
            ),
            "content_direction": "evergreen_guide",
            "page_type": "guide",
            "freshness_basis": "The regime took effect in 2023 and remains active.",
        }],
        excluded_topics=[],
        allowed_urls=set(),
        allowed_directions={
            "news", "industry_trend", "market_event", "evergreen_guide",
            "buyer_question", "deep_analysis",
        },
        allowed_types={"blog", "market_analysis", "product_review", "guide", "news", "landing"},
    )

    assert len(candidates) == 1
    assert "2023" not in candidates[0]["headline"]
    assert candidates[0]["freshness_basis"].endswith("remains active.")


def test_stale_news_candidate_is_dropped_but_current_historical_comparison_is_kept():
    current_year = datetime.now(UTC).year
    stale_year = current_year - 3
    common = {
        "excluded_topics": [],
        "allowed_urls": set(),
        "allowed_directions": {
            "news", "industry_trend", "market_event", "evergreen_guide",
            "buyer_question", "deep_analysis",
        },
        "allowed_types": {"blog", "market_analysis", "product_review", "guide", "news", "landing"},
    }
    candidates = AutomaticArticleWorkflow._clean_topic_candidates(
        [
            {
                "topic": f"Hong Kong precious metals rules update {stale_year}",
                "headline": f"{stale_year} Precious Metals Rules Update",
                "content_direction": "news",
                "page_type": "news",
            },
            {
                "topic": (
                    f"How Hong Kong compliance changed from {stale_year} to {current_year}"
                ),
                "headline": f"Hong Kong Compliance: {stale_year} vs {current_year}",
                "content_direction": "deep_analysis",
                "page_type": "market_analysis",
            },
        ],
        **common,
    )

    assert len(candidates) == 1
    assert candidates[0]["headline"] == (
        f"Hong Kong Compliance: {stale_year} vs {current_year}"
    )


def test_recent_titles_are_scoped_to_the_same_website(tmp_path, monkeypatch):
    monkeypatch.setattr(articles, "GENERATED_DIR", tmp_path)
    records = [
        {"id": "one", "website_url": "https://example.com/", "title": "Existing Guide", "topic": "Guide Topic"},
        {"id": "two", "website_url": "https://other.example/", "title": "Other Site Article"},
        {"id": "three", "website_url": "https://example.com/", "title": "Japanese Translation", "source_article_id": "one"},
    ]
    for record in records:
        (tmp_path / f"{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")

    titles = articles._recent_titles_for_website("https://www.example.com/")

    assert titles == ["Existing Guide", "Guide Topic"]


REFERENCE_HTML = """<!doctype html>
<html><head><title>Complete Logistics Guide</title>
<meta name="description" content="A detailed logistics guide."></head>
<body><nav>Navigation words should not count</nav><main>
<h1>Complete Logistics Guide</h1>
<p>{body}</p>
<h2>Planning</h2><p>{body}</p>
<h3>Documentation</h3><ul><li>Invoice</li><li>Packing list</li></ul>
<h2>Frequently Asked Questions</h2>
<table><tr><td>Mode</td><td>Timing</td></tr></table>
</main><footer>Footer words should not count</footer></body></html>""".format(
    body=" ".join(f"word{i}" for i in range(180)),
)


def test_extract_reference_article_keeps_structure_not_full_copy():
    result = AutomaticArticleWorkflow.extract_reference_article(
        "https://reference.example/guide",
        REFERENCE_HTML,
    )

    assert result.title == "Complete Logistics Guide"
    assert [item["level"] for item in result.headings] == ["h1", "h2", "h3", "h2"]
    assert result.word_count >= 360
    assert result.has_table is True
    assert result.has_faq is True
    assert result.list_count == 1
    assert not hasattr(result, "body")


def test_reference_article_counts_content_images_paragraphs_and_cta():
    html = """<html><head><title>Buyer Guide</title></head><body><article>
    <h1>Buyer Guide</h1><h2>Options</h2>
    <p>Useful buyer guidance.</p><p>More practical detail.</p>
    <img src="product.jpg" alt="Warehouse product">
    <img src="logo.svg" class="logo" alt="Company logo">
    <a href="/quote">Request a quote</a>
    </article></body></html>"""

    result = AutomaticArticleWorkflow.extract_reference_article(
        "https://reference.example/buyer-guide",
        html,
    )

    assert result.image_count == 1
    assert result.paragraph_count == 2
    assert result.has_cta is True


def test_writing_brief_uses_validated_queries_and_reference_benchmarks():
    profile = SiteProfile(
        website_url="https://example.com",
        site_name="Example",
        business_summary="Warehouse software.",
        niche="warehouse software",
        keywords=["warehouse inventory software"],
        recommended_topic="Warehouse accuracy",
    )
    references = [
        ReferenceArticle(
            url=f"https://reference.example/{index}",
            title=f"Guide {index}",
            description="",
            headings=[{"level": "h1", "text": "Guide"}, {"level": "h2", "text": "Steps"}],
            word_count=word_count,
            has_table=index == 1,
            has_faq=True,
            list_count=1,
            image_count=image_count,
            paragraph_count=12,
            has_cta=index == 1,
        )
        for index, (word_count, image_count) in enumerate(((1600, 3), (2200, 5)))
    ]
    intent_report = {
        "keywords_intents": [{
            "keyword": "warehouse inventory software",
            "user_queries": [
                {
                    "query": "how to improve warehouse inventory accuracy",
                    "source": "AI-simulated",
                    "validation_status": "search-validated",
                },
                {
                    "query": "invented query with no evidence",
                    "source": "AI-simulated",
                    "validation_status": "unverified",
                },
            ],
        }],
    }
    decision = EditorialDecision(
        topic="Warehouse Accuracy Guide",
        page_type="guide",
        headline_options=["How to Improve Warehouse Inventory Accuracy"],
        recommended_outline=["Causes", "Process", "FAQ"],
    )

    brief = AutomaticArticleWorkflow._build_writing_brief(
        profile,
        references,
        decision,
        intent_report,
    )

    assert brief["benchmark"]["median_word_count"] == 2200
    assert brief["benchmark"]["median_image_count"] == 5
    assert brief["image_count_min"] <= brief["image_count_max"]
    assert brief["target_queries"] == ["how to improve warehouse inventory accuracy"]
    assert "invented query with no evidence" not in brief["target_queries"]


def test_build_generation_context_marks_references_as_structure_only():
    research = AutomaticResearchResult(
        profile=SiteProfile(
            website_url="https://example.com",
            site_name="Example",
            business_summary="Example sells warehouse software.",
            niche="warehouse software",
            target_audience=["warehouse operators"],
            offerings=["inventory platform"],
            keywords=["warehouse inventory software"],
            recommended_topic="Warehouse inventory guide",
        ),
        references=[AutomaticArticleWorkflow.extract_reference_article(
            "https://reference.example/guide", REFERENCE_HTML,
        )],
        architecture_insights=["Most references use lists."],
        warnings=[],
        trend_candidates=[TrendCandidate(
            query="warehouse inventory trends 2026",
            title="Inventory Trends 2026",
            url="https://reference.example/trends",
        )],
        editorial_decision=EditorialDecision(
            topic="How automation improves warehouse inventory accuracy",
            page_type="market_analysis",
            reasoning="Current results emphasize automation trends and measurable impact.",
            trend_angle="2026 automation adoption",
            search_intent="informational",
            headline_options=["Warehouse Automation and Inventory Accuracy in 2026"],
            authority_source_urls=["https://reference.example/trends"],
            confidence=0.8,
        ),
        authority_sources=[TrendCandidate(
            query="warehouse official standards",
            title="Official Warehouse Standard",
            url="https://reference.example/standard",
            snippet="Official requirements for warehouse controls.",
        )],
    )

    context = AutomaticArticleWorkflow.build_generation_context(research)

    assert "structural and coverage signals" in context
    assert "never copy" in context.lower()
    assert "word179" not in context
    assert "https://reference.example/guide" in context
    assert "https://reference.example/standard" in context
    assert "clickable HTML link" in context
    assert "Never invent or alter a URL" in context


def test_unwraps_duckduckgo_result_url():
    wrapped = (
        "//duckduckgo.com/l/?uddg="
        "https%3A%2F%2Fexample.com%2Fuseful%3Fpage%3D1"
    )
    assert AutomaticArticleWorkflow._unwrap_search_url(wrapped) == (
        "https://example.com/useful?page=1"
    )


def test_word_count_supports_cjk_articles():
    assert AutomaticArticleWorkflow._count_words("仓库库存管理优化指南") == 10
    assert AutomaticArticleWorkflow._count_words("Silver 仓库 guide") == 4


def test_fallback_editor_selects_format_from_intent():
    profile = SiteProfile(
        website_url="https://example.com",
        site_name="Example",
        business_summary="Warehouse software.",
        niche="warehouse software",
        keywords=["warehouse market trends"],
        recommended_topic="Warehouse automation market forecast",
    )
    decision = AutomaticArticleWorkflow._fallback_editorial_decision(
        profile,
        topic_hint="",
        fixed_type="",
    )
    assert decision.page_type == "market_analysis"
    assert decision.topic == "Warehouse automation market forecast"


def test_research_report_serializes_trend_and_editorial_evidence():
    report = AutomaticResearchResult(
        profile=SiteProfile(
            website_url="https://example.com",
            site_name="Example",
            business_summary="Warehouse software.",
            niche="warehouse software",
        ),
        references=[],
        architecture_insights=[],
        warnings=[],
        trend_candidates=[TrendCandidate(
            query="warehouse trends 2026",
            title="Warehouse Trends",
            url="https://reference.example/trends",
        )],
        authority_sources=[TrendCandidate(
            query="warehouse official standard",
            title="Official Standard",
            url="https://reference.example/standard",
        )],
        editorial_decision=EditorialDecision(
            topic="Warehouse Trends to Watch",
            page_type="market_analysis",
            reasoning="The topic requires interpretation.",
        ),
    ).to_dict()
    assert report["editorial_decision"]["page_type"] == "market_analysis"
    assert report["trend_candidates"][0]["query"] == "warehouse trends 2026"
    assert report["authority_sources"][0]["title"] == "Official Standard"


def test_authority_urls_are_limited_to_actual_search_results():
    allowed = "https://authority.example/related-article"
    urls = AutomaticArticleWorkflow._clean_urls(
        [allowed, "https://invented.example/not-searched", allowed],
        allowed_urls={allowed},
        limit=5,
    )
    assert urls == [allowed]


def test_authoritative_citation_prompt_requires_clickable_exact_url():
    prompt = build_citation_prompt([ResearchFinding(
        title="Official Standard",
        url="https://authority.example/standard",
        snippet="The standard defines verified requirements.",
        source_label="Standards Authority",
        source_type="standards_body",
        found_at="2026-07-28T00:00:00",
    )])
    assert "clickable HTML link" in prompt
    assert "https://authority.example/standard" in prompt
    assert "Never invent or modify a URL" in prompt


@pytest.mark.asyncio
async def test_editorial_ai_selects_supported_type_and_respects_topic_hint():
    class FakeAI:
        async def generate_json(self, *args, **kwargs):
            return {
                "topic": "Ignored because the user supplied a topic",
                "page_type": "guide",
                "reasoning": "Readers need an actionable sequence.",
                "trend_angle": "New warehouse automation workflows",
                "content_direction": "industry_trend",
                "freshness_basis": "Current search results show new automation workflows.",
                "search_intent": "informational",
                "headline_options": ["A", "B", "C"],
                "confidence": 0.9,
            }

    workflow = AutomaticArticleWorkflow(
        Settings(deepseek_api_key="test"),
        http_client=object(),
        ai_client=FakeAI(),
    )
    profile = SiteProfile(
        website_url="https://example.com",
        site_name="Example",
        business_summary="Warehouse software.",
        niche="warehouse software",
        keywords=["warehouse automation"],
        recommended_topic="Warehouse automation",
    )
    decision = await workflow._choose_editorial_strategy(
        profile,
        references=[],
        trend_candidates=[],
        language="en",
        topic_hint="User-defined warehouse topic",
        requested_page_type="auto",
    )
    assert decision.topic == "User-defined warehouse topic"
    assert decision.page_type == "guide"
    assert decision.content_direction == "industry_trend"
    assert "search results" in decision.freshness_basis
    assert decision.topic_candidates[0]["topic"] == "User-defined warehouse topic"
    assert decision.confidence == 0.9


@pytest.mark.asyncio
async def test_auto_generate_endpoint_saves_research_report(tmp_path, monkeypatch):
    research = AutomaticResearchResult(
        profile=SiteProfile(
            website_url="https://example.com",
            site_name="Example",
            business_summary="Example provides warehouse software.",
            niche="warehouse software",
            target_audience=["warehouse teams"],
            offerings=["inventory platform"],
            keywords=["warehouse inventory software", "inventory accuracy"],
            recommended_topic="How to improve warehouse inventory accuracy",
            evidence_pages=["https://example.com"],
        ),
        references=[ReferenceArticle(
            url="https://reference.example/article",
            title="Inventory Accuracy Guide",
            description="",
            headings=[{"level": "h1", "text": "Inventory Accuracy Guide"}],
            word_count=1200,
            has_table=False,
            has_faq=False,
            list_count=2,
        )],
        architecture_insights=["Most references use lists."],
        warnings=[],
        trend_candidates=[TrendCandidate(
            query="warehouse inventory trends 2026",
            title="Inventory Trends 2026",
            url="https://reference.example/trends",
        )],
        editorial_decision=EditorialDecision(
            topic="How automation improves warehouse inventory accuracy",
            page_type="market_analysis",
            reasoning="Current results emphasize automation trends and measurable impact.",
            trend_angle="2026 automation adoption",
            search_intent="informational",
            authority_source_urls=["https://reference.example/trends"],
            confidence=0.8,
        ),
        authority_sources=[TrendCandidate(
            query="warehouse official source",
            title="Warehouse Standards Authority",
            url="https://reference.example/trends",
            snippet="Official warehouse control guidance.",
        )],
    )

    class FakeWorkflow:
        def __init__(self, settings):
            self.settings = settings

        async def run(self, *args, **kwargs):
            assert kwargs["requested_page_type"] == "auto"
            assert kwargs["excluded_topics"] == []
            assert kwargs["content_direction"] == "auto"
            return research

        async def close(self):
            return None

        build_generation_context = staticmethod(AutomaticArticleWorkflow.build_generation_context)

    class FakeDeepSeek:
        def __init__(self, **kwargs):
            pass

        async def generate_text(self, prompt, **kwargs):
            assert "warehouse inventory software" in prompt
            assert "structural and coverage signals" in prompt
            assert "https://reference.example/trends" in prompt
            assert "clickable HTML link" in prompt
            return (
                "<html><head><title>Warehouse Accuracy</title></head><body>"
                "<article><h1 onclick=\"alert(1)\">Warehouse Accuracy</h1>"
                "<a href=\"javascript:alert(1)\">Read</a><script>alert(1)</script>"
                "</article></body></html>"
            )

        async def close(self):
            return None

    import src.ai.automatic_article as workflow_module
    import src.ai.deepseek_client as deepseek_module

    monkeypatch.setattr(workflow_module, "AutomaticArticleWorkflow", FakeWorkflow)
    monkeypatch.setattr(deepseek_module, "DeepSeekClient", FakeDeepSeek)
    monkeypatch.setattr(articles, "GENERATED_DIR", tmp_path)

    app = FastAPI()
    app.include_router(articles.router)
    app.state.settings = Settings(
        deepseek_api_key="test-key",
        data_dir=tmp_path / "agent-data",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/articles/auto-generate",
            json={"website_url": "https://example.com", "language": "en"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["research_report"]["profile"]["niche"] == "warehouse software"
    assert payload["page_type"] == "market_analysis"
    assert payload["topic"] == "How automation improves warehouse inventory accuracy"
    assert payload["revision_count"] == 2
    assert payload["quality_report"]["passed"] is False
    stored = json.loads((tmp_path / f"{payload['id']}.json").read_text(encoding="utf-8"))
    assert stored["research_report"]["references"][0]["word_count"] == 1200
    assert "<script" not in stored["html"]
    assert "onclick" not in stored["html"]
    assert "javascript:" not in stored["html"]
    assert (tmp_path / f"{payload['id']}.html").exists()


@pytest.mark.asyncio
async def test_research_requires_confirmation_before_article_generation(tmp_path, monkeypatch):
    report = {
        "profile": {
            "website_url": "https://example.com",
            "site_name": "Example",
            "business_summary": "Example provides warehouse software.",
            "niche": "warehouse software",
            "keywords": ["warehouse inventory software"],
            "primary_language": "en",
            "detected_languages": ["en", "ja"],
        },
        "references": [{
            "url": "https://reference.example/guide",
            "title": "Inventory Guide",
            "headings": [{"level": "h1", "text": "Inventory Guide"}],
            "word_count": 1800,
            "image_count": 4,
        }],
        "user_intent_report": {
            "keywords_intents": [{
                "keyword": "warehouse inventory software",
                "user_queries": [{
                    "query": "how to improve warehouse inventory accuracy",
                    "source": "AI-simulated",
                    "validation_status": "search-validated",
                }],
            }],
        },
        "editorial_decision": {
            "topic": "Warehouse Inventory Accuracy Guide",
            "page_type": "guide",
        },
        "writing_brief": {
            "topic": "Warehouse Inventory Accuracy Guide",
            "page_type": "guide",
            "headline_options": ["How to Improve Warehouse Inventory Accuracy"],
            "target_keywords": ["warehouse inventory software"],
            "target_queries": ["how to improve warehouse inventory accuracy"],
            "recommended_word_count": 1900,
            "recommended_outline": ["Accuracy problems", "Improvement process", "FAQ"],
        },
        "warnings": [],
    }

    class FakeResult:
        def to_dict(self):
            return report

    class FakeWorkflow:
        def __init__(self, settings):
            pass

        async def run(self, *args, **kwargs):
            return FakeResult()

        async def close(self):
            return None

        @staticmethod
        def build_generation_context(result):
            return "VERIFIED RESEARCH CONTEXT"

    class FakeDeepSeek:
        def __init__(self, **kwargs):
            pass

        async def generate_text(self, prompt, **kwargs):
            assert "VERIFIED RESEARCH CONTEXT" in prompt
            assert "how to improve warehouse inventory accuracy" in prompt
            assert "Improvement process" in prompt
            body = " ".join(["warehouse inventory accuracy guidance"] * 500)
            return (
                "<html><head><title>Confirmed Warehouse Guide</title></head>"
                "<body><article><h1>Confirmed Warehouse Guide</h1>"
                f"<p>Useful original guidance. {body}</p>"
                "<h2>Accuracy problems</h2><p>Measure inventory discrepancies.</p>"
                "<h2>Improvement process</h2><p>Apply a repeatable audit process.</p>"
                "<h2>FAQ</h2><p>Review common implementation questions.</p>"
                "</article></body></html>"
            )

        async def close(self):
            return None

    import src.ai.automatic_article as workflow_module
    import src.ai.deepseek_client as deepseek_module

    generated_dir = tmp_path / "generated"
    research_dir = tmp_path / "research-plans"
    generated_dir.mkdir()
    research_dir.mkdir()
    monkeypatch.setattr(workflow_module, "AutomaticArticleWorkflow", FakeWorkflow)
    monkeypatch.setattr(deepseek_module, "DeepSeekClient", FakeDeepSeek)
    async def fake_translate(settings, html, source_lang, target_lang, lang_names, label=""):
        assert source_lang == "en"
        assert target_lang == "ja"
        return html.replace("Confirmed Warehouse Guide", "倉庫在庫ガイド").replace(
            "Useful original guidance.",
            "実用的なガイドです。",
        )

    monkeypatch.setattr(articles, "_do_translate", fake_translate)
    monkeypatch.setattr(articles, "GENERATED_DIR", generated_dir)
    monkeypatch.setattr(articles, "RESEARCH_DIR", research_dir)

    app = FastAPI()
    app.include_router(articles.router)
    app.state.settings = Settings(
        deepseek_api_key="test-key",
        data_dir=tmp_path / "agent-data",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        research_response = await client.post(
            "/api/articles/research",
            json={"website_url": "https://example.com", "language": "en"},
        )
        assert research_response.status_code == 200
        research_payload = research_response.json()
        assert research_payload["status"] == "awaiting_confirmation"
        assert research_payload["agent_stage"] == "awaiting_confirmation"
        assert len(research_payload["agent_run_id"]) == 32
        assert research_payload["writing_brief"]["recommended_word_count"] == 1900
        assert not list(generated_dir.glob("*.html"))

        generation_response = await client.post(
            "/api/articles/generate-from-research",
            json={
                "research_id": research_payload["research_id"],
                "headline": "How to Improve Warehouse Inventory Accuracy",
                "word_count": 2100,
                "page_type": "guide",
                "outline": ["Accuracy problems", "Improvement process", "FAQ"],
            },
        )
        generated_id = generation_response.json()["id"]
        agent_response = await client.get(
            f"/api/articles/agent-runs/{research_payload['agent_run_id']}"
        )
        reloaded_response = await client.get(f"/api/articles/{generated_id}")
        duplicate_response = await client.post(
            "/api/articles/generate-from-research",
            json={"research_id": research_payload["research_id"]},
        )

    assert generation_response.status_code == 200
    generated = generation_response.json()
    assert generated["title"] == "Confirmed Warehouse Guide"
    assert generated["confirmed_brief"]["confirmed_word_count"] == 2100
    assert generated["language"] == "en"
    assert generated["agent_run_id"] == research_payload["agent_run_id"]
    assert generated["agent_stage"] == "article_ready"
    assert agent_response.status_code == 200
    assert agent_response.json()["article_id"] == generated["id"]
    assert agent_response.json()["stage"] == "article_ready"
    assert len(generated["translations"]) == 1
    assert generated["translations"][0]["language"] == "ja"
    assert 'lang="ja"' in generated["translations"][0]["html"]
    assert (generated_dir / f"{generated['id']}.html").is_file()
    assert (generated_dir / f"{generated['translations'][0]['id']}.html").is_file()
    assert reloaded_response.status_code == 200
    assert reloaded_response.json()["translations"][0]["language"] == "ja"
    assert duplicate_response.status_code == 409
    saved_plan = json.loads(
        (research_dir / f"{research_payload['research_id']}.json").read_text(encoding="utf-8")
    )
    assert saved_plan["status"] == "generated"
    assert len(saved_plan["generated_article_ids"]) == 2
