from __future__ import annotations

import json

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
    app.state.settings = Settings(deepseek_api_key="test-key")
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
            return (
                "<html><head><title>Confirmed Warehouse Guide</title></head>"
                "<body><article><h1>Confirmed Warehouse Guide</h1>"
                "<p>Useful original guidance.</p></article></body></html>"
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
    monkeypatch.setattr(articles, "GENERATED_DIR", generated_dir)
    monkeypatch.setattr(articles, "RESEARCH_DIR", research_dir)

    app = FastAPI()
    app.include_router(articles.router)
    app.state.settings = Settings(deepseek_api_key="test-key")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        research_response = await client.post(
            "/api/articles/research",
            json={"website_url": "https://example.com", "language": "en"},
        )
        assert research_response.status_code == 200
        research_payload = research_response.json()
        assert research_payload["status"] == "awaiting_confirmation"
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

    assert generation_response.status_code == 200
    generated = generation_response.json()
    assert generated["title"] == "Confirmed Warehouse Guide"
    assert generated["confirmed_brief"]["confirmed_word_count"] == 2100
    assert (generated_dir / f"{generated['id']}.html").is_file()
    saved_plan = json.loads(
        (research_dir / f"{research_payload['research_id']}.json").read_text(encoding="utf-8")
    )
    assert saved_plan["status"] == "generated"
