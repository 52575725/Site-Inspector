from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.article_orchestrator import ArticleOrchestratorAgent
from src.agents.citation_agent import ArticleCitationAgent
from src.agents.image_agent import ArticleImageAgent
from src.agents.models import ArticleAgentStage
from src.agents.quality_agent import ArticleQualityAgent
from src.agents.writing_agent import ArticleWritingAgent, WritingTask


def _article_html(*, with_images: bool = False) -> str:
    body = " ".join(["Evidence based article content"] * 70)
    def figure(index: int, heading: str) -> str:
        if not with_images:
            return ""
        return (
            f'<figure class="article-media" data-target-heading="{heading}" '
            f'data-image-type="photo" data-image-purpose="Clarify {heading}">'
            f'<img src="images/{index}.webp" alt="{heading} scene">'
            f'<figcaption>{heading} scene</figcaption></figure>'
        )
    return (
        "<html><body><article><h1>Silver logistics guide</h1>"
        f"<p>{body}</p>"
        f"<h2>Air freight</h2><p>Airport cargo handling and security.</p>{figure(1, 'Air freight')}"
        f"<h2>Sea freight</h2><p>Container routes and port inspections.</p>{figure(2, 'Sea freight')}"
        f"<h2>Customs</h2><p>Documents and border verification.</p>{figure(3, 'Customs')}"
        "</article></body></html>"
    )


def test_image_agent_combines_article_structure_and_research_recommendation():
    plan = ArticleImageAgent().plan(
        _article_html(),
        research_report={"writing_brief": {"image_count_min": 5, "image_count_max": 7}},
    )

    assert plan.target_count == 4
    assert plan.needed_count == 4
    assert plan.placement_slots[0].kind == "hero"
    assert {slot.heading for slot in plan.placement_slots[1:]} == {
        "Air freight", "Sea freight", "Customs"
    }


def test_orchestrator_persists_joint_article_image_quality_state(tmp_path):
    agent = ArticleOrchestratorAgent(tmp_path / "runs")
    state = agent.start("https://example.com", {"language": "en"})
    state = agent.complete_research(
        state,
        "a" * 32,
        {
            "profile": {"site_name": "Example", "primary_language": "en"},
            "editorial_decision": {"topic": "Silver logistics"},
        },
    )
    state = agent.begin_writing(state)
    state = agent.complete_writing(state, {
        "id": "article-1",
        "title": "Silver logistics guide",
        "language": "en",
        "page_type": "guide",
        "content_direction": "evergreen_guide",
        "html": _article_html(),
    })
    state = agent.plan_images(
        state,
        {"html": _article_html(), "research_report": {}},
        3,
    )
    state = agent.image_proposal_ready(state, "b" * 32)
    state = agent.complete_images(state, _article_html(with_images=True))

    reloaded = agent.load(state.run_id)
    assert reloaded.stage == ArticleAgentStage.READY_TO_PUBLISH
    assert reloaded.content_quality and reloaded.content_quality.passed
    assert reloaded.final_quality and reloaded.final_quality.passed
    assert [event.stage for event in reloaded.events] == [
        ArticleAgentStage.RESEARCHING,
        ArticleAgentStage.AWAITING_CONFIRMATION,
        ArticleAgentStage.WRITING,
        ArticleAgentStage.ARTICLE_READY,
        ArticleAgentStage.IMAGE_PLANNING,
        ArticleAgentStage.AWAITING_IMAGE_SELECTION,
        ArticleAgentStage.IMAGE_REVIEW,
        ArticleAgentStage.READY_TO_PUBLISH,
    ]


def test_joint_quality_agent_detects_duplicate_or_incomplete_images():
    plan = ArticleImageAgent().plan(_article_html(), requested_target=3)
    broken = _article_html().replace(
        "<h2>Air freight</h2>",
        '<figure class="article-media"><img src="same.webp" alt=""></figure>' * 3
        + "<h2>Air freight</h2>",
    )

    report = ArticleQualityAgent().inspect_article_with_images(broken, plan)

    assert report.passed is False
    failed = {check.name for check in report.checks if not check.passed}
    assert {"image_metadata", "unique_images"} <= failed


def test_content_quality_failure_blocks_image_planning(tmp_path):
    agent = ArticleOrchestratorAgent(tmp_path / "runs")
    state = agent.start("https://example.com", {})
    state = agent.complete_research(state, "a" * 32, {"profile": {}})
    state = agent.begin_writing(state)
    state = agent.complete_writing(state, {
        "id": "short-draft",
        "html": "<article><h1>Short</h1><p>Too short.</p></article>",
    })

    assert state.stage == ArticleAgentStage.CONTENT_REVIEW
    with pytest.raises(ValueError, match="quality gate"):
        agent.plan_images(state, {"html": _article_html()}, 3)


def test_citation_agent_requires_approved_inline_sources_for_current_claims():
    source = "https://customs.example/regime"
    report = {
        "profile": {"website_url": "https://example.com"},
        "authority_sources": [{"url": source, "title": "Customs authority"}],
    }
    uncited = (
        "<article><p>The dealer registration regime became effective in 2023 "
        "and requires exporter registration.</p></article>"
    )
    cited = (
        "<article><p>The dealer registration regime became effective in 2023 "
        f'and requires exporter registration <a href="{source}">according to Customs</a>.'
        "</p></article>"
    )

    failed = ArticleCitationAgent().inspect(uncited, report)
    passed = ArticleCitationAgent().inspect(cited, report)

    assert failed.passed is False
    assert failed.metrics["uncited_factual_block_count"] == 1
    assert passed.passed is True
    assert passed.metrics["citation_coverage"] == 1.0


def test_quality_agent_blocks_reference_copying_and_repetitive_paragraphs():
    copied = (
        "Silver testing starts with a documented sampling plan that identifies each lot "
        "before laboratory analysis and preserves a traceable chain of custody for buyers."
    )
    repeated = (
        "A receiving team should record the lot number, package condition, seal identifier, "
        "delivery time, responsible operator, and any visible discrepancy before acceptance."
    )
    filler = " ".join(f"Distinct operational detail {index}." for index in range(90))
    html = (
        "<article><h1>Silver verification</h1>"
        f"<p>{copied}</p><p>{repeated}</p><p>{repeated}</p><p>{filler}</p>"
        "<h2>Sampling</h2><p>Record the sample location and method for every lot.</p>"
        "<h2>Receiving</h2><p>Compare seals and documents before accepting delivery.</p>"
        "</article>"
    )
    report = ArticleQualityAgent().inspect_content(
        html,
        research_report={
            "references": [{
                "url": "https://reference.example/testing",
                "similarity_text": copied,
            }],
        },
    )

    failed = {check.name for check in report.checks if not check.passed}
    assert {"source_overlap", "paragraph_redundancy"} <= failed
    assert report.metrics["source_max_match_tokens"] >= 16
    assert report.metrics["duplicate_paragraph_pairs"] == 1


def test_quality_agent_allows_attributed_quote_and_flags_template_filler():
    quote = (
        "Silver testing starts with a documented sampling plan that identifies each lot "
        "before laboratory analysis and preserves a traceable chain of custody for buyers."
    )
    body = " ".join(f"Useful verification detail {index}." for index in range(90))
    html = (
        "<article><h1>Verification process</h1>"
        f"<blockquote>{quote}</blockquote><p>{body}</p>"
        "<p>In today's rapidly changing market, it is important to note that testing plays "
        "a crucial role for every buyer.</p>"
        "<h2>Records</h2><p>Keep a signed record for each inspection step and exception.</p>"
        "<h2>Review</h2><p>Resolve document mismatches before accepting the shipment.</p>"
        "</article>"
    )
    report = ArticleQualityAgent().inspect_content(
        html,
        research_report={
            "references": [{
                "url": "https://reference.example/testing",
                "similarity_text": quote,
            }],
        },
    )

    checks = {check.name: check for check in report.checks}
    assert checks["source_overlap"].passed is True
    assert checks["template_language"].passed is False


def test_citation_agent_catches_uncited_record_and_typical_range_claims():
    html = (
        "<article><p>Premiums reached an all-time high, while typical dealer charges "
        "remained within a narrow range for wholesale buyers.</p></article>"
    )
    report = ArticleCitationAgent().inspect(
        html,
        {
            "profile": {"website_url": "https://example.com"},
            "authority_sources": [{"url": "https://authority.example/data"}],
        },
    )

    assert report.passed is False
    assert report.metrics["uncited_factual_block_count"] == 1


def test_publish_agent_requires_joint_quality_and_records_completion(tmp_path):
    agent = ArticleOrchestratorAgent(tmp_path / "runs")
    state = agent.start("https://example.com", {})
    state = agent.complete_research(state, "a" * 32, {"profile": {}})
    state = agent.begin_writing(state)
    state = agent.complete_writing(state, {"id": "article", "html": _article_html()})

    with pytest.raises(ValueError, match="joint quality"):
        agent.begin_publishing(state, "https://github.com/acme/site")

    state = agent.plan_images(state, {"html": _article_html()}, 3)
    state = agent.image_proposal_ready(state, "b" * 32)
    state = agent.complete_images(state, _article_html(with_images=True))
    state = agent.begin_publishing(state, "https://github.com/acme/site")
    state = agent.complete_publishing(state, "https://github.com/acme/site/pull/1")

    assert state.stage == ArticleAgentStage.COMPLETED
    assert state.publication["pr_url"].endswith("/pull/1")


@pytest.mark.asyncio
async def test_writing_agent_uses_direction_specific_generation_settings():
    captured = {}

    class FakeClient:
        async def generate_text(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured.update(kwargs)
            return "<html><body><article><h1>News</h1></article></body></html>"

    settings = SimpleNamespace(
        deepseek_api_key="configured",
        deepseek_model="deepseek-chat",
        deepseek_timeout=120,
    )
    agent = ArticleWritingAgent(settings, ai_client=FakeClient())
    result = await agent.write(WritingTask(
        prompt="Confirmed writing brief",
        page_type="news",
        content_direction="news",
        language="en",
    ))

    assert result.startswith("<html>")
    assert captured["temperature"] == 0.35
    assert "cite only verified authority URLs" in captured["system"]

    revised = await agent.revise(
        WritingTask(
            prompt="Confirmed writing brief",
            page_type="news",
            content_direction="news",
            language="en",
        ),
        result,
        ["Article content is too short."],
    )
    assert revised.startswith("<html>")
    assert "Article content is too short" in captured["prompt"]
    assert captured["temperature"] == 0.25
