from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bs4 import BeautifulSoup
from fastapi import FastAPI
from PIL import Image

from src.agents.article_orchestrator import ArticleOrchestratorAgent
from src.fixers.article_image_fixer import ArticleImageFixer
from src.integrations.image_search import ImageResult
from src.web.routes import article_images


def test_generated_article_image_target_supports_research_recommendations():
    request = article_images.DraftImageSearchRequest(target_count=9)
    proposal = article_images.ImageProposalRequest(
        search_id="a" * 32,
        candidate_ids=[f"image-{index}" for index in range(1, 10)],
    )

    assert request.target_count == 9
    assert len(proposal.candidate_ids) == 9


def _article_html() -> str:
    words = " ".join(["silver"] * 260)
    return f"""<html><head><title>Silver Care Guide</title></head><body>
    <article><h1>Silver Care Guide</h1><p>Keep silver clean and bright.</p>
    <h2>Cleaning</h2><p>{words}</p>
    <h2>Storage</h2><p>Store jewelry in a dry lined box.</p>
    <h2>Prevention</h2><p>Avoid moisture and harsh chemicals.</p>
    </article></body></html>"""


def _settings(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        unsplash_api_key="",
        pexels_api_key="",
        pixabay_api_key="",
        image_generation_enabled=False,
        openai_api_key="",
        image_generation_model="gpt-image-2",
        deepseek_enabled=False,
        deepseek_api_key="",
        deepseek_model="deepseek-chat",
        deepseek_timeout=120,
    )


def _app(tmp_path, source_root, monkeypatch):
    app = FastAPI()
    app.include_router(article_images.router)
    app.state.settings = _settings(tmp_path)
    monkeypatch.setattr(article_images, "_source_root", lambda settings: source_root)
    return app


def test_section_queries_inherit_article_visual_context():
    queries = article_images._contextualize_queries(
        ["LBMA Good Delivery Standards", "Verification process"],
        "Complete Guide to LBMA Silver Bar Quality",
    )

    assert queries[0].endswith("silver bullion bars")
    assert queries[1].endswith("silver bullion bars")


def test_non_visual_editorial_queries_become_photographable_product_queries():
    queries = article_images._contextualize_queries(
        ["Hong Kong precious metals trading office", "LBMA Good Delivery List website"],
        "LBMA Compliance Guide for Silver Bars Sourced from Hong Kong",
    )

    assert queries == ["Hong Kong silver bullion ingot", "silver bullion ingot"]


def test_generated_draft_without_article_wrapper_is_still_searchable():
    summary = article_images._article_summary_content(
        "<html><head><title>Silver storage</title></head>"
        "<body><h1>Silver storage</h1><p>Secure vault storage for bullion.</p></body></html>",
        "drafts/example/index.html",
    )

    assert summary is not None
    assert summary["title"] == "Silver storage"


def test_article_image_search_uses_more_queries_and_results():
    assert article_images.SEMANTIC_QUERY_COUNT == 6
    assert article_images.RESULTS_PER_QUERY == 6
    assert article_images.BROAD_QUERY_RESULTS == 18
    assert article_images.MAX_CANDIDATES == 24


def test_broad_product_query_fetches_deeper_candidate_pool():
    assert article_images._image_result_limit("silver bullion ingot") == 18
    assert article_images._image_result_limit("silver bar customs officer inspection") == 6


@pytest.mark.asyncio
async def test_article_image_search_caps_expanded_candidate_pool(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    article_path = source_root / "blog" / "care" / "index.html"
    article_path.parent.mkdir(parents=True)
    article_path.write_text(_article_html(), encoding="utf-8")

    async def fake_slots(*args, **kwargs):
        return [
            {
                "slot_id": f"slot-{index}",
                "kind": "section",
                "image_type": "photo",
                "heading": "Cleaning",
                "section_excerpt": "Silver care section",
                "search_query": f"distinct photo scene {index}",
                "visual_brief": f"Distinct silver care scene {index}",
                "insertion_reason": "Illustrate the article section.",
                "chart_spec": {},
            }
            for index in range(6)
        ]

    def fake_search(query, count, *keys):
        query_id = str(abs(hash(query)))
        return [
            ImageResult(
                url=f"https://images.example/{query_id}-{index}.jpg",
                thumb_url=f"https://images.example/{query_id}-{index}-thumb.jpg",
                alt_text=query,
                photographer="Jane Doe",
                source="pexels",
                page_url=f"https://commons.wikimedia.org/{query_id}/{index}",
                license_name="CC BY 4.0",
            )
            for index in range(count)
        ]

    monkeypatch.setattr(article_images, "_semantic_image_slots", fake_slots)
    monkeypatch.setattr(article_images, "search_images", fake_search)
    app = _app(tmp_path, source_root, monkeypatch)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/article-images/search",
            json={"article_path": "blog/care/index.html", "target_count": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["queries"]) == 6
    assert len(payload["candidates"]) == article_images.MAX_CANDIDATES


@pytest.mark.asyncio
async def test_semantic_queries_use_full_article_context(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.deepseek_enabled = True
    settings.deepseek_api_key = "configured"
    captured = {}

    class FakeDeepSeekClient:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

        async def generate_json(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return {
                "slots": [
                    {
                        "heading": "Cleaning",
                        "image_type": "photo",
                        "query": "hands polishing tarnished silver jewelry",
                        "visual_brief": "Hands using a soft cloth on visibly tarnished silver.",
                        "insertion_reason": "Shows the cleaning action described in this section.",
                    },
                    {
                        "heading": "Storage",
                        "image_type": "photo",
                        "query": "silver jewelry lined storage box",
                        "visual_brief": "Silver jewelry separated inside a dry lined box.",
                        "insertion_reason": "Demonstrates the recommended storage environment.",
                    },
                ]
            }

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(article_images, "DeepSeekClient", FakeDeepSeekClient)
    slots = await article_images._semantic_image_slots(
        settings,
        _article_html(),
        {"title": "Silver Care Guide", "sections": ["Cleaning", "Storage"]},
        3,
    )

    assert [slot["heading"] for slot in slots] == ["Cleaning", "Storage"]
    assert slots[0]["search_query"] == "hands polishing tarnished silver jewelry"
    assert slots[1]["section_index"] == 1
    assert "cleaning action" in slots[0]["insertion_reason"]
    assert "Keep silver clean and bright" in captured["prompt"]
    assert "Storage" in captured["prompt"]
    assert captured["closed"] is True


def test_grounded_trend_chart_requires_numeric_article_table(tmp_path):
    content = """<html><body><article><h1>Silver demand</h1>
    <h2>Solar demand trend</h2><p>Demand rose across the period.</p>
    <p>Source: <a href="https://authority.example/report">Industry report</a></p>
    <table><tr><th>Year</th><th>Demand (Moz)</th></tr>
    <tr><td>2023</td><td>140</td></tr><tr><td>2024</td><td>166</td></tr>
    <tr><td>2025</td><td>195</td></tr></table>
    </article></body></html>"""

    slots = article_images._extract_grounded_chart_slots(content)

    assert len(slots) == 1
    assert slots[0]["heading"] == "Solar demand trend"
    assert slots[0]["chart_spec"]["values"] == [140.0, 166.0, 195.0]
    assert slots[0]["chart_spec"]["source_url"] == "https://authority.example/report"
    chart_path = article_images._render_trend_chart(
        slots[0]["chart_spec"],
        tmp_path / "trend.webp",
    )
    with Image.open(chart_path) as chart:
        assert chart.size == (1200, 800)

    assert article_images._extract_grounded_chart_slots(
        "<article><h2>Market trend</h2><p>No numeric table.</p></article>"
    ) == []


def test_image_with_planned_heading_is_inserted_in_that_exact_section():
    soup = BeautifulSoup(_article_html(), "html.parser")
    image = {
        "local_path": "images/storage.webp",
        "query": "dry lined jewelry storage box",
        "alt_text": "Silver jewelry in a dry lined box",
        "caption": "Correct dry storage for silver jewelry",
        "width": 1200,
        "height": 800,
        "target_heading": "Storage",
        "target_section_index": 1,
        "insertion_reason": "Show the storage setup described in this section.",
    }

    inserted = ArticleImageFixer(max_images=3)._insert_images(soup, [image], include_hero=False)

    figure = soup.select_one("figure.article-media")
    assert inserted == 1
    assert figure is not None
    assert figure.find_previous("h2").get_text(" ", strip=True) == "Storage"
    assert figure["data-target-heading"] == "Storage"


@pytest.mark.asyncio
async def test_grounded_chart_candidate_can_be_selected_and_inserted(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = """<html><head><title>Silver outlook</title></head><body><article>
    <h1>Silver outlook</h1><p>Verified demand figures.</p>
    <h2>Demand trend</h2><p>Annual demand evidence.</p>
    <table><tr><th>Year</th><th>Demand</th></tr><tr><td>2023</td><td>100</td></tr>
    <tr><td>2024</td><td>125</td></tr><tr><td>2025</td><td>150</td></tr></table>
    </article></body></html>"""
    slot = article_images._extract_grounded_chart_slots(content)[0]
    result = ImageResult(
        url="",
        thumb_url=article_images._chart_thumbnail(slot["chart_spec"]),
        alt_text="Verified annual silver demand trend",
        photographer="",
        source="grounded-chart",
        width=1200,
        height=800,
        license_name="Data from article",
    )
    app = _app(tmp_path, source_root, monkeypatch)
    search_id = "c" * 32
    app.state.article_image_searches = {
        search_id: {
            "created_at": article_images.time.monotonic(),
            "article_path": "preview/index.html",
            "article_id": "",
            "agent_run_id": "",
            "content": content,
            "summary": {"title": "Silver outlook", "sections": ["Demand trend"], "image_count": 0},
            "target_count": 3,
            "needed": 1,
            "queries": ["Demand trend chart"],
            "semantic_slots": [],
            "candidates": {"chart-1": ("Demand trend chart", result, slot)},
        }
    }
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/article-images/proposals",
            json={"search_id": search_id, "candidate_ids": ["chart-1"]},
        )

    assert response.status_code == 200
    proposal = response.json()
    assert proposal["images"][0]["image_type"] == "chart"
    output = BeautifulSoup(Path(proposal["output_path"]).read_text(encoding="utf-8"), "html.parser")
    chart = output.select_one('figure.article-media[data-image-type="chart"]')
    assert chart is not None
    assert chart.find_previous("h2").get_text(" ", strip=True) == "Demand trend"


@pytest.mark.asyncio
async def test_article_image_workspace_creates_review_copy_without_touching_source(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "source"
    article_path = source_root / "blog" / "care" / "index.html"
    article_path.parent.mkdir(parents=True)
    original = _article_html()
    article_path.write_text(original, encoding="utf-8")

    def fake_search(query, count, *keys):
        query_id = str(abs(hash(query)))
        return [
            ImageResult(
                url=f"https://images.example/{query_id}-{index}.jpg",
                thumb_url=f"https://images.example/{query_id}-{index}-thumb.jpg",
                alt_text=f"{query} photo {index}",
                photographer="Jane Doe",
                source="wikimedia",
                width=1200,
                height=800,
                page_url=f"https://commons.wikimedia.org/example/{query_id}/{index}",
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            for index in range(count)
        ]

    def fake_download(url, dest_dir, filename):
        path = Path(dest_dir) / filename
        path.write_bytes(url.encode())
        return str(path)

    def fake_convert(path, quality):
        source = Path(path)
        target = source.with_suffix(".webp")
        target.write_bytes(b"webp:" + source.read_bytes())
        return str(target)

    async def allow_public_url(url):
        return url

    monkeypatch.setattr(article_images, "search_images", fake_search)
    monkeypatch.setattr(article_images, "download_image", fake_download)
    monkeypatch.setattr(article_images, "convert_to_webp", fake_convert)
    monkeypatch.setattr(article_images, "validate_public_http_url", allow_public_url)
    app = _app(tmp_path, source_root, monkeypatch)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        articles_response = await client.get("/api/article-images/articles")
        assert articles_response.status_code == 200
        assert articles_response.json()["articles"][0]["path"] == "blog/care/index.html"

        search_response = await client.post(
            "/api/article-images/search",
            json={"article_path": "blog/care/index.html", "target_count": 3},
        )
        assert search_response.status_code == 200
        search = search_response.json()
        assert search["needed"] == 3
        assert len(search["candidates"]) >= 3

        proposal_response = await client.post(
            "/api/article-images/proposals",
            json={
                "search_id": search["search_id"],
                "candidate_ids": [item["id"] for item in search["candidates"][:3]],
                "allow_ai_fallback": False,
            },
        )
        assert proposal_response.status_code == 200
        proposal = proposal_response.json()
        assert proposal["status"] == "review_required"
        assert proposal["inserted"] == 3

        preview_response = await client.get(proposal["preview_url"])
        assert preview_response.status_code == 200
        assert "default-src 'none'" in preview_response.headers["content-security-policy"]
        assert f"/api/article-images/proposals/{proposal['proposal_id']}/assets/" in preview_response.text

        first_image = Path(proposal["images"][0]["local_path"]).name
        asset_response = await client.get(
            f"/api/article-images/proposals/{proposal['proposal_id']}/assets/{first_image}"
        )
        assert asset_response.status_code == 200

    assert article_path.read_text(encoding="utf-8") == original
    output = Path(proposal["output_path"])
    assert output.is_file()
    soup = BeautifulSoup(output.read_text(encoding="utf-8"), "html.parser")
    assert len(soup.select("article figure.article-media")) == 3


@pytest.mark.asyncio
async def test_article_image_workspace_rejects_path_traversal(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    app = _app(tmp_path, source_root, monkeypatch)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/article-images/search",
            json={"article_path": "../outside.html", "target_count": 3},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_article_image_workspace_rejects_unknown_candidate(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    article_path = source_root / "blog" / "care" / "index.html"
    article_path.parent.mkdir(parents=True)
    article_path.write_text(_article_html(), encoding="utf-8")

    monkeypatch.setattr(
        article_images,
        "search_images",
        lambda *args: [
            ImageResult(
                url="https://images.example/one.jpg",
                thumb_url="https://images.example/one-thumb.jpg",
                alt_text="Silver ring",
                photographer="Jane Doe",
                source="wikimedia",
                license_name="CC BY 4.0",
            )
        ],
    )
    app = _app(tmp_path, source_root, monkeypatch)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        search = (
            await client.post(
                "/api/article-images/search",
                json={"article_path": "blog/care/index.html", "target_count": 3},
            )
        ).json()
        response = await client.post(
            "/api/article-images/proposals",
            json={
                "search_id": search["search_id"],
                "candidate_ids": ["image-forged"],
                "allow_ai_fallback": False,
            },
        )

    assert response.status_code == 400
    assert "Unknown candidate" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_excludes_sources_used_by_earlier_proposals(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    article_path = source_root / "blog" / "care" / "index.html"
    article_path.parent.mkdir(parents=True)
    article_path.write_text(_article_html(), encoding="utf-8")
    settings = _settings(tmp_path)
    manifest_dir = settings.data_dir / "fixed" / "article-images" / ("b" * 32)
    manifest_dir.mkdir(parents=True)
    used_url = "https://images.example/already-used.jpg"
    (manifest_dir / "manifest.json").write_text(
        '{"search_candidates":[{"url":"https://images.example/already-used.jpg"}]}',
        encoding="utf-8",
    )

    def fake_search(query, count, *keys):
        return [
            ImageResult(
                url=used_url if index == 0 else f"https://images.example/{abs(hash(query))}-{index}.jpg",
                thumb_url=used_url if index == 0 else f"https://images.example/{abs(hash(query))}-{index}.jpg",
                alt_text=f"{query} {index}",
                photographer="Jane Doe",
                source="wikimedia",
                page_url=f"https://commons.wikimedia.org/{abs(hash(query))}/{index}",
                license_name="CC BY 4.0",
            )
            for index in range(count)
        ]

    monkeypatch.setattr(article_images, "search_images", fake_search)
    app = _app(tmp_path, source_root, monkeypatch)
    app.state.settings = settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/article-images/search",
            json={"article_path": "blog/care/index.html", "target_count": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["history_duplicates_excluded"] > 0
    assert all(candidate["thumbnail_url"] != used_url for candidate in payload["candidates"])


@pytest.mark.asyncio
async def test_search_reuses_historical_candidates_when_no_new_images_exist(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "source"
    article_path = source_root / "blog" / "care" / "index.html"
    article_path.parent.mkdir(parents=True)
    article_path.write_text(_article_html(), encoding="utf-8")
    settings = _settings(tmp_path)
    manifest_dir = settings.data_dir / "fixed" / "article-images" / ("c" * 32)
    manifest_dir.mkdir(parents=True)

    used_urls = [
        f"https://images.example/used-{index}.jpg"
        for index in range(article_images.BROAD_QUERY_RESULTS)
    ]
    (manifest_dir / "manifest.json").write_text(
        json.dumps({
            "images": [
                {"source_url": url, "local_path": f"/images/{index}.webp"}
                for index, url in enumerate(used_urls)
            ],
        }),
        encoding="utf-8",
    )

    def fake_search(query, count, *keys):
        return [
            ImageResult(
                url=used_urls[index],
                thumb_url=used_urls[index],
                alt_text=f"Previously used {index}",
                photographer="Jane Doe",
                source="wikimedia",
                page_url=f"https://commons.wikimedia.org/used/{index}",
                license_name="CC BY 4.0",
            )
            for index in range(count)
        ]

    monkeypatch.setattr(article_images, "search_images", fake_search)
    app = _app(tmp_path, source_root, monkeypatch)
    app.state.settings = settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/article-images/search",
            json={"article_path": "blog/care/index.html", "target_count": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["candidates"]) >= 3
    assert payload["history_duplicates_reused"] >= 3
    assert all(candidate["previously_used"] for candidate in payload["candidates"])


def test_image_history_tracks_sources_and_visual_duplicates(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    source_root = tmp_path / "source"
    source_image = source_root / "images" / "published.webp"
    source_image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "navy").save(source_image, "WEBP")
    monkeypatch.setattr(article_images, "_source_root", lambda current_settings: source_root)
    proposal = settings.data_dir / "fixed" / "article-images" / ("a" * 32)
    image_path = proposal / "images" / "used.webp"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "silver").save(image_path, "WEBP")
    (proposal / "manifest.json").write_text(
        '{"images":[{"local_path":"/images/used.webp",'
        '"source_url":"https://images.example/used.jpg?size=large",'
        '"page_url":"https://images.example/photo/1"}]}',
        encoding="utf-8",
    )

    history = article_images._image_history(settings)
    content_hash, perceptual_hash = article_images._image_fingerprints(image_path)
    published_content_hash, _ = article_images._image_fingerprints(source_image)

    assert "https://images.example/used.jpg" in history["source_keys"]
    assert content_hash in history["content_hashes"]
    assert published_content_hash in history["content_hashes"]
    assert perceptual_hash in history["perceptual_hashes"]
    assert article_images._is_duplicate_image(
        content_hash,
        perceptual_hash,
        history["content_hashes"],
        history["perceptual_hashes"],
    )


def test_wikimedia_file_ids_remain_distinct_during_url_normalization():
    first = article_images._canonical_url(
        "https://commons.wikimedia.org/?curid=100&utm_source=test"
    )
    second = article_images._canonical_url(
        "https://commons.wikimedia.org/?curid=200"
    )

    assert first == "https://commons.wikimedia.org?curid=100"
    assert second == "https://commons.wikimedia.org?curid=200"
    assert first != second


def test_generation_prompt_uses_article_context_and_avoids_repetition():
    prompt = ArticleImageFixer._build_generation_prompt(
        "air freight security for silver bullion",
        article_title="Silver Shipping: Air vs Sea",
        section_headings=["Air Freight", "Sea Freight", "Customs"],
        avoid_concepts=["container ship at port"],
        variation_index=2,
    )

    assert "Silver Shipping: Air vs Sea" in prompt
    assert "air freight security for silver bullion" in prompt
    assert "container ship at port" in prompt
    assert "location-rich wide scene" in prompt


@pytest.mark.asyncio
async def test_generated_article_can_search_select_insert_and_display_images(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    generated = tmp_path / "generated"
    generated.mkdir()
    article_id = "draft-article"
    translation_id = "draft-article-ja"
    original = _article_html()
    translated = original.replace("Silver Care Guide", "銀製品のお手入れガイド").replace(
        "Keep silver clean and bright",
        "銀製品を美しく保つ方法",
    )
    orchestrator = ArticleOrchestratorAgent(tmp_path / "data" / "article-agent-runs")
    agent_state = orchestrator.start("https://example.com", {"language": "en"})
    agent_state = orchestrator.complete_research(agent_state, "a" * 32, {
        "profile": {"site_name": "Example", "primary_language": "en"},
        "editorial_decision": {"topic": "Silver care"},
    })
    agent_state = orchestrator.begin_writing(agent_state)
    agent_state = orchestrator.complete_writing(agent_state, {
        "id": article_id,
        "title": "Silver Care Guide",
        "language": "en",
        "page_type": "guide",
        "content_direction": "evergreen_guide",
        "html": original,
    })
    (generated / f"{article_id}.json").write_text(
        json.dumps({
            "id": article_id,
            "title": "Silver Care Guide",
            "topic": "Silver care",
            "language": "en",
            "html": original,
            "created_at": "2026-07-28T00:00:00+00:00",
            "agent_run_id": agent_state.run_id,
            "translation_group_id": article_id,
            "translations": [
                {"id": translation_id, "language": "ja", "title": "銀製品のお手入れガイド"}
            ],
        }),
        encoding="utf-8",
    )
    (generated / f"{article_id}.html").write_text(original, encoding="utf-8")
    (generated / f"{translation_id}.json").write_text(
        json.dumps({
            "id": translation_id,
            "title": "銀製品のお手入れガイド",
            "topic": "Silver care",
            "language": "ja",
            "source_language": "en",
            "source_article_id": article_id,
            "translation_group_id": article_id,
            "source_html": original,
            "html": translated,
            "created_at": "2026-07-28T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    (generated / f"{translation_id}.html").write_text(translated, encoding="utf-8")

    search_version = {"value": 1}

    def fake_search(query, count, *keys):
        return [
            ImageResult(
                url=(
                    f"https://images.example/v{search_version['value']}-"
                    f"{abs(hash(query))}-{index}.jpg"
                ),
                thumb_url=(
                    f"https://images.example/v{search_version['value']}-"
                    f"{abs(hash(query))}-{index}-thumb.jpg"
                ),
                alt_text=f"{query} image {index}",
                photographer="Jane Doe",
                source="wikimedia",
                width=1200,
                height=800,
                page_url=f"https://commons.wikimedia.org/{abs(hash(query))}/{index}",
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            for index in range(count)
        ]

    def fake_download(url, dest_dir, filename):
        path = Path(dest_dir) / filename
        path.write_bytes(url.encode())
        return str(path)

    def fake_convert(path, quality):
        source = Path(path)
        target = source.with_suffix(".webp")
        target.write_bytes(b"webp:" + source.read_bytes())
        return str(target)

    async def allow_public_url(url):
        return url

    monkeypatch.setattr(article_images, "GENERATED_DIR", generated)
    monkeypatch.setattr(article_images, "search_images", fake_search)
    monkeypatch.setattr(article_images, "download_image", fake_download)
    monkeypatch.setattr(article_images, "convert_to_webp", fake_convert)
    monkeypatch.setattr(article_images, "validate_public_http_url", allow_public_url)
    app = _app(tmp_path, source_root, monkeypatch)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        search_response = await client.post(
            f"/api/articles/{article_id}/images/search",
            json={"target_count": 3},
        )
        assert search_response.status_code == 200
        search = search_response.json()
        assert search["needed"] == 3
        assert search["agent_run_id"] == agent_state.run_id
        assert search["agent_stage"] == "awaiting_image_selection"
        assert search["image_plan"]["placement_slots"]

        proposal_response = await client.post(
            "/api/article-images/proposals",
            json={
                "search_id": search["search_id"],
                "candidate_ids": [item["id"] for item in search["candidates"][:3]],
                "allow_ai_fallback": False,
            },
        )
        assert proposal_response.status_code == 200
        proposal = proposal_response.json()
        assert proposal["article_id"] == article_id
        assert proposal["agent_stage"] == "image_review"

        apply_response = await client.post(
            f"/api/articles/{article_id}/images/apply",
            json={"proposal_id": proposal["proposal_id"]},
        )
        assert apply_response.status_code == 200
        applied = apply_response.json()
        assert applied["image_count"] == 3
        assert applied["synchronized_language_count"] == 2
        assert {version["language"] for version in applied["versions"]} == {"en", "ja"}
        assert f"/api/articles/{article_id}/assets/" in applied["html"]
        failed_checks = [
            check for check in applied["quality_report"]["checks"]
            if not check["passed"] and check["severity"] == "error"
        ]
        assert applied["agent_stage"] == "ready_to_publish", failed_checks
        assert applied["quality_report"]["passed"] is True

        first_asset = Path(applied["images"][0]["local_path"]).name
        asset_response = await client.get(
            f"/api/articles/{article_id}/assets/{first_asset}"
        )
        assert asset_response.status_code == 200
        translated_asset_response = await client.get(
            f"/api/articles/{translation_id}/assets/{first_asset}"
        )
        assert translated_asset_response.status_code == 200

        # Simulate a legacy group where the primary language lost its image metadata/markup.
        legacy_primary = json.loads(
            (generated / f"{article_id}.json").read_text(encoding="utf-8")
        )
        legacy_soup = BeautifulSoup(legacy_primary["html"], "html.parser")
        for figure in legacy_soup.select("figure.article-media"):
            figure.decompose()
        style = legacy_soup.find("style", id="site-inspector-article-images")
        if style:
            style.decompose()
        legacy_primary["html"] = str(legacy_soup)
        legacy_primary["images"] = []
        legacy_primary["image_count"] = 0
        (generated / f"{article_id}.json").write_text(
            json.dumps(legacy_primary),
            encoding="utf-8",
        )
        (generated / f"{article_id}.html").write_text(
            legacy_primary["html"],
            encoding="utf-8",
        )

        search_version["value"] = 2
        append_search_response = await client.post(
            f"/api/articles/{translation_id}/images/search",
            json={"target_count": 4},
        )
        assert append_search_response.status_code == 200
        append_search = append_search_response.json()
        assert append_search["needed"] == 1

        append_proposal_response = await client.post(
            "/api/article-images/proposals",
            json={
                "search_id": append_search["search_id"],
                "candidate_ids": [append_search["candidates"][0]["id"]],
                "allow_ai_fallback": False,
            },
        )
        assert append_proposal_response.status_code == 200
        append_proposal = append_proposal_response.json()

        append_apply_response = await client.post(
            f"/api/articles/{translation_id}/images/apply",
            json={"proposal_id": append_proposal["proposal_id"]},
        )
        assert append_apply_response.status_code == 200
        appended = append_apply_response.json()
        assert appended["image_count"] == 4
        assert len(appended["images"]) == 4

    stored = json.loads((generated / f"{article_id}.json").read_text(encoding="utf-8"))
    stored_translation = json.loads(
        (generated / f"{translation_id}.json").read_text(encoding="utf-8")
    )
    assert stored["image_count"] == 4
    assert stored_translation["image_count"] == 4
    assert stored["agent_stage"] == "ready_to_publish"
    assert stored["quality_report"]["passed"] is True
    assert len(stored["images"]) == 4
    assert len(stored_translation["images"]) == 4
    assert "images/" in stored_translation["html"]
    assert "images/" in stored_translation["source_html"]
    assert "/api/articles/" not in stored["html"]
    assert "images/" in stored["html"]
    soup = BeautifulSoup(stored["html"], "html.parser")
    assert len(soup.select("article figure.article-media")) == 4
    assert "Keep silver clean and bright" in soup.get_text(" ", strip=True)
