from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bs4 import BeautifulSoup
from fastapi import FastAPI
from PIL import Image

from src.fixers.article_image_fixer import ArticleImageFixer
from src.integrations.image_search import ImageResult
from src.web.routes import article_images


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

    assert queries[0].endswith("silver")
    assert queries[1].endswith("silver")


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
                "queries": [
                    "secured silver bullion air cargo",
                    "container ship loading precious metals",
                    "customs officer inspecting sealed cargo",
                ]
            }

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(article_images, "DeepSeekClient", FakeDeepSeekClient)
    queries = await article_images._semantic_image_queries(
        settings,
        _article_html(),
        {"title": "Silver Care Guide", "sections": ["Cleaning", "Storage"]},
        3,
    )

    assert queries == [
        "secured silver bullion air cargo",
        "container ship loading precious metals",
        "customs officer inspecting sealed cargo",
    ]
    assert "Keep silver clean and bright" in captured["prompt"]
    assert "Storage" in captured["prompt"]
    assert captured["closed"] is True


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

    used_urls = [f"https://images.example/used-{index}.jpg" for index in range(4)]
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
    original = _article_html()
    (generated / f"{article_id}.json").write_text(
        json.dumps({
            "id": article_id,
            "title": "Silver Care Guide",
            "topic": "Silver care",
            "html": original,
            "created_at": "2026-07-28T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    (generated / f"{article_id}.html").write_text(original, encoding="utf-8")

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

        apply_response = await client.post(
            f"/api/articles/{article_id}/images/apply",
            json={"proposal_id": proposal["proposal_id"]},
        )
        assert apply_response.status_code == 200
        applied = apply_response.json()
        assert applied["image_count"] == 3
        assert f"/api/articles/{article_id}/assets/" in applied["html"]

        first_asset = Path(applied["images"][0]["local_path"]).name
        asset_response = await client.get(
            f"/api/articles/{article_id}/assets/{first_asset}"
        )
        assert asset_response.status_code == 200

        search_version["value"] = 2
        append_search_response = await client.post(
            f"/api/articles/{article_id}/images/search",
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
            f"/api/articles/{article_id}/images/apply",
            json={"proposal_id": append_proposal["proposal_id"]},
        )
        assert append_apply_response.status_code == 200
        appended = append_apply_response.json()
        assert appended["image_count"] == 4
        assert len(appended["images"]) == 4

    stored = json.loads((generated / f"{article_id}.json").read_text(encoding="utf-8"))
    assert stored["image_count"] == 4
    assert len(stored["images"]) == 4
    assert "/api/articles/" not in stored["html"]
    assert "images/" in stored["html"]
    soup = BeautifulSoup(stored["html"], "html.parser")
    assert len(soup.select("article figure.article-media")) == 4
    assert "Keep silver clean and bright" in soup.get_text(" ", strip=True)
