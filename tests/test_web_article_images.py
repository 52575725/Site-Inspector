from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bs4 import BeautifulSoup
from fastapi import FastAPI

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
                page_url="https://commons.wikimedia.org/example",
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            for index in range(count)
        ]

    def fake_download(url, dest_dir, filename):
        path = Path(dest_dir) / filename
        path.write_bytes(b"downloaded")
        return str(path)

    def fake_convert(path, quality):
        source = Path(path)
        target = source.with_suffix(".webp")
        target.write_bytes(b"webp")
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
