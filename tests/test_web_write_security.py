import json

import httpx
import pytest
from fastapi import FastAPI

from config.settings import Settings
from src.web.app import create_app
from src.web.routes import articles


@pytest.mark.asyncio
async def test_app_blocks_cross_origin_mutations_and_untrusted_hosts():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post(
            "/api/scan/trigger",
            headers={"Origin": "https://attacker.example"},
        )
        assert response.status_code == 403

        response = await client.get("/", headers={"Host": "attacker.example"})
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_article_target_path_cannot_escape_clone(tmp_path, monkeypatch):
    article_id = "safe-article"
    (tmp_path / f"{article_id}.json").write_text(
        json.dumps({"html": "<article>Safe</article>", "title": "Safe"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(articles, "GENERATED_DIR", tmp_path)

    app = FastAPI()
    app.include_router(articles.router)
    app.state.settings = Settings(web_allow_repo_writes=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/articles/{article_id}/push",
            json={
                "repo_url": "https://github.com/acme/site.git",
                "branch": "main",
                "file_path": "../../outside.html",
            },
        )
    assert response.status_code == 400
    assert not (tmp_path.parent / "outside.html").exists()
