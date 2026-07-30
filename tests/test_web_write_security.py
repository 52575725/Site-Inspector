import json

import httpx
import pytest
from fastapi import FastAPI

from config.settings import Settings
from src.agents.article_orchestrator import ArticleOrchestratorAgent
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


@pytest.mark.asyncio
async def test_agent_article_cannot_publish_before_joint_quality_gate(tmp_path, monkeypatch):
    article_id = "gated-article"
    settings = Settings(
        web_allow_repo_writes=True,
        data_dir=tmp_path / "data",
    )
    orchestrator = ArticleOrchestratorAgent(settings.data_dir / "article-agent-runs")
    state = orchestrator.start("https://example.com", {})
    state = orchestrator.complete_research(state, "a" * 32, {"profile": {}})
    state = orchestrator.begin_writing(state)
    long_html = (
        "<article><h1>Ready draft</h1><p>"
        + " ".join(["useful original guidance"] * 100)
        + "</p><h2>First section</h2><p>Details.</p>"
        + "<h2>Second section</h2><p>Details.</p></article>"
    )
    state = orchestrator.complete_writing(state, {"id": article_id, "html": long_html})
    (tmp_path / f"{article_id}.json").write_text(
        json.dumps({
            "html": long_html,
            "title": "Ready draft",
            "agent_run_id": state.run_id,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(articles, "GENERATED_DIR", tmp_path)

    app = FastAPI()
    app.include_router(articles.router)
    app.state.settings = settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/articles/{article_id}/push",
            json={"repo_url": "https://github.com/acme/site", "branch": "main"},
        )

    assert response.status_code == 409
    assert "joint quality" in response.json()["detail"]
