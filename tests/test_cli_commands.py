from __future__ import annotations

from src.cli.commands import _create_engine


def test_create_engine_applies_target_and_page_limit(monkeypatch):
    captured = {}

    class FakeSettings:
        target_name = "default"
        target_base_url = "https://default.example"
        target_languages = ["en"]
        crawl_max_pages = 200

        @classmethod
        def load(cls):
            return cls()

        @classmethod
        def load_target(cls, name):
            assert name == "other"
            return {
                "base_url": "https://other.example",
                "languages": ["en", "ja"],
            }

    class FakeEngine:
        def __init__(self, settings, factory):
            captured["settings"] = settings
            captured["factory"] = factory

    monkeypatch.setattr("src.cli.commands.Settings", FakeSettings)
    monkeypatch.setattr("src.cli.commands.get_session_factory", lambda settings: "factory")
    monkeypatch.setattr("src.cli.commands.Engine", FakeEngine)

    _create_engine(target_name="other", page_limit=5)

    settings = captured["settings"]
    assert settings.target_name == "other"
    assert settings.target_base_url == "https://other.example"
    assert settings.target_languages == ["en", "ja"]
    assert settings.crawl_max_pages == 5
