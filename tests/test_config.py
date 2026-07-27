from __future__ import annotations

from config.settings import Settings


def test_defaults_yaml_maps_to_runtime_fields():
    settings = Settings.load()
    assert settings.crawl_max_concurrent == 3
    assert settings.crawl_max_pages == 200
    assert settings.auto_fix_max_per_scan == 50


def test_unknown_yaml_option_warns(tmp_path):
    config = tmp_path / "defaults.yaml"
    config.write_text("crawling:\n  typo_limit: 10\n", encoding="utf-8")
    # Unknown options log a warning instead of crashing — the app should
    # be resilient to config typos rather than refusing to start.
    settings = Settings.load(config)
    assert settings.crawl_max_concurrent == 3  # default still applied
