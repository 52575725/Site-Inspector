from __future__ import annotations

from config.settings import Settings


def test_defaults_yaml_maps_to_runtime_fields():
    settings = Settings.load()
    assert settings.crawl_max_concurrent == 3
    assert settings.crawl_max_pages == 200
    assert settings.auto_fix_max_per_scan == 200


def test_unknown_yaml_option_fails_fast(tmp_path):
    config = tmp_path / "defaults.yaml"
    config.write_text("crawling:\n  typo_limit: 10\n", encoding="utf-8")
    try:
        Settings.load(config)
    except ValueError as exc:
        assert "crawling.typo_limit" in str(exc)
    else:
        raise AssertionError("Unknown configuration option was silently accepted")
