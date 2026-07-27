from types import SimpleNamespace

from src.web.routes.scans import resolve_quick_scan_target_name


class FakeSettings(SimpleNamespace):
    @classmethod
    def load_target(cls, name):
        assert name == "helinsilver"
        return {"base_url": "https://helinsilver.com"}


def test_configured_domain_uses_configured_target_name():
    settings = FakeSettings(
        target_name="helinsilver",
        target_base_url="https://helinsilver.com",
    )

    assert resolve_quick_scan_target_name(
        "https://www.helinsilver.com/", None, settings,
    ) == "helinsilver"


def test_arbitrary_domain_keeps_isolated_quick_scan_name():
    settings = FakeSettings(
        target_name="helinsilver",
        target_base_url="https://helinsilver.com",
    )

    assert resolve_quick_scan_target_name(
        "https://example.org/", None, settings,
    ) == "example-org"
    assert resolve_quick_scan_target_name(
        "https://example.org/", "client-site", settings,
    ) == "client-site"
