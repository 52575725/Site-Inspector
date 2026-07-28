from __future__ import annotations

import pytest

from src.inspectors.headers import HeadersInspector


@pytest.mark.asyncio
async def test_server_level_headers_are_marked_for_site_aggregation():
    inspector = HeadersInspector()
    await inspector.setup()

    findings = await inspector.inspect(
        "https://example.com/a",
        "<html></html>",
        {"content-type": "text/html; charset=utf-8", "server": "nginx"},
    )

    csp = next(item for item in findings if item.category == "missing_content_security_policy")
    leak = next(item for item in findings if item.category == "info_leak_server")
    assert csp.scope == "site"
    assert csp.group_key == "missing_content_security_policy"
    assert leak.scope == "site"


@pytest.mark.asyncio
async def test_missing_response_headers_remains_page_scoped():
    findings = await HeadersInspector().inspect(
        "https://example.com/a", "<html></html>", {},
    )

    assert findings[0].category == "headers_no_response_headers"
    assert findings[0].scope == "page"
