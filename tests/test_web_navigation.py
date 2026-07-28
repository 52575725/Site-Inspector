from pathlib import Path

import pytest
from starlette.requests import Request

from src.web.routes.fix_actions import fixes_page
from src.web.routes.fixes import fixes_page as legacy_fixes_page
from src.web.routes.issues import issues_page
from src.web.routes.tools import tools_page


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": path, "headers": []})


def test_sidebar_omits_retired_manual_pages():
    template = (
        Path(__file__).parents[1] / "src" / "web" / "templates" / "base.html"
    ).read_text(encoding="utf-8")

    assert 'href="/issues"' not in template
    assert 'href="/fixes"' not in template
    assert 'href="/tools"' not in template


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "path"),
    [
        (issues_page, "/issues"),
        (fixes_page, "/fixes"),
        (legacy_fixes_page, "/fixes"),
        (tools_page, "/tools"),
    ],
)
async def test_removed_list_pages_redirect_to_dashboard(handler, path):
    response = await handler(_request(path))

    assert response.status_code == 307
    assert response.headers["location"] == "/"
