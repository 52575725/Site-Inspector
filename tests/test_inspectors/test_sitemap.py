from __future__ import annotations

import pytest

from src.inspectors.sitemap import SitemapInspector


@pytest.fixture
def sitemap_inspector():
    inspector = SitemapInspector()
    inspector.set_crawled_urls([
        "https://www.example.com/",
        "https://www.example.com/about/",
        "https://www.example.com/products/",
    ])
    inspector.set_sitemap_url("https://www.example.com/sitemap.xml")
    return inspector


def test_no_sitemap_url_reports_missing(sitemap_inspector):
    inspector = SitemapInspector()
    inspector.set_crawled_urls(["https://www.example.com/"])
    # Don't set sitemap_url


@pytest.mark.asyncio
async def test_no_sitemap_url_reports_missing():
    inspector = SitemapInspector()
    inspector.set_crawled_urls(["https://www.example.com/"])
    findings = await inspector.inspect("https://www.example.com/", "")
    categories = {f.category for f in findings}
    assert "sitemap_missing" in categories


def test_normalize_urls_strips_trailing_slash():
    urls = ["https://example.com/", "https://example.com/about/"]
    result = SitemapInspector._normalize_urls(urls)
    assert "https://example.com" in result
    assert "https://example.com/about" in result
    assert "https://example.com/" not in result
    assert "https://example.com/about/" not in result


def test_inspector_has_correct_name():
    inspector = SitemapInspector()
    assert inspector.inspector_name == "sitemap"
