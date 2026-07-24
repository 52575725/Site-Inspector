from __future__ import annotations

import pytest

from src.fixers.sitemap_fixer import SitemapFixer
from src.sources.local_source import LocalSource

MOCK_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
  <url>
    <loc>https://www.example.com/</loc>
    <lastmod>2024-01-15</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://www.example.com/about/</loc>
    <lastmod>2024-01-10</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://www.example.com/dead-page/</loc>
    <lastmod>2023-06-01</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>"""


@pytest.fixture
def sitemap_fixer():
    return SitemapFixer()


@pytest.mark.asyncio
async def test_removes_dead_url(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    issue = {
        "category": "sitemap_dead_url",
        "url": "https://www.example.com/dead-page/",
    }
    result = await SitemapFixer().generate_fix(issue, source, MOCK_SITEMAP)
    assert result.success
    assert "dead-page" not in result.after_content
    assert "https://www.example.com/" in result.after_content
    assert "https://www.example.com/about/" in result.after_content


@pytest.mark.asyncio
async def test_adds_missing_url(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    issue = {
        "category": "sitemap_missing_url",
        "url": "https://www.example.com/contact/",
    }
    result = await SitemapFixer().generate_fix(issue, source, MOCK_SITEMAP)
    assert result.success
    assert "https://www.example.com/contact/" in result.after_content
    assert "<lastmod>" in result.after_content
    assert "<priority>" in result.after_content


@pytest.mark.asyncio
async def test_does_not_duplicate_existing_url(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    issue = {
        "category": "sitemap_missing_url",
        "url": "https://www.example.com/about/",
    }
    result = await SitemapFixer().generate_fix(issue, source, MOCK_SITEMAP)
    assert result.success
    # Should only appear once (already exists)
    assert result.after_content.count("https://www.example.com/about/") == 1


@pytest.mark.asyncio
async def test_sitemap_missing_returns_failure(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    issue = {
        "category": "sitemap_missing",
        "url": "https://www.example.com/",
    }
    result = await SitemapFixer().generate_fix(issue, source, MOCK_SITEMAP)
    assert not result.success
    assert "cannot auto-create" in result.error_message


@pytest.mark.asyncio
async def test_guess_priority_homepage():
    fixer = SitemapFixer()
    assert fixer._guess_priority("https://example.com/") == 1.0


@pytest.mark.asyncio
async def test_guess_priority_product():
    fixer = SitemapFixer()
    assert fixer._guess_priority("https://example.com/products/silver/") == 0.8


@pytest.mark.asyncio
async def test_guess_priority_jp_home():
    fixer = SitemapFixer(language_paths={"en": "/", "ja": "/jp/"})
    assert fixer._guess_priority("https://example.com/jp/") == 0.9


@pytest.mark.asyncio
async def test_guess_changefreq_blog():
    fixer = SitemapFixer()
    assert fixer._guess_changefreq("https://example.com/blog/post/") == "weekly"


@pytest.mark.asyncio
async def test_guess_changefreq_static():
    fixer = SitemapFixer()
    assert fixer._guess_changefreq("https://example.com/about/") == "monthly"


@pytest.mark.asyncio
async def test_counterpart_url_en_to_jp():
    fixer = SitemapFixer(language_paths={"en": "/", "ja": "/jp/"})
    result = fixer._get_counterpart_url("https://example.com/about/")
    assert result == "https://example.com/jp/about/"


@pytest.mark.asyncio
async def test_counterpart_url_jp_to_en():
    fixer = SitemapFixer(language_paths={"en": "/", "ja": "/jp/"})
    result = fixer._get_counterpart_url("https://example.com/jp/about/")
    assert result == "https://example.com/about/"


@pytest.mark.asyncio
async def test_counterpart_url_homepage():
    fixer = SitemapFixer(language_paths={"en": "/", "ja": "/jp/"})
    result = fixer._get_counterpart_url("https://example.com/")
    assert result == "https://example.com/jp/"
