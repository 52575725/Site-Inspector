from __future__ import annotations

import json

import pytest

from src.fixers.breadcrumb_fixer import BreadcrumbFixer
from src.sources.local_source import LocalSource


@pytest.fixture
def breadcrumb_fixer():
    return BreadcrumbFixer()


@pytest.mark.asyncio
async def test_generates_breadcrumb_from_url(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>Products</title></head><body></body></html>"
    issue = {
        "category": "missing_breadcrumb",
        "url": "https://example.com/products/silver-bars/",
    }
    result = await BreadcrumbFixer().generate_fix(issue, source, html)
    assert result.success
    assert "application/ld+json" in result.after_content
    ld_json = json.loads(
        result.after_content.split("application/ld+json")[1].split(">")[1].split("<")[0]
    )
    assert ld_json["@type"] == "BreadcrumbList"
    assert len(ld_json["itemListElement"]) == 3  # Home > Products > Silver Bars
    assert ld_json["itemListElement"][0]["name"] == "Home"
    assert ld_json["itemListElement"][0]["position"] == 1


@pytest.mark.asyncio
async def test_generates_japanese_breadcrumb(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    fixer = BreadcrumbFixer(
        language_paths={"en": "/", "ja": "/jp/"},
        slug_translations={
            "ja": {"home": "ホーム", "products": "製品"},
        },
    )
    html = "<html><head><title>製品</title></head><body></body></html>"
    issue = {
        "category": "missing_breadcrumb",
        "url": "https://example.com/jp/products/",
    }
    result = await fixer.generate_fix(issue, source, html)
    assert result.success
    ld_json = json.loads(
        result.after_content.split("application/ld+json")[1].split(">")[1].split("<")[0]
    )
    assert ld_json["itemListElement"][0]["name"] == "ホーム"
    assert ld_json["itemListElement"][1]["name"] == "製品"
    assert "/jp/" in ld_json["itemListElement"][0]["item"]


@pytest.mark.asyncio
async def test_handles_schema_missing_type_breadcrumb(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>About</title></head><body></body></html>"
    issue = {
        "category": "schema_missing_type",
        "url": "https://example.com/about/",
        "suggested_value": "BreadcrumbList",
    }
    result = await BreadcrumbFixer().generate_fix(issue, source, html)
    assert result.success
    assert "BreadcrumbList" in result.after_content


@pytest.mark.asyncio
async def test_skips_non_breadcrumb_schema_missing(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>About</title></head><body></body></html>"
    issue = {
        "category": "schema_missing_type",
        "url": "https://example.com/about/",
        "suggested_value": "Organization, WebSite",
    }
    result = await BreadcrumbFixer().generate_fix(issue, source, html)
    assert not result.success
    assert "not BreadcrumbList" in result.error_message


@pytest.mark.asyncio
async def test_homepage_only_returns_failure(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>Home</title></head><body></body></html>"
    issue = {
        "category": "missing_breadcrumb",
        "url": "https://example.com/",
    }
    result = await BreadcrumbFixer().generate_fix(issue, source, html)
    assert not result.success
    assert "Could not generate breadcrumb" in result.error_message


@pytest.mark.asyncio
async def test_path_to_name_with_translations():
    fixer = BreadcrumbFixer(
        language_paths={"en": "/", "ja": "/jp/"},
        slug_translations={
            "ja": {"products": "製品", "about": "会社概要", "blog": "市場情報", "contact": "お問い合わせ"},
        },
    )
    assert fixer._path_to_name("products", "ja") == "製品"
    assert fixer._path_to_name("about", "ja") == "会社概要"
    assert fixer._path_to_name("blog", "ja") == "市場情報"
    assert fixer._path_to_name("contact", "ja") == "お問い合わせ"


@pytest.mark.asyncio
async def test_path_to_name_english():
    fixer = BreadcrumbFixer()
    assert fixer._path_to_name("silver-bars", "en") == "Silver Bars"
    assert fixer._path_to_name("about-us", "en") == "About Us"


@pytest.mark.asyncio
async def test_path_to_name_fallback():
    fixer = BreadcrumbFixer()
    assert fixer._path_to_name("some-page", "en") == "Some Page"


@pytest.mark.asyncio
async def test_appends_to_head(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head>"
        '<script type="application/ld+json">{"@type": "Organization"}</script>'
        "</head><body></body></html>"
    )
    issue = {
        "category": "missing_breadcrumb",
        "url": "https://example.com/products/",
    }
    result = await BreadcrumbFixer().generate_fix(issue, source, html)
    assert result.success
    # Both scripts should exist
    assert result.after_content.count("application/ld+json") == 2
