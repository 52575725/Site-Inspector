from __future__ import annotations

import pytest

from src.inspectors.seo import SEOInspector


@pytest.fixture
def seo_inspector():
    return SEOInspector()


@pytest.mark.asyncio
async def test_detects_missing_title(seo_inspector):
    html = "<html><head></head><body><h1>Test</h1></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "missing_title" in categories


@pytest.mark.asyncio
async def test_detects_title_too_short(seo_inspector):
    html = "<html><head><title>Hi</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "title_too_short" in categories


@pytest.mark.asyncio
async def test_detects_title_too_long(seo_inspector):
    html = f"<html><head><title>{'X' * 70}</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "title_too_long" in categories


@pytest.mark.asyncio
async def test_detects_missing_meta_description(seo_inspector):
    html = "<html><head><title>Test Page</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "missing_meta_description" in categories


@pytest.mark.asyncio
async def test_detects_missing_h1(seo_inspector):
    html = "<html><head><title>Test</title></head><body><p>No H1</p></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "missing_h1" in categories


@pytest.mark.asyncio
async def test_detects_multiple_h1(seo_inspector):
    html = "<html><body><h1>First</h1><h1>Second</h1></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "multiple_h1" in categories


@pytest.mark.asyncio
async def test_detects_h_tag_skip(seo_inspector):
    html = "<html><body><h1>One</h1><h3>Three</h3></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "h_tag_skip" in categories


@pytest.mark.asyncio
async def test_detects_missing_canonical(seo_inspector):
    html = "<html><head><title>Test</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "missing_canonical" in categories


@pytest.mark.asyncio
async def test_detects_missing_hreflang(seo_inspector):
    seo_inspector.set_target_languages({"en": "/", "ja": "/jp/"})
    html = "<html><head><title>Test</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://www.helinsilver.com/blog/test", html)
    categories = {f.category for f in findings}
    assert "missing_hreflang" in categories


@pytest.mark.asyncio
async def test_detects_missing_og_tags(seo_inspector):
    html = "<html><head><title>Test</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "missing_og_tags" in categories


@pytest.mark.asyncio
async def test_detects_missing_structured_data(seo_inspector):
    html = "<html><head><title>Test</title></head><body></body></html>"
    findings = await seo_inspector.inspect("https://example.com", html)
    categories = {f.category for f in findings}
    assert "missing_structured_data" in categories


@pytest.mark.asyncio
async def test_empty_page(seo_inspector):
    findings = await seo_inspector.inspect("https://example.com", "")
    assert any(f.category == "empty_page" for f in findings)
