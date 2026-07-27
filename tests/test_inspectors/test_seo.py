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


@pytest.mark.asyncio
async def test_detects_literal_title_and_description_ellipsis(seo_inspector):
    html = """<html><head><title>A useful silver jewelry guide...</title>
    <meta name="description" content="Learn how to choose sterling silver jewelry...">
    </head><body><h1>Silver guide</h1></body></html>"""
    findings = await seo_inspector.inspect("https://example.com/guide", html)
    categories = {finding.category for finding in findings}
    assert "title_truncated" in categories
    assert "meta_description_truncated" in categories


@pytest.mark.asyncio
async def test_detects_canonical_mismatch(seo_inspector):
    html = """<html><head><title>Silver Jewelry Product Guide</title>
    <link rel="canonical" href="https://example.com/wrong-page/">
    </head><body><h1>Silver guide</h1></body></html>"""
    findings = await seo_inspector.inspect("https://example.com/guide/", html)
    assert any(finding.category == "canonical_mismatch" for finding in findings)


@pytest.mark.asyncio
async def test_hreflang_accepts_x_default(seo_inspector):
    seo_inspector.set_target_languages({"en": "/", "ja": "/jp/"})
    html = """<html><head><title>Silver Jewelry Product Guide</title>
    <link rel="alternate" hreflang="en" href="https://example.com/guide/">
    <link rel="alternate" hreflang="ja" href="https://example.com/jp/guide/">
    <link rel="alternate" hreflang="x-default" href="https://example.com/guide/">
    </head><body><h1>Silver guide</h1></body></html>"""
    findings = await seo_inspector.inspect("https://example.com/guide/", html)
    assert not any(finding.category == "incomplete_hreflang" for finding in findings)


@pytest.mark.asyncio
async def test_detects_substantial_hidden_seo_text(seo_inspector):
    hidden = " ".join(["wholesale sterling silver jewelry supplier"] * 25)
    html = f"""<html><head><title>Silver Jewelry Product Guide</title></head>
    <body><h1>Silver guide</h1><div style="display:none"><p>{hidden}</p></div></body></html>"""
    findings = await seo_inspector.inspect("https://example.com/guide/", html)
    assert any(finding.category == "hidden_seo_text" for finding in findings)
