from __future__ import annotations

import pytest

from src.fixers.meta_fixer import MetaFixer
from src.sources.local_source import LocalSource


@pytest.fixture
def meta_fixer():
    return MetaFixer()


@pytest.mark.asyncio
async def test_fix_missing_title(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<!DOCTYPE html><html><head></head><body><h1>Test</h1></body></html>"
    issue = {"category": "missing_title", "url": "https://example.com/products/"}
    result = await MetaFixer().generate_fix(issue, source, html)
    assert result.success
    assert "<title>" in result.after_content
    assert "Products" in result.after_content


@pytest.mark.asyncio
async def test_fix_missing_meta_description(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>Test Page</title></head><body><p>Some content here.</p></body></html>"
    issue = {"category": "missing_meta_description", "url": "https://example.com"}
    result = await MetaFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'name="description"' in result.after_content


@pytest.mark.asyncio
async def test_fix_missing_og_tags(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head>"
        "<title>Test Page Title</title>"
        '<meta name="description" content="Test description">'
        "</head><body></body></html>"
    )
    issue = {"category": "missing_og_tags", "url": "https://example.com"}
    result = await MetaFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'property="og:title"' in result.after_content
    assert 'property="og:description"' in result.after_content


@pytest.mark.asyncio
async def test_fix_missing_canonical(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>Test</title></head><body></body></html>"
    issue = {"category": "missing_canonical", "url": "https://example.com/page"}
    result = await MetaFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'rel="canonical"' in result.after_content


@pytest.mark.asyncio
async def test_fix_missing_viewport(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>Test</title></head><body></body></html>"
    issue = {"category": "missing_viewport_meta", "url": "https://example.com"}
    result = await MetaFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'name="viewport"' in result.after_content
