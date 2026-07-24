from __future__ import annotations

import pytest

from src.fixers.og_image_fixer import OgImageFixer
from src.sources.local_source import LocalSource


@pytest.fixture
def og_fixer():
    return OgImageFixer()


@pytest.mark.asyncio
async def test_adds_og_image_tags(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head>"
        "<title>Test Page</title>"
        '<meta property="og:title" content="Test">'
        "</head><body>"
        '<img src="/images/hero.jpg" width="800">'
        "</body></html>"
    )
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'property="og:image"' in result.after_content
    assert 'property="og:image:width"' in result.after_content
    assert 'property="og:image:height"' in result.after_content


@pytest.mark.asyncio
async def test_og_image_uses_page_image_first(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head><title>Test</title></head><body>"
        '<img src="/images/product.jpg" width="600">'
        "</body></html>"
    )
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    assert "/images/product.jpg" in result.after_content


@pytest.mark.asyncio
async def test_og_image_falls_back_to_default(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><head><title>No Images</title></head><body><p>No img tags</p></body></html>"
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'property="og:image"' in result.after_content


@pytest.mark.asyncio
async def test_adds_twitter_card_tags(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head>"
        "<title>Twitter Test</title>"
        '<meta property="og:title" content="OG Title">'
        '<meta property="og:description" content="OG Desc">'
        '<meta property="og:image" content="https://example.com/img.jpg">'
        "</head><body></body></html>"
    )
    issue = {"category": "missing_twitter_cards", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    assert 'name="twitter:card"' in result.after_content
    assert 'name="twitter:title"' in result.after_content
    assert 'name="twitter:description"' in result.after_content
    assert 'name="twitter:image"' in result.after_content
    assert "summary_large_image" in result.after_content


@pytest.mark.asyncio
async def test_no_head_returns_failure(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = "<html><body><p>No head tag</p></body></html>"
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert not result.success
    assert "No <head> tag" in result.error_message


@pytest.mark.asyncio
async def test_skips_small_images(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head><title>Test</title></head><body>"
        '<img src="/images/icon.png" width="32">'
        '<img src="/images/banner.jpg" width="800">'
        "</body></html>"
    )
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    assert "banner.jpg" in result.after_content


@pytest.mark.asyncio
async def test_does_not_duplicate_existing_og_image(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head>"
        '<meta property="og:image" content="https://example.com/existing.jpg">'
        '<meta property="og:image:width" content="1200">'
        '<meta property="og:image:height" content="630">'
        "</head><body></body></html>"
    )
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    # Should not add duplicate og:image
    assert result.after_content.count('property="og:image"') == 1


@pytest.mark.asyncio
async def test_default_image_configurable():
    fixer = OgImageFixer(default_image="https://custom.com/img.jpg")
    assert fixer.default_image == "https://custom.com/img.jpg"


@pytest.mark.asyncio
async def test_uses_any_image_without_width_attr(mock_site):
    source = LocalSource(mock_site)
    await source.connect()
    html = (
        "<html><head><title>Test</title></head><body>"
        '<img src="/images/photo.jpg">'
        "</body></html>"
    )
    issue = {"category": "missing_og_image", "url": "https://example.com/page"}
    result = await OgImageFixer().generate_fix(issue, source, html)
    assert result.success
    assert "/images/photo.jpg" in result.after_content
