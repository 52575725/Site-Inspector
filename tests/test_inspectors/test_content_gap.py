from __future__ import annotations

import pytest

from src.inspectors.content_gap import ContentGapDetector


@pytest.fixture
def gap_detector():
    detector = ContentGapDetector()
    detector.set_page_pairs([
        {"en": "/", "jp": "/jp/"},
        {"en": "/products/", "jp": "/jp/products/"},
    ])
    detector.set_page_htmls({
        "/": (
            "<html><body>"
            "<h1>Welcome to Helin Silver</h1>"
            "<h2>Our Products</h2><p>Premium silver for industry. " * 30 + "</p>"
            "<h2>Why Choose Us</h2><p>Quality guaranteed. " * 20 + "</p>"
            '<a href="/products/">Products</a>'
            '<a href="/about/">About Us</a>'
            '<a href="/contact/">Contact</a>'
            "</body></html>"
        ),
        "/jp/": (
            "<html><body>"
            "<h1>ヘリンシルバーへようこそ</h1>"
            "<h2>製品</h2><p>高品質の銀。 " * 5 + "</p>"
            '<a href="/jp/products/">製品</a>'
            "</body></html>"
        ),
    })
    return detector


@pytest.mark.asyncio
async def test_detects_word_count_gap(gap_detector):
    findings = await gap_detector.inspect("/", "")
    categories = {f.category for f in findings}
    assert "content_gap_word_count" in categories


@pytest.mark.asyncio
async def test_detects_section_gap(gap_detector):
    findings = await gap_detector.inspect("/", "")
    categories = {f.category for f in findings}
    assert "content_gap_section" in categories


@pytest.mark.asyncio
async def test_detects_link_gap(gap_detector):
    findings = await gap_detector.inspect("/", "")
    categories = {f.category for f in findings}
    assert "content_gap_links" in categories


@pytest.mark.asyncio
async def test_only_runs_once_per_pair(gap_detector):
    findings1 = await gap_detector.inspect("/", "")
    findings2 = await gap_detector.inspect("/jp/", "")
    # Second call for the same pair should return empty
    assert len(findings1) > 0
    assert len(findings2) == 0


@pytest.mark.asyncio
async def test_no_pairs_returns_empty():
    detector = ContentGapDetector()
    findings = await detector.inspect("/", "")
    assert len(findings) == 0


@pytest.mark.asyncio
async def test_no_htmls_returns_empty():
    detector = ContentGapDetector()
    detector.set_page_pairs([{"en": "/", "jp": "/jp/"}])
    findings = await detector.inspect("/", "")
    assert len(findings) == 0


@pytest.mark.asyncio
async def test_equal_content_no_findings():
    detector = ContentGapDetector()
    html = (
        "<html><body>"
        "<h1>Welcome</h1>"
        "<h2>Products</h2><p>" + "Content here. " * 50 + "</p>"
        '<a href="/about/">About</a>'
        "</body></html>"
    )
    detector.set_page_pairs([{"en": "/", "jp": "/jp/"}])
    detector.set_page_htmls({"/": html, "/jp/": html})
    findings = await detector.inspect("/", "")
    # Same content means no gaps
    assert len(findings) == 0


def test_get_visible_text_strips_scripts():
    html = "<html><body><script>console.log('x')</script><h1>Hello World</h1></body></html>"
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    text = ContentGapDetector._get_visible_text(soup)
    assert "console.log" not in text
    assert "Hello World" in text


def test_get_heading_structure():
    html = "<html><body><h1>Main</h1><h2>Sub</h2><h3>Detail</h3><h4>X</h4></body></html>"
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    headings = ContentGapDetector._get_heading_structure(soup)
    assert "main" in headings
    assert "sub" in headings
    assert "detail" in headings
    assert len(headings) == 3  # h4 excluded


def test_heading_exists_in_direct_match():
    assert ContentGapDetector._heading_exists_in("products", ["products", "about us"])


def test_heading_exists_in_substring():
    assert ContentGapDetector._heading_exists_in("silver products", ["silver"])


def test_heading_exists_in_no_match():
    assert not ContentGapDetector._heading_exists_in("unique section", ["home", "contact"])


def test_get_significant_links_filters_boilerplate():
    html = (
        "<html><body>"
        '<a href="/products/">Products</a>'
        '<a href="/privacy/">Privacy Policy</a>'
        '<a href="#section">Skip</a>'
        "</body></html>"
    )
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    links = ContentGapDetector._get_significant_links(soup)
    assert "products" in links
    assert "privacy policy" not in links  # filtered
