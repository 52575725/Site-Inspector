import json

import pytest
from bs4 import BeautifulSoup

from src.fixers.jsonld_generator import JsonLdGenerator


HTML = """<html><head><title>Silver Market Guide</title><meta name="description" content="A practical market guide"><meta name="author" content="Mike Lin"><meta property="article:published_time" content="2026-07-24"><meta property="og:image" content="https://example.com/guide.jpg"></head><body><h1>Silver Market Guide</h1></body></html>"""


def test_page_type_detection_handles_english_and_japanese_articles():
    assert JsonLdGenerator._guess_page_type("https://example.com/blog/guide/") == "Article"
    assert JsonLdGenerator._guess_page_type("https://example.com/jp/blog/guide/") == "Article"
    assert JsonLdGenerator._guess_page_type("https://example.com/blog/") == "CollectionPage"
    assert JsonLdGenerator._guess_page_type("https://example.com/") == "Home"


def test_existing_types_are_collected_from_graphs():
    soup = BeautifulSoup(
        '<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"Organization"},{"@type":"WebSite"}]}</script>',
        "html.parser",
    )
    assert JsonLdGenerator._get_existing_types(soup) == {"Organization", "WebSite"}


@pytest.mark.asyncio
async def test_home_schema_generation_is_idempotent_and_uses_organization():
    fixer = JsonLdGenerator(domain="https://example.com")
    issue = {"id": 1, "category": "missing_structured_data", "url": "https://example.com/", "file_path": "index.html"}
    first = await fixer.generate_fix(issue, None, HTML)
    second = await fixer.generate_fix(issue, None, first.after_content)
    schemas = [json.loads(tag.string) for tag in BeautifulSoup(first.after_content, "html.parser").find_all("script", type="application/ld+json")]
    assert first.success
    assert {schema["@type"] for schema in schemas} == {"Organization", "WebSite"}
    assert not second.success
    assert fixer.fix_type == "semi_auto"


def test_article_schema_uses_only_available_source_fields():
    fixer = JsonLdGenerator()
    schema = fixer._article_schema(
        "https://example.com/blog/guide/", "Silver Market Guide",
        "A practical market guide", BeautifulSoup(HTML, "html.parser"),
    )
    assert schema["author"]["name"] == "Mike Lin"
    assert schema["datePublished"] == "2026-07-24"
    assert schema["image"] == "https://example.com/guide.jpg"


def test_faq_schema_preserves_source_markup():
    html = """<details><summary>What is fine silver?</summary><p>Fine silver is silver with a purity of at least 99.9 percent.</p></details>"""
    soup = BeautifulSoup(html, "html.parser")
    before = str(soup)

    schema = JsonLdGenerator()._generate_jsonld(
        "FAQPage", "https://example.com/faq/", "FAQ", "", soup,
    )

    assert schema["@type"] == "FAQPage"
    assert schema["mainEntity"][0]["name"] == "What is fine silver?"
    assert str(soup) == before


def test_product_offer_requires_an_explicit_source_price():
    fixer = JsonLdGenerator(domain="https://example.com")
    no_price = fixer._product_schema(
        "https://example.com/products/bar/", "Silver Bar", "A silver bar",
        BeautifulSoup("<main><p>Contact us for a quote.</p></main>", "html.parser"),
    )
    priced = fixer._product_schema(
        "https://example.com/products/bar/", "Silver Bar", "A silver bar",
        BeautifulSoup("<main><p>Price: HK$ 1,280.00</p></main>", "html.parser"),
    )

    assert "offers" not in no_price
    assert priced["offers"]["price"] == "1280.00"
    assert priced["offers"]["priceCurrency"] == "HKD"
    assert "availability" not in priced["offers"]
