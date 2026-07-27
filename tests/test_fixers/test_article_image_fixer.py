from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

import src.fixers.article_image_fixer as article_module
from src.fixers.article_image_fixer import ArticleImageFixer
from src.integrations.image_search import ImageResult


def _article_html(word_count: int = 260) -> str:
    words = " ".join(["silver"] * word_count)
    return f"""<html><head><title>Complete Silver Jewelry Care Guide</title></head>
    <body><article><h1>How to Care for Sterling Silver Jewelry</h1>
    <p>Practical advice for keeping silver jewelry clean and bright.</p>
    <h2>Clean silver safely</h2><p>{words}</p>
    <h2>Store jewelry correctly</h2><p>Use a dry lined box for storage.</p>
    <h2>Prevent tarnish</h2><p>Keep each item away from moisture.</p>
    </article></body></html>"""


@pytest.mark.asyncio
async def test_inserts_three_searched_images_across_article(monkeypatch, tmp_path):
    def fake_search(query, count, *keys):
        slug = str(abs(hash(query)))
        return [
            ImageResult(
                url=f"https://images.example/{slug}-{index}.jpg",
                thumb_url="",
                alt_text=f"{query} image {index}",
                photographer="Example Photographer",
                source="wikimedia",
                width=1200,
                height=800,
                page_url="https://commons.wikimedia.org/example",
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            for index in range(count)
        ]

    def fake_download(url, dest_dir, filename):
        path = Path(dest_dir) / filename
        path.write_bytes(b"source image")
        return str(path)

    def fake_convert(path, quality):
        source = Path(path)
        target = source.with_suffix(".webp")
        target.write_bytes(b"webp image")
        return str(target)

    monkeypatch.setattr(article_module, "search_images", fake_search)
    monkeypatch.setattr(article_module, "download_image", fake_download)
    monkeypatch.setattr(article_module, "convert_to_webp", fake_convert)

    fixer = ArticleImageFixer(max_images=4)
    result = await fixer.generate_fix(
        {
            "id": 1,
            "category": "article_no_images",
            "url": "https://example.com/blog/silver-care/",
            "file_path": "blog/silver-care/index.html",
        },
        SimpleNamespace(root=tmp_path),
        _article_html(),
    )

    assert result.success
    soup = BeautifulSoup(result.after_content, "html.parser")
    figures = soup.select("article figure.article-media")
    assert len(figures) == 3
    assert figures[0].find_previous_sibling("p").get_text(strip=True).startswith("Practical")
    assert all(figure.select_one("figcaption .article-image-credit a") for figure in figures)
    assert len(list((tmp_path / "images").glob("*.webp"))) == 3
    assert not list((tmp_path / "images").glob("*.jpg"))


@pytest.mark.asyncio
async def test_ai_fallback_is_disabled_without_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(article_module, "search_images", lambda *args, **kwargs: [])
    fixer = ArticleImageFixer(openai_api_key="present-but-disabled", ai_fallback_enabled=False)

    result = await fixer.generate_fix(
        {"id": 2, "url": "https://example.com/blog/test/", "file_path": "test.html"},
        SimpleNamespace(root=tmp_path),
        _article_html(),
    )

    assert not result.success
    assert fixer.image_generator is None
    assert "AI fallback is disabled" in result.error_message


def test_images_are_placed_in_semantically_matching_sections():
    html = """<html><head><title>Shipping Guide</title></head><body><article>
    <h1>Silver Shipping</h1><p>Compare the available routes.</p>
    <h2>Quick Comparison: Air Freight vs Sea Freight</h2>
    <p>Compare aircraft speed with container ship capacity.</p>
    <h2>Air Freight</h2><p>Aircraft provide fast airport-to-airport delivery.</p>
    <h2>Sea Freight</h2><p>Container ships move large cargo through ports.</p>
    <h2>Security Considerations</h2>
    <p>Inspect cargo and customs risks before choosing a secure shipment.</p>
    <h2>Customs Clearance</h2><p>Border inspection requires import documents.</p>
    </article></body></html>"""
    images = [
        {
            "local_path": "/images/ship.webp",
            "query": "container ship port",
            "alt_text": "Container vessel at sea",
            "caption": "Container vessel",
            "width": 1200,
            "height": 800,
        },
        {
            "local_path": "/images/plane.webp",
            "query": "air cargo aircraft airport",
            "alt_text": "Cargo airplane",
            "caption": "Cargo airplane",
            "width": 1200,
            "height": 800,
        },
        {
            "local_path": "/images/customs.webp",
            "query": "customs cargo inspection",
            "alt_text": "Customs inspection",
            "caption": "Customs inspection",
            "width": 1200,
            "height": 800,
        },
    ]
    soup = BeautifulSoup(html, "html.parser")

    inserted = ArticleImageFixer()._insert_images(soup, images, include_hero=True)

    assert inserted == 3
    placements = {
        image["local_path"]: image["placement_heading"] for image in images
    }
    assert placements == {
        "/images/plane.webp": "Air Freight",
        "/images/ship.webp": "Sea Freight",
        "/images/customs.webp": "Customs Clearance",
    }
    assert images[1]["placement_heading"] == "Air Freight"
    assert ArticleImageFixer._validate_document_integrity(html, soup, 3) is None


def test_integrity_validation_rejects_article_text_changes():
    html = """<html><body><article><h1>Guide</h1><p>Original text.</p>
    <h2>Air Freight</h2><p>Aircraft delivery details.</p></article></body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    soup.find("p").string = "Changed text."

    error = ArticleImageFixer._validate_document_integrity(html, soup, 0)

    assert error == "article text changed during image insertion"


def test_semantic_terms_do_not_treat_airport_or_import_as_port():
    airport_terms = ArticleImageFixer._semantic_terms("air cargo airport")
    import_terms = ArticleImageFixer._semantic_terms("customs import documents")

    assert "concept:air-freight" in airport_terms
    assert "concept:sea-freight" not in airport_terms
    assert "concept:customs" in import_terms
    assert "concept:sea-freight" not in import_terms
