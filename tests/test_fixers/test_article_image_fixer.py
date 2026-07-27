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
