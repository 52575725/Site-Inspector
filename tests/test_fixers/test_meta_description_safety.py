from bs4 import BeautifulSoup

from src.fixers.meta_fixer import MetaFixer


def test_description_uses_main_copy_not_navigation_or_footer():
    html = """<html><body><header>Home Products About Contact</header><main><p>Hong Kong Changjiang supplies high-purity silver bars to international industrial buyers with documented quality controls and reliable export logistics.</p></main><footer>Privacy WhatsApp Telegram</footer></body></html>"""
    description = MetaFixer._extract_description_text(BeautifulSoup(html, "html.parser"))
    assert description.startswith("Hong Kong Changjiang")
    assert "Products About Contact" not in description
    assert "WhatsApp" not in description
    assert len(description) <= 160
    assert "..." not in description


def test_description_stops_on_a_word_boundary():
    text = "A detailed sentence about international silver trading and responsible sourcing. " * 4
    description = MetaFixer._extract_description_text(
        BeautifulSoup(f"<main><p>{text}</p></main>", "html.parser")
    )
    assert len(description) <= 160
    assert not description.endswith("...")
