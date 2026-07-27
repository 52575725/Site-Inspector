from __future__ import annotations

from bs4 import BeautifulSoup

from src.fixers.htag_restructurer import HTagRestructurer


def test_multiple_h1_keeps_visible_main_heading():
    html = """<html><body>
    <div aria-hidden="true"><h1>Hidden SEO heading</h1></div>
    <main><section class="hero"><h1>Visible product heading</h1></section></main>
    </body></html>"""

    result = HTagRestructurer()._fix_multiple_h1(BeautifulSoup(html, "html.parser"))
    soup = BeautifulSoup(result, "html.parser")

    assert [tag.get_text(strip=True) for tag in soup.find_all("h1")] == ["Visible product heading"]
    assert soup.find("h2").get_text(strip=True) == "Hidden SEO heading"
