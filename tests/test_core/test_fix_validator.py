from src.core.fix_validator import validate_html


BASE = """<html><head><title>About Us</title><meta name="description" content="A clear company description"></head><body><h1>About Us</h1><p>Substantial body content that is retained across the proposed SEO change.</p></body></html>"""


def test_rejects_duplicate_h1_insertion():
    changed = BASE.replace("<h1>About Us</h1>", "<h1>About Us</h1><h1>About Us</h1>")
    result = validate_html("about/index.html", BASE, changed)
    assert not result.passed
    assert any("Multiple H1" in error for error in result.errors)


def test_rejects_new_truncated_jsonld():
    changed = BASE.replace(
        "</head>",
        '<script type="application/ld+json">{"@type":"Article","url":"https://example.com/bl..."}</script></head>',
    )
    result = validate_html("about/index.html", BASE, changed)
    assert not result.passed
    assert any("truncat" in error.lower() for error in result.errors)


def test_accepts_non_destructive_metadata_change():
    changed = BASE.replace("</head>", '<link rel="canonical" href="https://example.com/about/"></head>')
    assert validate_html("about/index.html", BASE, changed).passed
