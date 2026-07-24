from src.fixers.content_generator import ContentGenerator


def test_generated_html_removes_executable_content():
    fixer = ContentGenerator()
    output = fixer._sanitize_output(
        '<h2 onclick="alert(1)">Title</h2>'
        '<a href="javascript:alert(1)">link</a>'
        '<script>alert(1)</script>'
    )
    assert "<script" not in output
    assert "onclick" not in output
    assert "javascript:" not in output
    assert "Title" in output
