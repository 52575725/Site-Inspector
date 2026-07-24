import pytest

from src.fixers.mobile_css_fixer import MobileCssFixer


HTML = "<html><head><title>Page</title></head><body><a href='/'>Home</a><main>Content</main></body></html>"


@pytest.mark.asyncio
async def test_mobile_css_fix_is_idempotent():
    fixer = MobileCssFixer()
    issue = {"id": 1, "category": "small_font_size", "file_path": "index.html"}

    first = await fixer.generate_fix(issue, None, HTML)
    second = await fixer.generate_fix(issue, None, first.after_content)

    assert first.success
    assert "[Site Inspector:small_font_size]" in first.after_content
    assert not second.success
    assert second.after_content == first.after_content


@pytest.mark.asyncio
async def test_touch_target_fix_does_not_restyle_every_link():
    fixer = MobileCssFixer()
    issue = {"id": 2, "category": "small_touch_targets", "file_path": "index.html"}
    result = await fixer.generate_fix(issue, None, HTML)
    assert result.success
    assert "a, button" not in result.after_content
    assert "display: inline-flex" not in result.after_content
