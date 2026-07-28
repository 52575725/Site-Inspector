from __future__ import annotations

import pytest

from src.inspectors.mobile import MobileInspector


class FakePage:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, script):
        self.calls += 1
        if self.calls == 1:
            return False
        if self.calls == 2:
            return [{
                "tag": "P", "size": 10, "text": "Small readable text",
                "selector": "main > p:nth-of-type(2)", "html": "<p>Small readable text</p>",
            }]
        return [{
            "tag": "A", "w": 18, "h": 20, "text": "Buy",
            "selector": "nav > a:nth-of-type(2)", "html": "<a href='/buy'>Buy</a>",
        }]


@pytest.mark.asyncio
async def test_mobile_findings_include_dom_evidence_and_calibrated_thresholds():
    findings = await MobileInspector()._check_viewport(
        FakePage(), "https://example.com", "small_phone", {"width": 375, "height": 812},
    )

    font = next(item for item in findings if item.category == "small_font_size")
    target = next(item for item in findings if item.category == "small_touch_targets")
    assert font.element == "main > p:nth-of-type(2)"
    assert font.element_html.startswith("<p>")
    assert font.scope == "element"
    assert target.element == "nav > a:nth-of-type(2)"
    assert "24px" in target.description
