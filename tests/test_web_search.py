import asyncio

import httpx
import pytest

from src.integrations.web_search import (
    _filter_relevant,
    _parse_bing_html,
    _parse_bing_rss,
    suggest_public_queries,
)


def test_bing_rss_parser_preserves_real_result_evidence():
    xml = """<?xml version="1.0"?><rss><channel>
    <item><title>How to Buy Silver Bars Safely</title>
    <link>https://example.com/silver-guide</link>
    <description>A practical buyer guide.</description></item>
    </channel></rss>"""

    results = _parse_bing_rss(xml, "how to buy silver bars", 5)

    assert results == [{
        "query": "how to buy silver bars",
        "title": "How to Buy Silver Bars Safely",
        "url": "https://example.com/silver-guide",
        "snippet": "A practical buyer guide.",
        "provider": "bing-rss",
    }]


def test_bing_html_parser_and_relevance_filter_reject_generic_buy_results():
    html = """<ol>
    <li class="b_algo"><h2><a href="https://bestbuy.example/">Best Buy Store</a></h2>
    <div class="b_caption"><p>Shop electronics today.</p></div></li>
    <li class="b_algo"><h2><a href="https://silver.example/guide">How to Buy Silver Bars</a></h2>
    <div class="b_caption"><p>Compare bullion dealers and storage options.</p></div></li>
    </ol>"""

    parsed = _parse_bing_html(html, "how to buy silver bars safely", 5)
    relevant = _filter_relevant(parsed, "how to buy silver bars safely")

    assert len(relevant) == 1
    assert relevant[0]["url"] == "https://silver.example/guide"
    assert relevant[0]["provider"] == "bing"


@pytest.mark.asyncio
async def test_google_autocomplete_suggestions_are_deduplicated():
    async def handler(request):
        seed = request.url.params["q"]
        return httpx.Response(200, json=[seed, ["silver bars for sale", "silver bars near me"]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        suggestions = await suggest_public_queries(
            client,
            "silver bars",
            semaphore=asyncio.Semaphore(2),
            limit=10,
        )

    assert suggestions == ["silver bars for sale", "silver bars near me"]
