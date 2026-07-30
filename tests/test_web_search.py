import asyncio

import httpx
import pytest

from src.integrations.web_search import (
    _filter_relevant,
    _parse_bing_html,
    _parse_bing_rss,
    _parse_yahoo_html,
    _unwrap_bing_url,
    _unwrap_yahoo_url,
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


def test_relevance_filter_rejects_geography_only_result():
    results = [{
        "query": "Hong Kong silver exporters for industrial use",
        "title": "Hong Kong - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Hong_Kong",
        "snippet": "Hong Kong is a city with industrial businesses.",
        "provider": "yahoo",
    }]

    assert _filter_relevant(results, results[0]["query"]) == []


def test_bing_tracking_url_is_unwrapped_to_the_real_target():
    target = "https://www.lbma.org.uk/prices-and-data"
    import base64
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?u=a1{encoded}&ntb=1"

    assert _unwrap_bing_url(wrapped) == target


def test_yahoo_result_parser_unwraps_target_and_keeps_structure():
    target = "https://bullion.example/lbma-guide"
    wrapped = (
        "https://r.search.yahoo.com/path/RU="
        "https%3A%2F%2Fbullion.example%2Flbma-guide/RK=2/RS=test"
    )
    html = f"""<div class="algo"><div class="compTitle"><a href="{wrapped}">
    <h3>Why LBMA Approval Matters for Silver Bars</h3></a></div>
    <div class="compText"><p>LBMA standards for silver bullion buyers.</p></div></div>"""

    results = _parse_yahoo_html(html, "why LBMA silver bars matter", 5)

    assert _unwrap_yahoo_url(wrapped) == target
    assert results[0]["url"] == target
    assert results[0]["provider"] == "yahoo"


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
