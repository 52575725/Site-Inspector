from __future__ import annotations

import json
import urllib.request

from src.integrations import image_search


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_wikimedia_result_preserves_attribution_and_license(monkeypatch):
    payload = {
        "query": {
            "pages": {
                "1": {
                    "pageid": 123,
                    "title": "File:Silver ring.jpg",
                    "imageinfo": [{
                        "mime": "image/jpeg",
                        "thumburl": "https://upload.wikimedia.org/silver-ring.jpg",
                        "thumbwidth": 1200,
                        "thumbheight": 800,
                        "extmetadata": {
                            "ImageDescription": {"value": "<b>Silver ring</b> on white"},
                            "Artist": {"value": "Jane Doe"},
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                        },
                    }],
                }
            }
        }
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))

    result = image_search._search_wikimedia("silver ring", 1)[0]

    assert result.source == "wikimedia"
    assert result.photographer == "Jane Doe"
    assert result.license_name == "CC BY-SA 4.0"
    assert result.page_url.endswith("curid=123")


def test_search_has_no_random_placeholder_fallback(monkeypatch):
    monkeypatch.setattr(image_search, "_search_unsplash", lambda *args: [])
    monkeypatch.setattr(image_search, "_search_pexels", lambda *args: [])
    monkeypatch.setattr(image_search, "_search_pixabay", lambda *args: [])
    monkeypatch.setattr(image_search, "_search_wikimedia", lambda *args: [])

    assert image_search.search_images("silver jewelry", count=3) == []


def test_wikimedia_query_variants_broaden_long_editorial_title():
    variants = image_search._wikimedia_query_variants(
        "LBMA Good Delivery Standards: Complete Guide to Silver Bar Quality Specifications"
    )

    assert variants[0].startswith("LBMA Good Delivery Standards")
    assert variants[1] == "silver bullion ingot"
    assert "Silver Bar" in variants


def test_visual_intent_rejects_silver_homonym_and_accepts_ingot():
    assert not image_search._matches_visual_intent(
        "silver bullion ingot",
        "Bar of Silver",
        "A fresh spring salmon returned to the river",
        "Fish of Scotland",
    )
    assert image_search._matches_visual_intent(
        "silver bullion ingot",
        "Cast silver bar",
        "A close-up of a silver bullion bar",
        "Silver ingots",
    )


def test_visual_queries_vary_by_article_intent():
    assert image_search._visual_query_for("Air freight security for silver") == (
        "cargo aircraft freight"
    )
    assert image_search._visual_query_for("Silver price benchmark and futures") == (
        "silver price chart"
    )
    assert image_search._visual_query_for("Industrial demand from solar panels silver") == (
        "solar panels industry"
    )
    assert image_search._visual_query_for("LBMA silver bar purity") == "silver bullion ingot"


def test_extract_keywords_prioritizes_places_transport_and_scenes():
    html = """<html><head><title>Silver Shipping Guide</title></head><body><article>
    <h1>Choosing an International Shipping Route</h1>
    <p>Hong Kong exporters often use air freight through the airport.</p>
    <p>Large orders can use sea freight and container ships.</p>
    <p>At Rotterdam Port, customs clearance includes cargo inspection.</p>
    </article></body></html>"""

    queries = image_search.extract_keywords_from_html(html, max_queries=4)

    assert any("Hong Kong" in query and "air cargo" in query for query in queries)
    assert any("container ship" in query for query in queries)
    assert any("Rotterdam Port" in query and "customs" in query for query in queries)


def test_extract_visual_facets_recognizes_unlisted_named_place():
    facets = image_search._extract_visual_facets(
        "Cargo arrives at Rotterdam Port before entering a secure warehouse."
    )

    assert any("Rotterdam Port" in facet for facet in facets)


def test_extract_visual_facets_recognizes_unlisted_route_locations():
    facets = image_search._extract_visual_facets(
        "Shipments travel from Antwerp to Busan by sea freight."
    )

    assert any("Antwerp" in facet for facet in facets)
    assert any("Busan" in facet for facet in facets)


def test_visual_query_keeps_location_context():
    assert image_search._visual_query_for("Hong Kong air freight silver") == (
        "Hong Kong cargo aircraft freight"
    )
    assert image_search._visual_query_for("Japan silver") == "Japan"
