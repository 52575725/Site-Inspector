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
    monkeypatch.setattr(image_search, "_search_openverse", lambda *args: [])
    monkeypatch.setattr(image_search, "_search_wikimedia", lambda *args: [])

    assert image_search.search_images("silver jewelry", count=3) == []


def test_openverse_result_preserves_underlying_source_and_license(monkeypatch):
    payload = {
        "results": [{
            "url": "https://live.staticflickr.com/silver.jpg",
            "thumbnail": "https://api.openverse.org/silver/thumb/",
            "foreign_landing_url": "https://www.flickr.com/photos/example/123",
            "title": "Silver bullion ingot",
            "creator": "Jane Doe",
            "source": "flickr",
            "license": "by-sa",
            "license_version": "4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "width": 1200,
            "height": 800,
            "tags": [{"name": "silver"}, {"name": "bullion"}],
        }]
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))

    result = image_search._search_openverse("silver bullion ingot", 1)[0]

    assert result.source == "flickr"
    assert result.photographer == "Jane Doe"
    assert result.license_name == "CC BY-SA 4.0"
    assert result.license_url.endswith("/by-sa/4.0/")
    assert result.page_url.startswith("https://www.flickr.com/")


def test_search_interleaves_configured_and_keyless_sources(monkeypatch):
    def result(source, index):
        return image_search.ImageResult(
            url=f"https://images.example/{source}-{index}.jpg",
            thumb_url="",
            alt_text="Silver",
            photographer="Contributor",
            source=source,
        )

    monkeypatch.setattr(
        image_search, "_search_unsplash",
        lambda *args: [result("unsplash", index) for index in range(3)],
    )
    monkeypatch.setattr(
        image_search, "_search_pexels",
        lambda *args: [result("pexels", index) for index in range(3)],
    )
    monkeypatch.setattr(image_search, "_search_pixabay", lambda *args: [])
    monkeypatch.setattr(
        image_search, "_search_openverse",
        lambda *args: [result("flickr", index) for index in range(3)],
    )
    monkeypatch.setattr(
        image_search, "_search_wikimedia",
        lambda *args: [result("wikimedia", index) for index in range(3)],
    )

    results = image_search.search_images("silver", count=4)

    assert [item.source for item in results] == [
        "unsplash", "pexels", "flickr", "wikimedia",
    ]


def test_wikimedia_numbered_photo_series_are_deduplicated():
    def result(number):
        filename = f"DB_Cargo_freight_train_at_Hedehusene_Station_{number:02d}.jpg"
        return image_search.ImageResult(
            url=(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/"
                f"{filename}/1280px-{filename}"
            ),
            thumb_url="",
            alt_text="Freight train",
            photographer="Contributor",
            source="wikimedia",
        )

    first = result(1)
    second = result(18)

    assert image_search.image_family_key(first) == image_search.image_family_key(second)
    assert image_search._deduplicate([first, second]) == [first]


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


def test_visual_intent_requires_officer_action_and_documents():
    query = "customs officer inspecting import documents"

    assert not image_search._matches_visual_intent(
        query,
        "Historic customs boundary posts",
        "A rural landscape beside an old customs house",
        "Customs buildings and grasslands",
    )
    assert image_search._matches_visual_intent(
        query,
        "Customs officer checks import paperwork",
        "A border control officer inspecting customs declaration documents",
        "Customs inspections",
    )


def test_visual_intent_rejects_unpacked_silver_for_packaging_query():
    assert not image_search._matches_visual_intent(
        "silver bars in tamper-evident packaging",
        "Silver bullion bars",
        "Loose silver ingots on a table",
        "Silver bullion",
    )
    assert image_search._matches_visual_intent(
        "silver bullion ingot",
        "Cast silver bar",
        "A close-up of a silver bullion bar",
        "Silver ingots",
    )
    assert not image_search._matches_visual_intent(
        "silver bullion ingot",
        "Gold bullion bar",
        "A certified investment gold ingot",
        "Gold bullion; Silver objects",
    )
    assert not image_search._matches_visual_intent(
        "Hong Kong silver bullion ingot",
        "Early medieval ingot",
        "A small ingot of gold photographed beside a silver comparison scale",
        "Silver objects",
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
    assert image_search.broaden_image_query(
        "customs officer inspecting import documents"
    ) == "customs cargo inspection"


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
