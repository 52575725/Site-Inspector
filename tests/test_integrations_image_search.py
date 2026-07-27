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
