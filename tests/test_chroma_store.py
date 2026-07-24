from __future__ import annotations

import pytest

from config.settings import Settings
from src.storage.chroma_store import ChromaStore


class FakeCollection:
    def __init__(self, distance):
        self.distance = distance
        self.where = None

    def get(self):
        return {"ids": ["existing"]}

    def query(self, **kwargs):
        self.where = kwargs.get("where")
        return {"ids": [["match"]], "distances": [[self.distance]]}


@pytest.mark.asyncio
async def test_cosine_distance_and_target_filter(tmp_path):
    store = ChromaStore(Settings(data_dir=tmp_path))
    store._available = True
    store._collection = FakeCollection(distance=0.05)

    assert await store.find_similar("fingerprint", target_id=7) == "match"
    assert store._collection.where == {"target_id": 7}

    store._collection = FakeCollection(distance=0.95)
    assert await store.find_similar("fingerprint", target_id=7) is None
