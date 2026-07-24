from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from config.settings import Settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Vector store for issue deduplication across scans.

    Gracefully degrades to in-memory dedup if chromadb is not installed.
    """

    COLLECTION_NAME = "issue_signatures"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._available = False
        self._collection = None
        self._client = None
        self._in_memory: dict[str, dict] = {}

        data_dir = settings.data_dir / "chroma"
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(data_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info("ChromaDB initialized for issue deduplication")
        except ImportError:
            logger.warning("chromadb not installed, using in-memory dedup")
        except Exception as e:
            logger.warning(f"ChromaDB init failed ({e}), using in-memory dedup")

    def build_fingerprint(self, url: str, inspector: str, category: str,
                          element: str | None, description: str) -> str:
        element_part = element or ""
        desc_part = " ".join(description.split()[:50])
        return f"{url}|{inspector}|{category}|{element_part}|{desc_part}"

    def build_doc_id(self, target_id: int, url_hash: str, inspector: str,
                     category: str, element_hash: str) -> str:
        return f"{target_id}:{url_hash}:{inspector}:{category}:{element_hash}"

    async def find_similar(
        self, fingerprint: str, target_id: int, threshold: float = 0.92,
    ) -> Optional[str]:
        """Find similar existing issue. Returns existing doc_id or None."""
        if not self._available:
            # In-memory fallback: exact fingerprint match only
            if fingerprint in self._in_memory:
                return self._in_memory[fingerprint].get("doc_id")
            return None

        try:
            for attempt in range(2):
                try:
                    existing = self._collection.get()
                    if not existing or not existing["ids"]:
                        return None
                    query_terms = fingerprint.replace("|", " ")
                    results = self._collection.query(
                        query_texts=[query_terms],
                        n_results=1,
                        where={"target_id": target_id},
                    )
                    if results and results["ids"] and results["ids"][0]:
                        distances = results.get("distances", [[1.0]])
                        # Chroma returns cosine distance: 0 is identical and
                        # 1 is dissimilar. threshold is expressed as similarity.
                        if distances[0] and distances[0][0] <= (1.0 - threshold):
                            return results["ids"][0][0]
                    return None
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"ChromaDB query failed: {e}")

        return None

    async def upsert_issue(self, doc_id: str, fingerprint: str, target_id: int,
                           url: str, inspector: str, category: str,
                           severity: float, status: str = "active") -> None:
        now = datetime.utcnow().isoformat()

        if not self._available:
            # In-memory fallback
            existing = self._in_memory.get(fingerprint)
            if existing:
                existing["last_seen"] = now
                existing["status"] = status
            else:
                self._in_memory[fingerprint] = {
                    "doc_id": doc_id, "target_id": target_id,
                    "url": url, "inspector": inspector,
                    "category": category, "severity": severity,
                    "first_seen": now, "last_seen": now, "status": status,
                }
            return

        try:
            existing = self._collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                self._collection.update(
                    ids=[doc_id],
                    metadatas=[{
                        "target_id": target_id, "url": url,
                        "inspector": inspector, "category": category,
                        "severity": severity,
                        "first_seen": existing["metadatas"][0].get("first_seen", now),
                        "last_seen": now, "status": status,
                    }],
                )
            else:
                self._collection.add(
                    ids=[doc_id], documents=[fingerprint],
                    metadatas=[{
                        "target_id": target_id, "url": url,
                        "inspector": inspector, "category": category,
                        "severity": severity,
                        "first_seen": now, "last_seen": now, "status": status,
                    }],
                )
        except Exception as e:
            logger.debug(f"ChromaDB upsert failed: {e}")
            self._in_memory[fingerprint] = {"doc_id": doc_id, "status": status}

    async def mark_resolved(self, doc_id: str) -> None:
        try:
            if self._available:
                self._collection.update(ids=[doc_id], metadatas=[{"status": "resolved"}])
        except Exception:
            pass

    async def mark_dismissed(self, doc_id: str) -> None:
        try:
            if self._available:
                self._collection.update(ids=[doc_id], metadatas=[{"status": "dismissed"}])
        except Exception:
            pass

    def get_active_count(self) -> int:
        try:
            if self._available:
                existing = self._collection.get()
                if existing and existing["metadatas"]:
                    return sum(1 for m in existing["metadatas"] if m.get("status") in ("active", None))
        except Exception:
            pass
        return sum(1 for m in self._in_memory.values() if m.get("status") in ("active", None))
