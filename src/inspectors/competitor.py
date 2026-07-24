"""Track competitor website changes over time."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("data/competitors")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


class CompetitorTracker:
    """Periodically snapshot competitor pages and detect changes."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def snapshot(self, url: str) -> dict:
        """Take a snapshot of a competitor page."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "SiteInspector/1.0"})
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            logger.warning(f"Failed to fetch competitor {url}: {e}")
            return {"url": url, "error": str(e), "timestamp": datetime.utcnow().isoformat()}

        soup = BeautifulSoup(html, "html.parser")
        title = (soup.find("title") or "").get_text(strip=True) if soup.find("title") else ""
        h1 = (soup.find("h1") or "").get_text(strip=True) if soup.find("h1") else ""
        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            meta_desc = meta.get("content", "")
        word_count = len(soup.get_text(separator=" ", strip=True).split())
        html_hash = hashlib.md5(html.encode()).hexdigest()

        snapshot = {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "h1": h1,
            "meta_description": meta_desc,
            "word_count": word_count,
            "html_hash": html_hash,
        }

        # Save snapshot
        domain = urlparse(url).netloc.replace(".", "_")
        snap_file = SNAPSHOT_DIR / f"{domain}.jsonl"
        with open(snap_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        # Compare with previous
        previous = self._get_previous(snap_file)
        if previous:
            changes = []
            for field in ["title", "h1", "meta_description", "word_count"]:
                if previous.get(field) != snapshot.get(field):
                    changes.append({
                        "field": field,
                        "before": previous.get(field, ""),
                        "after": snapshot.get(field, ""),
                    })
            if html_hash != previous.get("html_hash", ""):
                changes.append({"field": "content_hash", "before": "", "after": "changed"})
            snapshot["changes"] = changes
            snapshot["previous_timestamp"] = previous["timestamp"]

        return snapshot

    async def snapshot_all(self, urls: list[str]) -> list[dict]:
        """Take snapshots of multiple competitor URLs."""
        results = []
        for url in urls:
            results.append(await self.snapshot(url))
        return results

    def get_history(self, url: str, limit: int = 10) -> list[dict]:
        """Get snapshot history for a competitor."""
        domain = urlparse(url).netloc.replace(".", "_")
        snap_file = SNAPSHOT_DIR / f"{domain}.jsonl"
        if not snap_file.exists():
            return []
        snapshots = []
        with open(snap_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    snapshots.append(json.loads(line))
        return snapshots[-limit:]

    @staticmethod
    def _get_previous(snap_file: Path) -> dict | None:
        """Get the most recent previous snapshot."""
        snapshots = []
        with open(snap_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    snapshots.append(json.loads(line))
        return snapshots[-2] if len(snapshots) >= 2 else None
