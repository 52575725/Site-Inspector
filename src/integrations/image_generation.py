"""Optional AI fallback for article images.

Search providers must be exhausted before this client is called.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class OpenAIImageGenerator:
    def __init__(self, api_key: str, model: str = "gpt-image-2"):
        self.api_key = api_key
        self.model = model

    async def generate(self, prompt: str, output_path: str | Path) -> Path | None:
        if not self.api_key:
            return None

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": "1536x1024",
            "quality": "medium",
            "output_format": "webp",
            "output_compression": 82,
            "n": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=150) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                encoded = response.json()["data"][0]["b64_json"]
            path.write_bytes(base64.b64decode(encoded))
            return path
        except Exception as exc:
            logger.warning("AI image fallback failed for %s: %s", path.name, exc)
            return None
