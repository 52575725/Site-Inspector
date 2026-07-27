from __future__ import annotations

import json
from typing import Optional

import httpx

from config.settings import Settings


class OllamaClient:
    """Async wrapper around Ollama HTTP API."""

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.default_model = settings.ollama_model
        self.embed_model = settings.ollama_embed_model
        self.timeout = settings.ollama_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def generate_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> dict:
        """Generate a completion with JSON output format."""
        client = await self._get_client()
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "format": "json",
        }
        resp = await client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            return json.loads(data["response"])
        except json.JSONDecodeError:
            return {"raw": data["response"], "error": "JSON parse failed"}

    async def generate_text(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 800,
    ) -> str:
        """Generate free-text completion."""
        client = await self._get_client()
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = await client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        client = await self._get_client()
        payload = {"model": self.embed_model, "input": text}
        resp = await client.post(f"{self.base_url}/api/embed", json=payload)
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
