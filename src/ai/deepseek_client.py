from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Async client for DeepSeek API (OpenAI-compatible chat completions).

    Matches the same generate_text / health_check / close interface as
    OllamaClient so it can be used as a drop-in replacement for content
    generation tasks that need higher-quality output.
    """

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-chat",
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if DeepSeek API key is valid and reachable."""
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/models",
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def generate_text(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a chat completion via DeepSeek's OpenAI-compatible API."""
        client = await self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            resp = await client.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API error: {e.response.status_code} {e.response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            raise
