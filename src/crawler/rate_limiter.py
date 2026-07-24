from __future__ import annotations

import asyncio
import random
import time


class TokenBucketRateLimiter:
    """Token bucket algorithm for polite crawling."""

    def __init__(self, rate: float = 0.5, burst: int = 3):
        self.tokens = float(burst)
        self.max_tokens = float(burst)
        self.rate = rate
        self.last_refill = time.monotonic()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        while True:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            wait = 1.0 / self.rate
            await asyncio.sleep(wait)


def backoff_delay(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> float:
    """Exponential backoff with jitter."""
    delay = min(base * (2 ** attempt), max_delay)
    delay *= 0.5 + random.random()
    return delay
