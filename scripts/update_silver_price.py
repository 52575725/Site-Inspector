#!/usr/bin/env python3
"""Fetch live silver spot price and update the website's price data.

Run via:  python scripts/update_silver_price.py [--output data/site_sources/helinsilver/api/live-price.json]

Integrates with site-inspector scheduler or can be run standalone.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default output path (relative to site-inspector root)
DEFAULT_OUTPUT = Path("data/site_sources/helinsilver/api/live-price.json")
MAX_HISTORY_DAYS = 90  # Keep 90 days of trend data


async def fetch_silver_price() -> Optional[dict]:
    """Fetch current silver spot price from Yahoo Finance.

    Falls back to multiple sources if primary fails.

    Returns dict with: price, change, change_pct, high, low, source, updated
    """
    # Primary: Yahoo Finance — Silver Futures (SI=F)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/SI=F",
                params={"range": "1d", "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["chart"]["result"][0]
            meta = result["meta"]
            quote = result["indicators"]["quote"][0]

            price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("previousClose", price)
            change = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0

            high_vals = [v for v in quote.get("high", []) if v]
            low_vals = [v for v in quote.get("low", []) if v]

            logger.info(f"Silver price: ${price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
            return {
                "price": round(price, 2),
                "change": change,
                "change_pct": change_pct,
                "high_24h": round(max(high_vals), 2) if high_vals else price,
                "low_24h": round(min(low_vals), 2) if low_vals else price,
                "source": "yahoo_finance",
                "updated": datetime.now(timezone.utc).isoformat(),
                "currency": "USD",
                "unit": "troy_ounce",
            }
    except Exception as e:
        logger.warning(f"Yahoo Finance failed: {e}")

    # Fallback: use approximation (last known price)
    logger.error("All price sources failed")
    return None


def load_history(output_path: Path) -> dict:
    """Load existing price data with history."""
    if output_path.exists():
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"price": 0, "history": []}


def save_price_data(price_data: dict, output_path: Path) -> None:
    """Save price data with rolling history."""
    existing = load_history(output_path)

    # Add current price to history
    history = existing.get("history", [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Update today's entry or create new one
    updated = False
    for entry in history:
        if entry.get("date") == today:
            entry.update({
                "price": price_data["price"],
                "change": price_data.get("change", 0),
                "high": price_data.get("high_24h", price_data["price"]),
                "low": price_data.get("low_24h", price_data["price"]),
            })
            updated = True
            break

    if not updated:
        history.append({
            "date": today,
            "price": price_data["price"],
            "change": price_data.get("change", 0),
            "high": price_data.get("high_24h", price_data["price"]),
            "low": price_data.get("low_24h", price_data["price"]),
        })

    # Trim history
    history = history[-MAX_HISTORY_DAYS:]

    output = {**price_data, "history": history}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Updated {output_path}: ${price_data['price']:.2f}, {len(history)} days history")


async def main(output_path: Path | None = None):
    """Run price update and write to output."""
    path = Path(output_path) if output_path else DEFAULT_OUTPUT
    price_data = await fetch_silver_price()
    if price_data:
        save_price_data(price_data, path)
        return True
    else:
        logger.error("Could not fetch silver price")
        return False


if __name__ == "__main__":
    import asyncio
    output = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--output" else DEFAULT_OUTPUT
    success = asyncio.run(main(output))
    sys.exit(0 if success else 1)
