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


async def fetch_silver_price(history_days: int = 30) -> Optional[dict]:
    """Fetch current silver spot price + historical data from Yahoo Finance.

    Uses COMEX Silver Futures (SI=F) as the benchmark.

    Returns dict with: price, change, change_pct, high, low, history, source, updated
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch 30-day history (includes current price)
            resp = await client.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/SI=F",
                params={"range": "1mo", "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["chart"]["result"][0]
            meta = result["meta"]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("previousClose", price)
            change = round(price - prev_close, 2) if prev_close else 0
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0

            # Build real historical data from daily closes
            history = []
            for i in range(len(timestamps)):
                close_val = quotes["close"][i]
                if close_val is None:
                    continue
                dt = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).strftime("%Y-%m-%d")
                high_val = quotes["high"][i]
                low_val = quotes["low"][i]
                history.append({
                    "date": dt,
                    "price": round(close_val, 2),
                    "high": round(high_val, 2) if high_val else round(close_val, 2),
                    "low": round(low_val, 2) if low_val else round(close_val, 2),
                })

            # 24h high/low from the most recent day's range
            recent_highs = [h["high"] for h in history[-3:]]
            recent_lows = [h["low"] for h in history[-3:]]

            logger.info(
                f"Silver price: ${price:.2f} ({change:+.2f}, {change_pct:+.2f}%), "
                f"{len(history)} days history"
            )
            return {
                "price": round(price, 2),
                "change": change,
                "change_pct": change_pct,
                "high_24h": round(max(recent_highs), 2) if recent_highs else price,
                "low_24h": round(min(recent_lows), 2) if recent_lows else price,
                "source": "yahoo_finance",
                "updated": datetime.now(timezone.utc).isoformat(),
                "currency": "USD",
                "unit": "troy_ounce",
                "history": history,
            }
    except Exception as e:
        logger.warning(f"Yahoo Finance failed: {e}")

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
    """Save price data with history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(price_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_len = len(price_data.get("history", []))
    logger.info(f"Updated {output_path}: ${price_data['price']:.2f}, {history_len} days history")


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
