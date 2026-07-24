from __future__ import annotations

import asyncio
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LighthouseCI:
    """Run Lighthouse performance checks directly (no LHCI server needed)."""

    def __init__(self, lighthouse_path: str = "lighthouse"):
        self.lighthouse_path = shutil.which(lighthouse_path) or lighthouse_path
        self._available = shutil.which(lighthouse_path) is not None

    @property
    def available(self) -> bool:
        return self._available

    async def measure_cwv(self, url: str) -> dict[str, float]:
        """Measure Core Web Vitals for a single URL.

        Returns {lcp_ms, cls, inp_ms, ttfb_ms}.
        """
        if not self._available:
            return {}

        try:
            import json
            import tempfile
            from pathlib import Path

            output_dir = Path(tempfile.mkdtemp())
            output_file = output_dir / "lighthouse.json"

            proc = await asyncio.create_subprocess_exec(
                self.lighthouse_path,
                url,
                "--output=json",
                f"--output-path={output_file}",
                "--chrome-flags=--headless --no-sandbox",
                "--only-categories=performance",
                "--quiet",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)

            if output_file.exists():
                with open(output_file, encoding="utf-8") as f:
                    report = json.load(f)

                audits = report.get("audits", {})
                metrics = {}
                for metric_name, audit_id in [
                    ("lcp_ms", "largest-contentful-paint"),
                    ("cls", "cumulative-layout-shift"),
                    ("ttfb_ms", "server-response-time"),
                ]:
                    audit = audits.get(audit_id)
                    if audit:
                        metrics[metric_name] = audit.get("numericValue", 0)

                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
                return metrics
        except Exception as e:
            logger.error(f"Lighthouse CI measurement failed for {url}: {e}")

        return {}

    async def measure_urls(self, urls: list[str]) -> dict[str, dict]:
        """Measure CWV for multiple URLs sequentially."""
        results = {}
        for url in urls:
            results[url] = await self.measure_cwv(url)
        return results
