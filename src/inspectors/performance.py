from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


class PerformanceInspector(BaseInspector):
    """Inspect page performance using Lighthouse CLI."""

    inspector_name = "performance"

    LCP_THRESHOLDS = {"good": 2500, "needs_improvement": 4000}  # ms
    INP_THRESHOLDS = {"good": 200, "needs_improvement": 500}  # ms
    CLS_THRESHOLDS = {"good": 0.1, "needs_improvement": 0.25}
    TTFB_THRESHOLD = 800  # ms

    def __init__(self, lighthouse_path: str = "lighthouse",
                 lighthouse_flags: str = ""):
        self.lighthouse_path = lighthouse_path
        self.lighthouse_flags = lighthouse_flags
        self._available = shutil.which(lighthouse_path) is not None

    async def setup(self) -> None:
        if not self._available:
            logger.warning("Lighthouse CLI not found, performance inspection will be limited")

    async def teardown(self) -> None:
        pass

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not self._available:
            # Fallback: check basic indicators from HTML
            findings.extend(self._check_basic_performance(url, html_content))
            return findings

        # Run Lighthouse
        lh_result = await self._run_lighthouse(url)
        if not lh_result:
            findings.extend(self._check_basic_performance(url, html_content))
            return findings

        # Parse Lighthouse results
        findings.extend(self._parse_lighthouse_results(url, lh_result))
        return findings

    def _check_basic_performance(self, url: str, html: str) -> list[RawFinding]:
        """Basic performance checks without Lighthouse."""
        findings = []

        if not html:
            return findings

        # Check number of external resources
        import re
        external_count = len(re.findall(r'<(?:script|link|img)\s[^>]*(?:src|href)="https?://', html))
        if external_count > 30:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="too_many_resources",
                description=f"Page has {external_count} external resources (30+ may slow loading)",
                raw_metadata={"external_resource_count": external_count},
            ))

        # Check inline styles (could be extracted to CSS)
        inline_style_count = len(re.findall(r'style="[^"]*"', html))
        if inline_style_count > 20:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="excessive_inline_styles",
                description=f"Page has {inline_style_count} inline styles (consider extracting to CSS)",
                raw_metadata={"inline_style_count": inline_style_count},
            ))

        # Check for large base64 embedded images
        b64_images = re.findall(r'src="data:image/[^"]{1000,}"', html)
        if b64_images:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="large_inline_images",
                description=f"Page has {len(b64_images)} large inline base64 images",
                raw_metadata={"inline_image_count": len(b64_images)},
            ))

        return findings

    async def _run_lighthouse(self, url: str) -> dict | None:
        """Run Lighthouse CLI and return parsed JSON results."""
        output_dir = Path(tempfile.mkdtemp())
        output_file = output_dir / "lighthouse.json"

        try:
            cmd = [
                self.lighthouse_path,
                url,
                "--output=json",
                f"--output-path={output_file}",
                "--chrome-flags=--headless --no-sandbox",
                "--quiet",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120,
            )

            if proc.returncode != 0:
                logger.warning(f"Lighthouse exited with code {proc.returncode}")
                return None

            if output_file.exists():
                with open(output_file, encoding="utf-8") as f:
                    return json.load(f)

        except asyncio.TimeoutError:
            logger.warning(f"Lighthouse timed out for {url}")
        except FileNotFoundError:
            logger.debug("Lighthouse binary not found")
        except Exception as e:
            logger.error(f"Lighthouse error for {url}: {e}")
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

        return None

    def _parse_lighthouse_results(self, url: str, report: dict) -> list[RawFinding]:
        findings = []

        audits = report.get("audits", {})

        # Core Web Vitals from Lighthouse
        metrics = {
            "lcp": ("largest-contentful-paint", self.LCP_THRESHOLDS),
            "cls": ("cumulative-layout-shift", self.CLS_THRESHOLDS),
            "tti": ("interactive", None),
            "speed_index": ("speed-index", None),
        }

        for metric_name, (audit_id, thresholds) in metrics.items():
            audit = audits.get(audit_id)
            if not audit or audit.get("score") is None:
                continue

            score = audit["score"]
            display_value = audit.get("displayValue", "")
            numeric = audit.get("numericValue", 0)

            if score < 0.5:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"poor_{metric_name}",
                    description=f"Poor {metric_name.upper()}: {display_value} "
                                f"(score: {score:.0%})",
                    current_value=display_value,
                    raw_metadata={
                        "metric": metric_name,
                        "score": score,
                        "numeric_value": numeric,
                        "display_value": display_value,
                    },
                ))
            elif score < 0.9:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"needs_improvement_{metric_name}",
                    description=f"{metric_name.upper()} needs improvement: {display_value} "
                                f"(score: {score:.0%})",
                    current_value=display_value,
                    raw_metadata={
                        "metric": metric_name,
                        "score": score,
                        "numeric_value": numeric,
                        "display_value": display_value,
                    },
                ))

        # Check specific optimization opportunities
        opportunity_checks = {
            "render_blocking": "render-blocking-resources",
            "unused_css": "unused-css-rules",
            "unused_js": "unused-javascript",
            "offscreen_images": "offscreen-images",
            "total_byte_weight": "total-byte-weight",
            "uses_webp": "uses-webp-images",
            "efficient_animated": "efficient-animated-content",
        }

        for cat, audit_id in opportunity_checks.items():
            audit = audits.get(audit_id)
            if audit and audit.get("score") is not None and audit["score"] < 0.5:
                display_value = audit.get("displayValue", "")
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category=f"optimize_{cat}",
                    description=f"Performance opportunity: {audit.get('title', cat)} — "
                                f"{display_value}",
                    raw_metadata={"audit_id": audit_id, "score": audit["score"]},
                ))

        return findings
