from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


def _find_exe(name: str) -> str | None:
    """Find executable in PATH, with Windows .cmd fallback."""
    path = shutil.which(name)
    if path:
        return path
    if sys.platform == "win32":
        path = shutil.which(name + ".cmd")
    return path


class AccessibilityInspector(BaseInspector):
    """Inspect accessibility using axe-core (via Node.js subprocess) + direct HTML checks."""

    inspector_name = "accessibility"

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        # Direct HTML checks (no external binary needed)
        findings.extend(self._check_images_alt(url, html_content))
        findings.extend(self._check_lang_attribute(url, html_content))
        findings.extend(self._check_form_labels(url, html_content))
        findings.extend(self._check_iframe_titles(url, html_content))

        # Try axe-core subprocess analysis
        axe_findings = await self._run_axe_core(url, html_content)
        findings.extend(axe_findings)

        return findings

    def _check_images_alt(self, url: str, html: str) -> list[RawFinding]:
        soup = BeautifulSoup(html, "html.parser")
        findings = []

        for img in soup.find_all("img"):
            alt = img.get("alt")
            src = img.get("src", "")[:100]

            if alt is None:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="missing_alt_text",
                    description=f"Image missing alt attribute: {src}",
                    element=str(img)[:200],
                    current_value="missing",
                ))
            elif alt.strip() == "":
                # Only flag for non-decorative images
                role = img.get("role", "")
                if role != "presentation":
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="empty_alt_text",
                        description=f"Image has empty alt text (may need descriptive text): {src}",
                        element=str(img)[:200],
                        current_value="empty",
                    ))

        return findings

    def _check_lang_attribute(self, url: str, html: str) -> list[RawFinding]:
        soup = BeautifulSoup(html, "html.parser")
        html_tag = soup.find("html")
        if html_tag and not html_tag.get("lang"):
            return [RawFinding(
                url=url, inspector=self.inspector_name,
                category="missing_lang_attribute",
                description="<html> element has no lang attribute",
            )]
        return []

    def _check_form_labels(self, url: str, html: str) -> list[RawFinding]:
        soup = BeautifulSoup(html, "html.parser")
        findings = []

        for inp in soup.find_all(["input", "select", "textarea"]):
            inp_type = inp.get("type", "text")
            if inp_type in ("hidden", "submit", "reset", "button", "image"):
                continue

            inp_id = inp.get("id")
            has_aria_label = inp.get("aria-label") or inp.get("aria-labelledby")
            has_placeholder = inp.get("placeholder")

            # Check for associated label (explicit `for` attribute)
            if inp_id:
                label = soup.find("label", attrs={"for": inp_id})
                if label:
                    continue

            # Check for wrapped/implicit label (<label><input> Text</label>)
            parent_label = inp.find_parent("label")
            if parent_label:
                continue

            if has_aria_label:
                continue

            # Only flag as missing if it has neither placeholder nor aria-label
            if not has_placeholder:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="missing_form_label",
                    description=f"Form field missing associated <label>: "
                                f"<{inp.name} type='{inp_type}'>",
                    element=str(inp)[:150],
                ))

        return findings

    def _check_iframe_titles(self, url: str, html: str) -> list[RawFinding]:
        soup = BeautifulSoup(html, "html.parser")
        findings = []

        for iframe in soup.find_all("iframe"):
            if not iframe.get("title"):
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="missing_iframe_title",
                    description="iframe missing title attribute",
                    element=str(iframe)[:150],
                ))

        return findings

    async def _run_axe_core(self, url: str, html: str) -> list[RawFinding]:
        """Run axe-core via Node.js subprocess on saved HTML file."""
        findings = []

        node_path = _find_exe("node")
        if not node_path:
            logger.debug("Node.js not found, skipping axe analysis")
            return findings

        # Check if axe-core CLI is available
        try:
            proc = await asyncio.create_subprocess_exec(
                node_path, "-e", "require('@axe-core/cli');",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if proc.returncode != 0:
                logger.debug("@axe-core/cli not installed, skipping axe analysis")
                return findings
        except Exception:
            return findings

        # Write HTML to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            temp_path = f.name

        npx_path = _find_exe("npx") or "npx"

        try:
            proc = await asyncio.create_subprocess_exec(
                npx_path, "axe", temp_path, "--stdout",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30,
            )

            if proc.returncode != 0 and not stdout:
                logger.debug(f"axe-core returned code {proc.returncode}")
                return findings

            if stdout:
                try:
                    results = json.loads(stdout.decode())
                    for violation in results:
                        if isinstance(violation, dict):
                            for node in violation.get("nodes", []):
                                findings.append(RawFinding(
                                    url=url, inspector=self.inspector_name,
                                    category=f"wcag_{violation.get('id', 'unknown')}",
                                    description=f"[{violation.get('impact', 'unknown')}] "
                                                f"{violation.get('help', '')}: "
                                                f"{node.get('failureSummary', '')}",
                                    element=node.get("target", [""])[0] if node.get("target") else None,
                                    raw_metadata={
                                        "wcag_id": violation.get("id"),
                                        "impact": violation.get("impact"),
                                        "help_url": violation.get("helpUrl"),
                                    },
                                ))
                except json.JSONDecodeError:
                    logger.debug("Failed to parse axe-core output")
        except asyncio.TimeoutError:
            logger.debug("axe-core timed out")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return findings
