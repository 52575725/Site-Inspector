"""Post-fix validation: verify HTML integrity after fixes are applied.

Runs automatically after each batch of fixes before git commit.
Catches common corruption patterns: duplicate elements, truncation,
structural damage, and content loss.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Warning thresholds
MAX_H1_COUNT = 1       # public SEO pages must have a single primary heading
MIN_BODY_LENGTH = 50   # body text shorter than this = content was deleted
MIN_TITLE_LENGTH = 5   # title shorter than this = truncation damage
TRUNCATION_MARKERS = ("...", "…", "[rest truncated")


def _contains_truncation(value) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in TRUNCATION_MARKERS)
    if isinstance(value, dict):
        return any(_contains_truncation(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_truncation(item) for item in value)
    return False


@dataclass
class ValidationResult:
    file_path: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_html(file_path: str, before: str, after: str) -> ValidationResult:
    """Validate that a fix didn't corrupt the HTML.

    Checks:
    - HTML is still parseable
    - Title wasn't truncated (no "...")
    - No excessive H1 duplication
    - Body content wasn't deleted
    - Meta tags still exist
    - No broken tag structure
    """
    result = ValidationResult(file_path=file_path)

    if not after or not after.strip():
        result.passed = False
        result.errors.append("Fix produced empty output — file would be deleted")
        return result

    # Parse both versions
    try:
        before_soup = BeautifulSoup(before, "html.parser")
        after_soup = BeautifulSoup(after, "html.parser")
    except Exception as e:
        result.passed = False
        result.errors.append(f"HTML unparseable after fix: {e}")
        return result

    # ── Title checks ──────────────────────────────────────────
    before_title = before_soup.find("title")
    after_title = after_soup.find("title")

    if before_title and after_title:
        after_text = after_title.get_text(strip=True)
        before_text = before_title.get_text(strip=True)

        # Check for truncation
        if _contains_truncation(after_text) and not _contains_truncation(before_text):
            result.errors.append(
                f"Title was truncated: '{before_text[:80]}' → '{after_text[:80]}'"
            )
            result.passed = False

        # Check for excessive shortening (>50% loss)
        if len(before_text) > 20 and len(after_text) < len(before_text) * 0.5:
            result.errors.append(
                f"Title lost >50% length: {len(before_text)} → {len(after_text)} chars"
            )
            result.passed = False

        # Check for too-short title
        if len(after_text) < MIN_TITLE_LENGTH:
            result.errors.append(f"Title too short after fix: '{after_text}'")
            result.passed = False

    elif before_title and not after_title:
        result.errors.append("Title tag was deleted by fix")
        result.passed = False

    # ── H1 count check ────────────────────────────────────────
    before_h1s = len(before_soup.find_all("h1"))
    after_h1s = len(after_soup.find_all("h1"))
    result.stats["h1_before"] = before_h1s
    result.stats["h1_after"] = after_h1s

    if after_h1s > MAX_H1_COUNT:
        result.errors.append(
            f"Multiple H1 tags after fix: {before_h1s} → {after_h1s}"
        )
        result.passed = False

    # ── Body content checks ───────────────────────────────────
    before_body = before_soup.find("body")
    after_body = after_soup.find("body")

    if before_body and after_body:
        before_text = before_body.get_text(separator=" ", strip=True)
        after_text = after_body.get_text(separator=" ", strip=True)

        result.stats["body_chars_before"] = len(before_text)
        result.stats["body_chars_after"] = len(after_text)

        # Check for content deletion (>30% loss)
        if len(before_text) > 100 and len(after_text) < len(before_text) * 0.7:
            result.errors.append(
                f"Body content lost >30%: {len(before_text)} → {len(after_text)} chars"
            )
            result.passed = False

        if len(after_text) < MIN_BODY_LENGTH:
            result.errors.append(f"Body nearly empty after fix: {len(after_text)} chars")
            result.passed = False

    # ── Meta tag check ────────────────────────────────────────
    before_meta_count = len(before_soup.find_all("meta"))
    after_meta_count = len(after_soup.find_all("meta"))
    result.stats["meta_before"] = before_meta_count
    result.stats["meta_after"] = after_meta_count

    if before_meta_count > 3 and after_meta_count < before_meta_count * 0.5:
        result.errors.append(
            f"Meta tags halved: {before_meta_count} → {after_meta_count}"
        )
        result.passed = False

    before_desc = before_soup.find("meta", attrs={"name": "description"})
    after_desc = after_soup.find("meta", attrs={"name": "description"})
    before_desc_text = before_desc.get("content", "") if before_desc else ""
    after_desc_text = after_desc.get("content", "") if after_desc else ""
    if _contains_truncation(after_desc_text) and not _contains_truncation(before_desc_text):
        result.errors.append("Meta description contains a truncation marker")
        result.passed = False

    # ── Duplicate element detection ───────────────────────────
    # Check for identical consecutive H1s (definite bug)
    all_h1s = [h.get_text(strip=True) for h in after_soup.find_all("h1")]
    duplicate_h1s = sum(1 for i in range(1, len(all_h1s)) if all_h1s[i] == all_h1s[i-1])
    if duplicate_h1s > 0:
        result.errors.append(
            f"Found {duplicate_h1s} consecutive identical H1s — duplicate insertion bug"
        )
        result.passed = False

    # ── Truncation marker check (global) ──────────────────────
    # If "..." appears in places it wasn't before, flag it
    before_marker_count = sum(before.lower().count(marker) for marker in TRUNCATION_MARKERS)
    after_marker_count = sum(after.lower().count(marker) for marker in TRUNCATION_MARKERS)
    if after_marker_count > before_marker_count:
        result.errors.append(
            "New truncation markers were introduced by the fix"
        )
        result.passed = False

    # ── Structured data check ─────────────────────────────────
    before_scripts = before_soup.find_all("script", type="application/ld+json")
    after_scripts = after_soup.find_all("script", type="application/ld+json")

    # Check that schema blocks are still valid JSON
    import json
    parsed_schemas = []
    for script in after_scripts:
        try:
            schema = json.loads(script.string or "{}")
            parsed_schemas.append(schema)
            if _contains_truncation(schema):
                result.errors.append("JSON-LD contains truncated content")
                result.passed = False
        except (json.JSONDecodeError, TypeError):
            result.errors.append("JSON-LD schema is invalid after fix")
            result.passed = False

    normalized_schemas = [
        json.dumps(item, sort_keys=True, ensure_ascii=False)
        for item in parsed_schemas
    ]
    if len(normalized_schemas) != len(set(normalized_schemas)):
        result.errors.append("Duplicate JSON-LD blocks were introduced")
        result.passed = False

    # Log summary
    if result.errors:
        logger.error(f"Validation FAILED for {file_path}: {'; '.join(result.errors)}")
    elif result.warnings:
        logger.warning(f"Validation WARN for {file_path}: {'; '.join(result.warnings)}")
    else:
        logger.debug(f"Validation passed for {file_path}")

    return result


def validate_fix_batch(
    original_files: dict[str, str],   # file_path → original content
    modified_files: dict[str, str],   # file_path → modified content
) -> dict[str, ValidationResult]:
    """Validate all modified files. Returns {file_path: result}."""
    results = {}
    for file_path, after_content in modified_files.items():
        before_content = original_files.get(file_path, "")
        results[file_path] = validate_html(file_path, before_content, after_content)
    return results


def summarize_validation(results: dict[str, ValidationResult]) -> str:
    """Human-readable summary of validation results."""
    failed = [fp for fp, r in results.items() if not r.passed]
    warned = [fp for fp, r in results.items() if r.passed and r.warnings]
    passed = [fp for fp, r in results.items() if r.passed and not r.warnings]

    lines = [f"Validation: {len(passed)} passed, {len(warned)} warnings, {len(failed)} failed"]
    for fp in failed:
        lines.append(f"  ❌ {fp}: {'; '.join(results[fp].errors)}")
    for fp in warned:
        lines.append(f"  ⚠ {fp}: {'; '.join(results[fp].warnings)}")
    return "\n".join(lines)
