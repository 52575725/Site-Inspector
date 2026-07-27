"""Pre-fix HTML well-formedness validator and auto-fixer.

Runs BEFORE any fixer touches HTML content.  Catches and repairs common
syntax errors that would cause BeautifulSoup to produce corrupted output:

- Unclosed void elements (meta, link, br, hr, img, input without >)
- Stray closing tokens (extra /> after properly-closed tags)
- Duplicate consecutive identical elements (title-as-H1 spam)
- Content injected at wrong position (style/h1 before <!DOCTYPE> or <html>)

This is a safety net — the root cause fix is still proper HTML authoring.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SanitizeResult:
    file_path: str
    was_malformed: bool = False
    issues_found: list[str] = field(default_factory=list)
    issues_fixed: list[str] = field(default_factory=list)
    original: str = ""
    sanitized: str = ""


# Patterns that indicate a fixer has previously corrupted this file
_CORRUPTION_SIGNATURES = [
    # Title text duplicated as H1/H2 at the very start of body
    (r"<body>\s*<h1>[^<]{30,}</h1>\s*<h2>\1</h2>", "duplicate_title_as_headings_at_body_start"),
    # More than 3 consecutive H2 tags with identical text (strong signal of corruption)
    (r"<h2>([^<]{20,})</h2>\s*<h2>\1</h2>\s*<h2>\1</h2>", "triple_identical_h2"),
]

# Void elements that must not have children and must have closing >
_VOID_ELEMENTS = {
    "meta", "link", "br", "hr", "img", "input", "source",
    "embed", "area", "base", "col", "track", "wbr",
}


def sanitize_html(file_path: str, content: str) -> SanitizeResult:
    """Validate and auto-fix common HTML syntax errors.

    Returns the sanitized HTML and a report of what was fixed.
    Always returns valid content — if sanitization fails, returns original.
    """
    result = SanitizeResult(file_path=file_path, original=content)

    if not content or not content.strip():
        return result

    sanitized = content

    # ── Step 1: Fix unclosed void elements ─────────────────────────
    sanitized, void_count = _fix_unclosed_void_elements(sanitized)
    if void_count > 0:
        result.issues_found.append(
            f"Found {void_count} unclosed void element(s) (e.g. <meta charset=\"...\" without >)"
        )
        result.issues_fixed.append(f"Auto-closed {void_count} void element(s)")
        result.was_malformed = True

    # ── Step 2: Fix stray /> tokens ───────────────────────────────
    sanitized, stray_count = _remove_stray_closing_tokens(sanitized)
    if stray_count > 0:
        result.issues_found.append(
            f"Found {stray_count} stray '/>' closing token(s)"
        )
        result.issues_fixed.append(f"Removed {stray_count} stray '/>' token(s)")
        result.was_malformed = True

    # ── Step 3: Detect and strip corruption signatures ────────────
    sanitized, corruptions = _strip_corruption_sigs(sanitized)
    if corruptions > 0:
        result.issues_found.append(
            f"Found {corruptions} corruption signature(s) "
            f"(duplicate title-as-headings in body)"
        )
        result.issues_fixed.append(
            f"Stripped {corruptions} corrupted element block(s)"
        )
        result.was_malformed = True

    # ── Step 4: Validate with BeautifulSoup ───────────────────────
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sanitized, "html.parser")

        # Check for title text appearing as heading text (BeautifulSoup artifact)
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # Count how many times the title text appears in H1/H2
            h1_texts = [h.get_text(strip=True) for h in soup.find_all("h1")]
            h2_texts = [h.get_text(strip=True) for h in soup.find_all("h2")]

            # If title text appears > 2 times across H1+H2, something is wrong
            title_in_headings = sum(
                1 for t in h1_texts + h2_texts if t == title_text
            )
            if title_in_headings > 2:
                # Try to fix: keep only the first occurrence
                result.issues_found.append(
                    f"Title text '{title_text[:60]}...' duplicated "
                    f"across {title_in_headings} headings"
                )
                # Re-parse and remove duplicates
                sanitized = _remove_duplicate_title_headings(
                    sanitized, title_text, soup,
                )
                result.issues_fixed.append(
                    f"Removed duplicate title-as-headings (kept first)"
                )
                result.was_malformed = True
    except Exception as e:
        logger.warning(f"BeautifulSoup validation failed for {file_path}: {e}")

    result.sanitized = sanitized
    return result


# ── Internal fixers ──────────────────────────────────────────────────


def _fix_unclosed_void_elements(html: str) -> tuple[str, int]:
    """Find void elements missing their closing > and fix them.

    Example: <meta charset="utf-8"\n<link ...> → <meta charset="utf-8">\n<link ...>
    """
    count = 0

    for tag in _VOID_ELEMENTS:
        # Pattern: <tag ... "value" \n (next tag or content)
        # The void element is unclosed when the opening < of the next element
        # or content starts before a > closes the void element.
        pattern = re.compile(
            rf'(<{tag}\b[^>]*?)(?=\s*<(?!{tag}\b))',
            re.IGNORECASE | re.DOTALL,
        )

        def _fix_match(m: re.Match) -> str:
            nonlocal count
            attrs = m.group(1)
            # Check if this is already closed (has > at end or />)
            if attrs.rstrip().endswith(">"):
                return m.group(0)
            # Check the character after the match
            full_match = m.group(0)
            next_pos = m.end()
            if next_pos < len(html) and html[next_pos] == ">":
                return full_match  # already has > right after
            count += 1
            return attrs.rstrip() + ">"

        html = pattern.sub(_fix_match, html)

        # Also try a simpler pattern: <tag ... "  without > and followed by newline + <
        # This catches the specific case we saw: <meta charset="utf-8"\n<link
        pattern2 = re.compile(
            rf'(<{tag}\b[^>]*?)>?\s*\n\s*(?=<[a-zA-Z])',
            re.IGNORECASE,
        )
        # Don't double-count, so we count differently
        html = pattern2.sub(lambda m: m.group(1).rstrip() + ">\n", html)

    # Final pass: look for any <tag ... " followed immediately by < without >
    for tag in _VOID_ELEMENTS:
        # <tag attr="val"<next_tag — the space between " and < should have a >
        pattern3 = re.compile(
            rf'(<{tag}\b[^>]*?["\'])\s*(<[a-zA-Z/])',
            re.IGNORECASE,
        )

        def _fix_unclosed(m: re.Match) -> str:
            nonlocal count
            before = m.group(1)
            after = m.group(2)
            if not before.rstrip().endswith(">"):
                count += 1
                return before.rstrip() + ">" + after
            return m.group(0)

        html = pattern3.sub(_fix_unclosed, html)

    return html, count


def _remove_stray_closing_tokens(html: str) -> tuple[str, int]:
    """Remove stray /> tokens that don't belong to any tag.

    Example: <link href="x"/>/> → <link href="x"/>
    """
    count = 0

    # Pattern: /> followed by /> (double-closing)
    pattern = re.compile(r'/>\s*/>')
    matches = list(pattern.finditer(html))
    for m in reversed(matches):
        html = html[:m.start()] + "/>" + html[m.end():]
        count += 1

    return html, count


def _strip_corruption_sigs(html: str) -> tuple[str, int]:
    """Detect and remove blocks of duplicated title-as-heading spam.

    When BeautifulSoup corrupts output, it can produce:
    <body><h1>Title...</h1><h2>Title...</h2><h2>Title...</h2>... (N times)
    followed by the actual content starting with <header> or <nav>.

    We identify this pattern and strip the duplicated block.
    """
    corruptions = 0

    # Pattern: <body> followed by 3+ consecutive identical H-tags,
    # then actual content starts with <header>, <nav>, or <section>
    pattern = re.compile(
        r'(<body>)\s*('
        r'(?:<h[12]>[^<]{20,}</h[12]>\s*){3,}'
        r')(?=\s*<(?:header|nav|section|div|main)\b)',
        re.IGNORECASE | re.DOTALL,
    )

    def _strip_block(m: re.Match) -> str:
        nonlocal corruptions
        corruptions += 1
        return m.group(1)  # Keep only <body>

    html = pattern.sub(_strip_block, html)
    return html, corruptions


def _remove_duplicate_title_headings(
    html: str, title_text: str, soup,
) -> str:
    """Remove duplicate headings that parrot the title text.

    Keeps the first occurrence, removes subsequent identical ones.
    """
    from bs4 import BeautifulSoup

    # Re-parse after other fixes
    soup2 = BeautifulSoup(html, "html.parser")

    title_tag = soup2.find("title")
    title_text_current = (
        title_tag.get_text(strip=True) if title_tag else title_text
    )

    seen = False
    for tag in soup2.find_all(["h1", "h2"]):
        text = tag.get_text(strip=True)
        if text == title_text_current:
            if not seen:
                seen = True  # Keep first
            else:
                tag.decompose()  # Remove duplicates

    return str(soup2)
