"""AI-powered semantic content analysis.

Goes beyond word counts and readability scores to check whether the
page body actually delivers on what the title/H1 promises.  Uses DeepSeek
(configured via SI_DEEPSEEK_API_KEY) with graceful fallback when unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.deepseek_client import DeepSeekClient
from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Minimum word count to trigger semantic analysis (below this, thin-content
# checks from ContentQualityInspector are sufficient).
MIN_WORDS_FOR_SEMANTIC = 150

# Maximum body characters sent to the model (DeepSeek context is large, but
# we keep prompts lean for speed and cost).
MAX_BODY_CHARS = 5000

SYSTEM_PROMPT = """\
You are a senior SEO content auditor.  Critically evaluate whether a page's
body content actually fulfills the promise made by its title and H1 heading.

Rules:
- Be critical — it is worse to miss a content problem than to flag something that is fine.
- Look for: off-topic tangents, keyword-stuffed but shallow sections, missing logical progression, filler paragraphs.
- Consider the apparent search intent behind the title.
- Return ONLY valid JSON (no markdown fences, no extra text)."""

ANALYSIS_PROMPT = """\
Page URL: {url}
Title: {title}
H1: {h1}
Content word count: {word_count}

--- Body content (first {max_chars} chars) ---
{body_text}

Analyze the content and return a JSON object with these fields:

{{
  "title_body_alignment": <0-10: does the body deliver what the title promises? 10=perfect>,
  "topic_coverage": <0-10: are all important subtopics covered? 10=comprehensive>,
  "logical_structure": <0-10: is the content well-organized with clear progression?>,
  "fluff_ratio": <0.0-1.0: what fraction feels like filler/generic text?>,
  "search_intent_match": <0-10: does content match the intent implied by the title?>,
  "missing_topics": ["..."],
  "weak_sections": ["..."],
  "off_topic_segments": ["..."],
  "strengths": ["..."],
  "overall_assessment": "<one-paragraph summary>",
  "needs_rewrite": <true/false>
}}"""


class SemanticContentInspector(BaseInspector):
    """AI-powered: does the content deliver what the title promises?

    Uses DeepSeek to evaluate title-body alignment, topic coverage,
    logical structure, fluff detection, and search-intent matching.
    Falls back gracefully when no AI client is configured.
    """

    inspector_name = "semantic_content"

    def __init__(self, deepseek: DeepSeekClient | None = None):
        self.deepseek = deepseek
        self._available: bool | None = None

    async def setup(self) -> None:
        self._available = None

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        if self.deepseek is None:
            return findings

        # Lazy health-check (cached across pages in this scan)
        if self._available is None:
            try:
                self._available = await self.deepseek.health_check()
            except Exception:
                self._available = False
                logger.debug("DeepSeek not available, skipping semantic analysis")
        if not self._available:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else ""

        h1_tag = soup.find("h1")
        h1 = h1_tag.get_text(strip=True) if h1_tag else ""

        # Strip boilerplate before extracting body text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body_text = soup.get_text(separator=" ", strip=True)
        word_count = len(body_text.split())

        if word_count < MIN_WORDS_FOR_SEMANTIC:
            return findings

        body_excerpt = body_text[:MAX_BODY_CHARS]

        try:
            result = await self.deepseek.generate_json(
                prompt=ANALYSIS_PROMPT.format(
                    url=url,
                    title=title,
                    h1=h1,
                    word_count=str(word_count),
                    max_chars=str(MAX_BODY_CHARS),
                    body_text=body_excerpt,
                ),
                system=SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=1500,
            )
        except Exception as e:
            logger.warning(f"Semantic analysis failed for {url}: {e}")
            return findings

        if result.get("error") or result.get("raw"):
            logger.debug(f"Semantic analysis returned unparseable result for {url}")
            return findings

        # ── Convert AI assessment to findings ──────────────────────────

        # Title-body misalignment
        alignment = result.get("title_body_alignment", 10)
        if isinstance(alignment, (int, float)) and alignment <= 4:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="semantic_title_mismatch",
                description=(
                    f"Title '{title[:100]}' does not match body content "
                    f"(alignment score: {alignment}/10). "
                    f"{result.get('overall_assessment', '')}"
                ),
                current_value=f"alignment={alignment}/10",
                suggested_value="Rewrite content to deliver on the title's promise",
                raw_metadata=result,
            ))
        elif isinstance(alignment, (int, float)) and alignment <= 6:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="semantic_title_partial_match",
                description=(
                    f"Body content only partially delivers on title promise "
                    f"(alignment: {alignment}/10). "
                    f"Weak sections: {result.get('weak_sections', [])}"
                ),
                current_value=f"alignment={alignment}/10",
                suggested_value="Deepen sections that address the core topic",
                raw_metadata=result,
            ))

        # Poor topic coverage
        coverage = result.get("topic_coverage", 10)
        if isinstance(coverage, (int, float)) and coverage <= 4:
            missing = result.get("missing_topics", [])
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="semantic_topic_gap",
                description=(
                    f"Page misses key subtopics (coverage: {coverage}/10). "
                    f"Missing: {', '.join(missing[:5])}"
                ),
                current_value=f"coverage={coverage}/10",
                suggested_value=f"Add sections covering: {', '.join(missing[:5])}",
                raw_metadata=result,
            ))

        # High fluff ratio
        fluff = result.get("fluff_ratio", 0)
        if isinstance(fluff, (int, float)) and fluff > 0.4:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="semantic_high_fluff",
                description=(
                    f"Content is {fluff:.0%} filler/generic text — "
                    f"needs more substantive information"
                ),
                current_value=f"fluff={fluff:.0%}",
                suggested_value="Replace generic paragraphs with data, examples, and specifics",
                raw_metadata=result,
            ))

        # Search intent mismatch
        intent = result.get("search_intent_match", 10)
        if isinstance(intent, (int, float)) and intent <= 5:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="semantic_intent_mismatch",
                description=(
                    f"Content structure doesn't match the implied search intent "
                    f"(score: {intent}/10). {result.get('overall_assessment', '')}"
                ),
                current_value=f"intent_match={intent}/10",
                suggested_value="Align content format with user intent",
                raw_metadata=result,
            ))

        # Needs full rewrite
        if result.get("needs_rewrite") and alignment <= 5:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="semantic_needs_rewrite",
                description=(
                    f"AI assessment recommends full content rewrite. "
                    f"Strengths: {result.get('strengths', [])}. "
                    f"Assessment: {result.get('overall_assessment', '')}"
                ),
                current_value="content needs rewrite",
                suggested_value="Restructure and rewrite content from outline",
                raw_metadata=result,
            ))

        return findings
