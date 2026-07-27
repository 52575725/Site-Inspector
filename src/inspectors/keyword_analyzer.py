from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.ai.deepseek_client import DeepSeekClient
from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# Common English stopwords to filter from keyword extraction
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "this", "that",
    "these", "those", "it", "its", "we", "you", "they", "he", "she",
    "not", "no", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "now", "how", "when", "where", "which", "who",
    "whom", "what", "why", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "over", "out", "off",
    "up", "down", "then", "here", "there", "if", "as", "while", "until",
    "because", "since", "although", "though", "whether", "without",
}

AI_KEYWORD_SYSTEM = """\
You are an SEO keyword strategist. Extract the key topics and phrases a page
should rank for, based on its actual content.

Rules:
- Identify 3-5 primary keywords (specific, commercially relevant phrases)
- Identify 2-4 secondary keywords (related terms, long-tail variations)
- Determine the dominant search intent
- Evaluate whether the page's keywords match its apparent purpose
- Return ONLY valid JSON."""

AI_KEYWORD_PROMPT = """\
Page URL: {url}
Title: {title}
H1: {h1}
Meta description: {meta_description}
Content excerpt (first {max_chars} chars):
{content_excerpt}

Return JSON:
{{
  "primary_keywords": ["phrase 1", "phrase 2", "phrase 3"],
  "secondary_keywords": ["phrase 4", "phrase 5"],
  "search_intent": "informational|commercial|transactional|navigational",
  "page_purpose": "what this page appears to be about",
  "keyword_title_match": 0-10,
  "topic_focus": "narrow|balanced|scattered",
  "suggested_title": "SEO-optimized title with primary keyword",
  "suggested_meta_description": "Compelling 120-160 char meta description",
  "missing_semantic_keywords": ["keyword phrase the page should target but doesn't"],
  "issues": ["any keyword strategy issues found"]
}}"""


class KeywordAnalyzer(BaseInspector):
    """Analyzes page content for keyword usage and SEO optimization.

    Two modes:
    - Statistical (default): frequency-based keyword extraction, density checks
    - AI (when deepseek is configured): semantic keyword extraction,
      intent analysis, missing keyword detection, topic focus assessment
    """

    inspector_name = "keyword_analyzer"

    MIN_WORDS_FOR_ANALYSIS = 50
    MAX_BODY_CHARS = 4000

    def __init__(self, deepseek: DeepSeekClient | None = None):
        super().__init__()
        self.deepseek = deepseek
        self._deepseek_available: bool | None = None

    async def setup(self) -> None:
        self._deepseek_available = None

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.find("body")
        if not body:
            return findings

        # Strip boilerplate
        for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        visible_text = body.get_text(separator=" ", strip=True)
        words = [w.lower().strip(".,!?;:()[]{}\"'") for w in visible_text.split()]
        words = [w for w in words if w and w not in STOPWORDS and len(w) > 2]

        if len(words) < self.MIN_WORDS_FOR_ANALYSIS:
            return findings

        # ── AI-powered analysis ────────────────────────────────────────
        if self.deepseek is not None:
            if self._deepseek_available is None:
                try:
                    self._deepseek_available = await self.deepseek.health_check()
                except Exception:
                    self._deepseek_available = False

            if self._deepseek_available:
                ai_findings = await self._ai_keyword_analysis(
                    url, soup, visible_text, words,
                )
                findings.extend(ai_findings)
                # Continue with statistical checks even if AI produced findings —
                # AI catches semantic issues but may miss structural problems
                # like keyword-not-in-URL or density issues.

        # ── Statistical fallback ───────────────────────────────────────

        word_freq = Counter(words)
        total_words = len(words)

        # Extract key phrases (bigrams)
        bigrams = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words) - 1)
            if len(words[i]) > 3 and len(words[i+1]) > 3
            and words[i] not in STOPWORDS
            and words[i+1] not in STOPWORDS
        ]
        bigram_freq = Counter(bigrams)

        # Target keywords from content frequency
        target_keywords = [
            w for w, c in word_freq.most_common(10)
            if c >= 3 and len(w) > 3
        ][:5]
        target_phrases = [
            p for p, c in bigram_freq.most_common(8)
            if c >= 2
        ][:3]
        all_targets = target_keywords + target_phrases
        if not all_targets:
            return findings
        all_targets = all_targets[:5]

        # Density check
        for kw in all_targets[:3]:
            count = word_freq.get(kw, 0)
            density = (count / total_words) * 100
            if density < 0.5:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="keyword_density_low",
                    description=(
                        f"Keyword '{kw}' density is {density:.1f}% "
                        f"(appears {count} times in {total_words} words). "
                        f"Target: 1-3% density for primary keywords."
                    ),
                    current_value=f"{density:.1f}%",
                    suggested_value="1.0-3.0%",
                    raw_metadata={
                        "keyword": kw, "count": count,
                        "total_words": total_words,
                        "density": round(density, 2),
                    },
                ))

        # Positional checks
        title_tag = soup.find("title")
        title_text = title_tag.get_text(strip=True).lower() if title_tag else ""
        h1_tag = soup.find("h1")
        h1_text = h1_tag.get_text(strip=True).lower() if h1_tag else ""
        first_p = body.find("p")
        first_p_text = first_p.get_text(strip=True).lower() if first_p else ""
        url_path = urlparse(url).path.lower()

        for kw in all_targets[:3]:
            kw_lower = kw.lower()
            if title_text and kw_lower not in title_text:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="keyword_not_in_title",
                    description=f"Primary keyword '{kw}' not found in page title.",
                    current_value=title_text[:200],
                    suggested_value=f"Include '{kw}' in title",
                    raw_metadata={"keyword": kw},
                ))
            if h1_text and kw_lower not in h1_text:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="keyword_not_in_h1",
                    description=f"Primary keyword '{kw}' not found in H1 heading.",
                    current_value=h1_text[:200],
                    suggested_value=f"Include '{kw}' in H1",
                    raw_metadata={"keyword": kw},
                ))
            if first_p_text and kw_lower not in first_p_text:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="keyword_not_in_first_paragraph",
                    description=f"Primary keyword '{kw}' not found in opening paragraph.",
                    current_value=first_p_text[:200],
                    suggested_value=f"Include '{kw}' in first paragraph",
                    raw_metadata={"keyword": kw},
                ))
            if kw_lower not in url_path:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="keyword_not_in_url",
                    description=f"No target keyword found in URL path '{url_path}'.",
                    current_value=url_path,
                    suggested_value=f"Include keywords like '{all_targets[0]}' in URL slug",
                    raw_metadata={"keywords": all_targets[:3]},
                ))
                break  # Only report once for URL

        return findings

    # ── AI mode ────────────────────────────────────────────────────

    async def _ai_keyword_analysis(
        self, url: str, soup: BeautifulSoup, text: str, words: list[str],
    ) -> list[RawFinding]:
        """Use DeepSeek for semantic keyword analysis."""
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        h1_tag = soup.find("h1")
        h1 = h1_tag.get_text(strip=True) if h1_tag else ""

        desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = desc_tag.get("content", "").strip() if desc_tag else ""

        try:
            result = await self.deepseek.generate_json(
                prompt=AI_KEYWORD_PROMPT.format(
                    url=url, title=title, h1=h1,
                    meta_description=meta_desc,
                    max_chars=str(self.MAX_BODY_CHARS),
                    content_excerpt=text[:self.MAX_BODY_CHARS],
                ),
                system=AI_KEYWORD_SYSTEM,
                temperature=0.3,
                max_tokens=1200,
            )
        except Exception as e:
            logger.debug(f"AI keyword analysis failed for {url}: {e}")
            return []

        if result.get("error") or result.get("raw"):
            return []

        findings: list[RawFinding] = []

        # Check scattered focus
        if result.get("topic_focus") == "scattered":
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="keyword_scattered_focus",
                description=(
                    f"Page topic focus is scattered — content touches on too many "
                    f"unrelated keywords. Focus on a primary topic cluster."
                ),
                current_value="scattered focus",
                suggested_value=(
                    f"Primary keywords: {result.get('primary_keywords', [])}. "
                    f"Tighten content around one core topic."
                ),
                raw_metadata=result,
            ))

        # Title-keyword mismatch
        kw_title_match = result.get("keyword_title_match", 10)
        if isinstance(kw_title_match, (int, float)) and kw_title_match <= 5:
            primaries = result.get("primary_keywords", [])
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="keyword_title_mismatch",
                description=(
                    f"Title doesn't adequately reflect target keywords "
                    f"(match: {kw_title_match}/10). Suggested title: "
                    f"'{result.get('suggested_title', '')}'"
                ),
                current_value=title[:200],
                suggested_value=result.get("suggested_title", ""),
                raw_metadata=result,
            ))

        # Missing semantic keywords
        missing = result.get("missing_semantic_keywords", [])
        if missing:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="keyword_missing_semantic",
                description=(
                    f"Page is missing these semantically relevant keywords: "
                    f"{', '.join(missing[:5])}. Including these can improve "
                    f"topical authority."
                ),
                current_value=f"missing {len(missing)} relevant terms",
                suggested_value=f"Add content covering: {', '.join(missing[:5])}",
                raw_metadata=result,
            ))

        # Intent clarity
        intent = result.get("search_intent", "")
        if intent and intent in ("informational", "commercial", "transactional"):
            title_tag = soup.find("title")
            h1_tag = soup.find("h1")
            # Check if the page has conversion elements when intent is transactional
            if intent == "transactional":
                has_cta = soup.find(
                    "a", href=re.compile(r"(buy|order|shop|purchase|quote|contact)", re.I)
                ) or soup.find(
                    "button", string=re.compile(r"(buy|order|shop|add.to.cart|get.quote)", re.I)
                )
                if not has_cta:
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="keyword_intent_no_cta",
                        description=(
                            f"Page has transactional intent keywords but lacks clear "
                            f"calls-to-action (purchase buttons, quote forms, etc.)."
                        ),
                        current_value="no CTA found",
                        suggested_value="Add clear CTAs matching transactional intent",
                        raw_metadata=result,
                    ))

        # Suggested improvements
        issues = result.get("issues", [])
        if issues and len(findings) == 0:
            # General issues that don't fit other categories
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="keyword_strategy_issues",
                description=f"Keyword strategy issues: {'; '.join(issues[:3])}",
                current_value="issues found",
                suggested_value=result.get("suggested_meta_description", ""),
                raw_metadata=result,
            ))

        return findings
