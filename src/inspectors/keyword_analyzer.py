from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urlparse

from bs4 import BeautifulSoup

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


class KeywordAnalyzer(BaseInspector):
    """Analyzes page content for keyword usage and SEO optimization.

    Detects: keyword_missing, keyword_density_low, keyword_not_in_title,
    keyword_not_in_h1, keyword_not_in_first_paragraph.
    """

    inspector_name = "keyword_analyzer"

    # Minimum word count required for keyword analysis
    MIN_WORDS_FOR_ANALYSIS = 50

    def __init__(self):
        super().__init__()

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract text content
        body = soup.find("body")
        if not body:
            return findings

        # Remove script, style, nav, footer for content analysis
        for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        visible_text = body.get_text(separator=" ", strip=True)
        words = [w.lower().strip(".,!?;:()[]{}\"'") for w in visible_text.split()]
        words = [w for w in words if w and w not in STOPWORDS and len(w) > 2]

        if len(words) < self.MIN_WORDS_FOR_ANALYSIS:
            return findings

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

        # Determine target keywords from content
        # Top single keywords (by frequency, min 3 occurrences)
        target_keywords = [
            w for w, c in word_freq.most_common(10)
            if c >= 3 and len(w) > 3
        ][:5]

        # Top bigrams (min 2 occurrences)
        target_phrases = [
            p for p, c in bigram_freq.most_common(8)
            if c >= 2
        ][:3]

        all_targets = target_keywords + target_phrases
        if not all_targets:
            return findings
        all_targets = all_targets[:5]

        # Check 1: keyword density
        for kw in all_targets[:3]:
            count = word_freq.get(kw, 0)
            density = (count / total_words) * 100
            if density < 0.5:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="keyword_density_low",
                    description=(
                        f"Keyword '{kw}' density is {density:.1f}% "
                        f"(appears {count} times in {total_words} words). "
                        f"Target: 1-3% density for primary keywords."
                    ),
                    current_value=f"{density:.1f}%",
                    suggested_value="1.0-3.0%",
                    raw_metadata={
                        "keyword": kw,
                        "count": count,
                        "total_words": total_words,
                        "density": round(density, 2),
                    },
                ))

        # Check 2: keyword in title
        title_tag = soup.find("title")
        title_text = title_tag.get_text(strip=True).lower() if title_tag else ""
        h1_tag = soup.find("h1")
        h1_text = h1_tag.get_text(strip=True).lower() if h1_tag else ""

        # Check 3: keyword in first paragraph
        first_p = body.find("p")
        first_p_text = first_p.get_text(strip=True).lower() if first_p else ""

        for kw in all_targets[:3]:
            kw_lower = kw.lower()

            if title_text and kw_lower not in title_text:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="keyword_not_in_title",
                    description=(
                        f"Primary keyword '{kw}' not found in page title. "
                        f"Including target keywords in <title> improves "
                        f"search engine ranking for those terms."
                    ),
                    current_value=title_text[:200],
                    suggested_value=f"Include '{kw}' in title",
                    raw_metadata={"keyword": kw},
                ))

            if h1_text and kw_lower not in h1_text:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="keyword_not_in_h1",
                    description=(
                        f"Primary keyword '{kw}' not found in H1 heading. "
                        f"H1 is the most important on-page SEO element after title."
                    ),
                    current_value=h1_text[:200],
                    suggested_value=f"Include '{kw}' in H1",
                    raw_metadata={"keyword": kw},
                ))

            if first_p_text and kw_lower not in first_p_text:
                findings.append(RawFinding(
                    url=url,
                    inspector=self.inspector_name,
                    category="keyword_not_in_first_paragraph",
                    description=(
                        f"Primary keyword '{kw}' not found in opening paragraph. "
                        f"Early keyword placement signals topical relevance to search engines."
                    ),
                    current_value=first_p_text[:200],
                    suggested_value=f"Include '{kw}' in first paragraph",
                    raw_metadata={"keyword": kw},
                ))

        # Check 4: URL contains keyword
        url_path = urlparse(url).path.lower()
        url_has_keyword = any(kw.lower() in url_path for kw in all_targets[:3])
        if not url_has_keyword and all_targets:
            findings.append(RawFinding(
                url=url,
                inspector=self.inspector_name,
                category="keyword_not_in_url",
                description=(
                    f"No target keyword found in URL path '{url_path}'. "
                    f"Keywords in URLs are a ranking factor."
                ),
                current_value=url_path,
                suggested_value=f"Include keywords like '{all_targets[0]}' in URL slug",
                raw_metadata={"keywords": all_targets[:3]},
            ))

        return findings
