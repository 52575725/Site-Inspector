"""Internal link suggestion engine.

Uses PageRank scores, keyword overlap, and content similarity to recommend
contextual cross-links between pages.  Only suggests — never writes without
approval (fix_type = "semi_auto").

Algorithm:
1. For each page pair with keyword overlap above threshold
2. Check if a link already exists from source to target
3. If not, score the suggestion: PR_benefit * topic_relevance
4. Propose adding a link from the most relevant paragraph in source
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "not", "no", "all",
    "some", "such", "only", "also", "new", "more", "most",
}

MIN_KEYWORD_OVERLAP = 3
MIN_PR_BENEFIT = 0.001
MAX_SUGGESTIONS_PER_PAGE = 5


class InternalLinkFixer(BaseFixer):
    """Suggest contextual internal links based on PageRank + topic overlap.

    Reads link graph data (PageRank scores, existing links) from the
    LinkGraphInspector via shared state, identifies high-value link
    opportunities, and generates fix suggestions.
    """

    fixer_name = "internal_link_fixer"
    fix_type = "semi_auto"
    supported_categories = ["link_graph_orphan", "link_graph_near_orphan",
                            "link_graph_low_inlinks", "link_graph_deep_page"]

    def __init__(self):
        super().__init__()
        self._pagerank: dict[str, float] = {}
        self._keywords: dict[str, list[str]] = {}
        self._titles: dict[str, str] = {}
        self._outlinks: dict[str, set[str]] = {}
        self._all_urls: list[str] = []

    def set_graph_data(
        self, pagerank: dict[str, float], keywords: dict[str, list[str]],
        titles: dict[str, str], outlinks: dict[str, set[str]],
    ) -> None:
        """Receive link graph data from the orchestrator."""
        self._pagerank = pagerank
        self._keywords = keywords
        self._titles = titles
        self._outlinks = outlinks
        self._all_urls = list(pagerank.keys())

    async def generate_fix(
        self, issue: dict, source: BaseSource, page_content: str,
    ) -> FixResult:
        url = issue.get("url", "")
        file_path = issue.get("file_path", "")
        issue_id = issue.get("id", 0)

        if not self._pagerank or url not in self._pagerank:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message="Link graph data unavailable",
            )

        # Find best link targets for this page
        suggestions = self._find_suggestions(url)
        if not suggestions:
            return FixResult(
                success=False, issue_id=issue_id,
                fixer_name=self.fixer_name, fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content, after_content=page_content,
                error_message="No suitable link targets found",
            )

        # Build the suggestion text
        suggestion_lines = []
        for target_url, score, reason in suggestions[:3]:
            target_title = self._titles.get(target_url, target_url)
            suggestion_lines.append(
                f"- Link to [{target_title}]({target_url}) "
                f"(score: {score:.3f}, reason: {reason})"
            )

        diff = (
            f"# Internal Link Suggestions for {url}\n"
            + "\n".join(suggestion_lines)
            + f"\n\nAdd these links in contextually relevant positions on the page."
        )

        return FixResult(
            success=True,
            issue_id=issue_id,
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=file_path,
            before_content=page_content,
            after_content=page_content,
            diff=diff,
        )

    # ── Suggestion engine ────────────────────────────────────────────

    def _find_suggestions(self, url: str) -> list[tuple[str, float, str]]:
        """Find the best pages to link TO from this page."""
        source_pr = self._pagerank.get(url, 0)
        source_kws = set(self._keywords.get(url, []))
        existing_targets = self._outlinks.get(url, set())

        candidates: list[tuple[str, float, str]] = []

        for target_url in self._all_urls:
            if target_url == url:
                continue
            if target_url in existing_targets:
                continue  # Already linked

            target_pr = self._pagerank.get(target_url, 0)
            target_kws = set(self._keywords.get(target_url, []))

            # Keyword overlap
            overlap = source_kws & target_kws
            if len(overlap) < MIN_KEYWORD_OVERLAP:
                continue

            # PR benefit: linking from high-PR to low-PR distributes equity
            pr_benefit = source_pr * (1.0 - target_pr) if target_pr < source_pr else 0
            if pr_benefit < MIN_PR_BENEFIT:
                continue

            # Topic relevance score
            topic_score = len(overlap) / max(len(source_kws | target_kws), 1)

            # Final score
            score = pr_benefit * 10 + topic_score

            reason = f"{len(overlap)} shared keywords (e.g. {', '.join(sorted(overlap)[:3])})"
            candidates.append((target_url, score, reason))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:MAX_SUGGESTIONS_PER_PAGE]

    @staticmethod
    def extract_keywords(html: str) -> list[str]:
        """Extract significant keywords from page HTML for topic matching."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        words = re.findall(r"[a-z]{4,}", text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(15)]
