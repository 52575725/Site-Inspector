from __future__ import annotations

import re
from difflib import SequenceMatcher

from bs4 import BeautifulSoup

from src.agents.citation_agent import ArticleCitationAgent
from src.agents.models import ImagePlan, QualityCheck, QualityReport


class ArticleQualityAgent:
    """Run content checks and joint article-image checks between agent stages."""

    def __init__(self) -> None:
        self.citation_agent = ArticleCitationAgent()

    def inspect_content(
        self,
        html: str,
        *,
        research_report: dict | None = None,
        expected_word_count: int | None = None,
    ) -> QualityReport:
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article") or soup.find("main") or soup.body or soup
        text = article.get_text(" ", strip=True)
        h1_count = len(article.find_all("h1"))
        headings = [tag.get_text(" ", strip=True) for tag in article.find_all("h2")]
        normalized = [heading.casefold() for heading in headings if heading]
        word_count = self._word_count(text)
        minimum_words = max(250, int(expected_word_count * 0.85)) if expected_word_count else 250
        paragraphs = [
            node.get_text(" ", strip=True)
            for node in article.find_all("p")
            if self._word_count(node.get_text(" ", strip=True)) >= 18
        ]
        duplicate_pairs = self._duplicate_paragraph_pairs(paragraphs)
        originality_article = BeautifulSoup(str(article), "html.parser")
        for quoted in originality_article.find_all(["blockquote", "q"]):
            quoted.decompose()
        originality_text = originality_article.get_text(" ", strip=True)
        source_overlap = self._source_overlap(originality_text, research_report or {})
        template_phrases = self._template_phrases(text)
        substantive_sections, total_sections = self._substantive_sections(article)
        section_ratio = substantive_sections / total_sections if total_sections else 0.0
        checks = [
            QualityCheck(
                name="article_content",
                passed=word_count >= minimum_words,
                severity="error",
                message=(
                    f"Article contains {word_count} words/characters; minimum is {minimum_words}."
                ),
            ),
            QualityCheck(
                name="single_h1",
                passed=h1_count == 1,
                severity="error",
                message=f"Found {h1_count} H1 heading(s); exactly one is recommended.",
            ),
            QualityCheck(
                name="section_structure",
                passed=len(headings) >= 2,
                severity="error",
                message=f"Found {len(headings)} H2 sections.",
            ),
            QualityCheck(
                name="unique_sections",
                passed=len(normalized) == len(set(normalized)),
                severity="error",
                message="Section headings are unique." if len(normalized) == len(set(normalized)) else "Duplicate section headings were detected.",
            ),
            QualityCheck(
                name="paragraph_redundancy",
                passed=not duplicate_pairs,
                severity="error",
                message=(
                    "Paragraphs make distinct contributions."
                    if not duplicate_pairs
                    else f"Found {len(duplicate_pairs)} substantially repetitive paragraph pair(s)."
                ),
            ),
            QualityCheck(
                name="source_overlap",
                passed=source_overlap["max_match_tokens"] < 16,
                severity="error",
                message=(
                    "No long verbatim overlap with reference articles was detected."
                    if source_overlap["max_match_tokens"] < 16
                    else (
                        "Draft contains a long unquoted sequence also present in research source "
                        f"{source_overlap['source_url']}."
                    )
                ),
            ),
            QualityCheck(
                name="template_language",
                passed=len(template_phrases) < 3,
                severity="error",
                message=(
                    "Template-like filler is limited."
                    if len(template_phrases) < 3
                    else f"Found {len(template_phrases)} template-like filler phrases."
                ),
            ),
            QualityCheck(
                name="section_depth",
                passed=not total_sections or section_ratio >= 0.6,
                severity="warning",
                message=(
                    f"{substantive_sections} of {total_sections} H2 sections contain substantive detail."
                ),
            ),
        ]
        citation_report = self.citation_agent.inspect(html, research_report)
        checks.extend(citation_report.checks)
        return QualityReport(
            passed=not any(not check.passed and check.severity == "error" for check in checks),
            checks=checks,
            metrics={
                "characters": len(text),
                "word_count": word_count,
                "minimum_word_count": minimum_words,
                "h1_count": h1_count,
                "h2_count": len(headings),
                "duplicate_paragraph_pairs": len(duplicate_pairs),
                "source_max_match_tokens": source_overlap["max_match_tokens"],
                "template_phrase_count": len(template_phrases),
                "substantive_section_ratio": round(section_ratio, 3),
                **citation_report.metrics,
            },
        )

    def inspect_article_with_images(
        self,
        html: str,
        plan: ImagePlan,
        *,
        research_report: dict | None = None,
        expected_word_count: int | None = None,
    ) -> QualityReport:
        content_report = self.inspect_content(
            html,
            research_report=research_report,
            expected_word_count=expected_word_count,
        )
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article") or soup.find("main") or soup.body or soup
        figures = article.select("figure.article-media")
        image_sources = [str(image.get("src", "")) for image in article.find_all("img")]
        complete_figures = all(
            figure.find("img", src=True)
            and figure.find("img").get("alt", "").strip()
            and figure.find("figcaption")
            for figure in figures
        )
        relevant_figures = sum(self._figure_matches_nearby_section(figure) for figure in figures)
        relevance_passed = not figures or relevant_figures == len(figures)
        checks = [
            *content_report.checks,
            QualityCheck(
                name="image_target",
                passed=len(image_sources) >= plan.target_count,
                severity="error",
                message=f"Found {len(image_sources)} images; target is {plan.target_count}.",
            ),
            QualityCheck(
                name="image_metadata",
                passed=complete_figures,
                severity="error",
                message="Every inserted image has alt text and a caption." if complete_figures else "An inserted image is missing alt text or a caption.",
            ),
            QualityCheck(
                name="unique_images",
                passed=len(image_sources) == len(set(image_sources)),
                severity="error",
                message="Image sources are unique." if len(image_sources) == len(set(image_sources)) else "Duplicate image sources were detected.",
            ),
            QualityCheck(
                name="image_section_relevance",
                passed=relevance_passed,
                severity="error",
                message=f"{relevant_figures} of {len(figures)} images are placed in their planned section.",
            ),
        ]
        return QualityReport(
            passed=not any(not check.passed and check.severity == "error" for check in checks),
            checks=checks,
            metrics={
                **content_report.metrics,
                "figure_count": len(figures),
                "image_count": len(image_sources),
                "section_relevant_images": relevant_figures,
            },
        )

    @staticmethod
    def _figure_matches_nearby_section(figure) -> bool:
        heading = figure.find_previous(["h1", "h2", "h3"])
        target_heading = " ".join(str(figure.get("data-target-heading", "")).split())
        nearby_heading = heading.get_text(" ", strip=True) if heading else ""
        if target_heading:
            if target_heading.casefold() == nearby_heading.casefold():
                return True
        try:
            target_index = int(figure.get("data-target-section-index", -1))
        except (TypeError, ValueError):
            target_index = -1
        if heading and heading.name == "h2" and target_index >= 0:
            article = figure.find_parent(["article", "main", "body"])
            headings = article.find_all("h2") if article else []
            return target_index < len(headings) and headings[target_index] is heading
        image = figure.find("img")
        caption = figure.find("figcaption")
        section_text = heading.get_text(" ", strip=True) if heading else ""
        image_text = " ".join([
            str(image.get("alt", "")) if image else "",
            caption.get_text(" ", strip=True) if caption else "",
        ])
        stop_words = {
            "about", "article", "guide", "image", "photo", "scene", "the", "this",
            "with", "from", "into", "your", "their", "relevant",
        }
        tokenize = lambda value: {
            token for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) >= 4 and token not in stop_words
        }
        section_tokens = tokenize(section_text)
        return bool(section_tokens and section_tokens & tokenize(image_text))

    @staticmethod
    def _word_count(text: str) -> int:
        latin_words = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text)
        cjk_characters = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
        return len(latin_words) + len(cjk_characters)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
        ]

    @classmethod
    def _duplicate_paragraph_pairs(cls, paragraphs: list[str]) -> list[tuple[int, int]]:
        token_rows = [cls._tokens(paragraph) for paragraph in paragraphs]
        duplicates: list[tuple[int, int]] = []
        for first in range(len(token_rows)):
            for second in range(first + 1, len(token_rows)):
                left, right = token_rows[first], token_rows[second]
                if min(len(left), len(right)) < 18:
                    continue
                ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
                if ratio >= 0.86:
                    duplicates.append((first, second))
        return duplicates

    @classmethod
    def _source_overlap(cls, text: str, report: dict) -> dict[str, int | str]:
        draft_tokens = cls._tokens(text)
        best_size = 0
        best_url = ""
        for reference in report.get("references", []):
            if not isinstance(reference, dict):
                continue
            source_tokens = cls._tokens(str(reference.get("similarity_text", "")))
            if len(source_tokens) < 16 or len(draft_tokens) < 16:
                continue
            match = SequenceMatcher(None, draft_tokens, source_tokens, autojunk=False).find_longest_match()
            if match.size > best_size:
                best_size = match.size
                best_url = str(reference.get("url", ""))
        return {"max_match_tokens": best_size, "source_url": best_url}

    @staticmethod
    def _template_phrases(text: str) -> list[str]:
        patterns = {
            "in today's rapidly changing": r"\bin today(?:'s|s) rapidly changing\b",
            "in today's fast-paced": r"\bin today(?:'s|s) fast-paced\b",
            "delve into": r"\bdelve into\b",
            "ever-evolving landscape": r"\bever-evolving landscape\b",
            "it is important to note": r"\bit is important to note\b",
            "plays a crucial role": r"\bplays? a crucial role\b",
            "navigate the complexities": r"\bnavigate the complexities\b",
            "comprehensive guide": r"\bthis comprehensive guide\b",
            "当今快速变化": r"在当今快速变化的",
            "至关重要": r"至关重要",
            "深入探讨": r"深入探讨",
        }
        return [label for label, pattern in patterns.items() if re.search(pattern, text, re.I)]

    @classmethod
    def _substantive_sections(cls, article) -> tuple[int, int]:
        headings = article.find_all("h2")
        substantive = 0
        for heading in headings:
            parts: list[str] = []
            for sibling in heading.next_siblings:
                if getattr(sibling, "name", None) == "h2":
                    break
                if hasattr(sibling, "get_text"):
                    parts.append(sibling.get_text(" ", strip=True))
            if cls._word_count(" ".join(parts)) >= 45:
                substantive += 1
        return substantive, len(headings)
