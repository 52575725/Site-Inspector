from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.ollama_client import OllamaClient
from src.ai.prompt_manager import PromptManager
from src.ai.text_quality import (
    assess_readability,
    detect_duplicate_risk,
    detect_duplicate_risk_simhash,
)
from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)


class ContentQualityInspector(BaseInspector):
    """Inspect content quality: readability, duplicates, thin pages."""

    inspector_name = "content_quality"

    def __init__(self, ollama: Optional[OllamaClient] = None,
                 prompt_manager: Optional[PromptManager] = None):
        self.ollama = ollama
        self.prompts = prompt_manager or PromptManager()
        self._all_page_texts: list[str] = []
        self._ollama_healthy: bool | None = None

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    def set_all_texts(self, texts: list[str]) -> None:
        """Pre-load all page texts for duplicate detection across the scan."""
        self._all_page_texts = texts

    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script/style content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        word_count = len(text.split())

        # Get page metadata
        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else ""

        # Detect language
        language = self._detect_language_from_content(text, url)

        # Check thin content
        if word_count < 300:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="thin_content",
                description=f"Page has thin content ({word_count} words, minimum 300 recommended)",
                current_value=str(word_count),
                raw_metadata={"word_count": word_count},
            ))

        # Count editorial images inside the article, excluding global header/footer media.
        content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"article|post-content|entry-content", re.I))
            or soup.body
            or soup
        )
        content_word_count = len(content.get_text(" ", strip=True).split())
        image_count = len(content.find_all("img"))
        is_blog = "/blog/" in url.lower() or "/insights/" in url.lower()
        if is_blog and content_word_count >= 200 and image_count < 3:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="article_no_images" if image_count == 0 else "article_image_shortage",
                description=(
                    f"Blog article has {image_count} content images ({content_word_count} words). "
                    "Add enough relevant images to reach 3-4, subject to editorial review."
                ),
                current_value=f"{image_count} images",
                suggested_value=(
                    "Search licensed sources first and add 3-4 relevant images "
                    "with attribution, alt text, captions, and section-aware placement."
                ),
                raw_metadata={"word_count": content_word_count, "image_count": image_count},
            ))

        # Readability assessment
        if word_count >= 50:
            readability = assess_readability(text, language)
            if readability.get("is_thin"):
                # Already reported above
                pass
            elif readability.get("score", 100) < 30:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="low_readability",
                    description=f"Low readability score: {readability['score']}",
                    raw_metadata=readability,
                ))

        # Duplicate content detection — embedding-based if Ollama available
        if self._all_page_texts and len(self._all_page_texts) > 1:
            embed_fn = None
            if self.ollama and self._ollama_healthy is None:
                try:
                    self._ollama_healthy = await self.ollama.health_check()
                except Exception:
                    self._ollama_healthy = False
            if self.ollama and self._ollama_healthy:
                embed_fn = self.ollama.embed

            dup_score = await detect_duplicate_risk(
                text, self._all_page_texts, embed_fn=embed_fn,
            )
            if dup_score > 0.85:
                detection_method = "embedding" if embed_fn else "simhash"
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="duplicate_content",
                    description=(
                        f"Content is {dup_score:.0%} similar to another page "
                        f"(detected via {detection_method})"
                    ),
                    raw_metadata={
                        "similarity": round(dup_score, 3),
                        "method": detection_method,
                    },
                ))

        # Ollama-based nuance analysis (optional, uses cached health check)
        if self.ollama and word_count >= 100 and self._ollama_healthy:
            try:
                system, prompt = self.prompts.build_prompt(
                    "content_quality",
                    url=url,
                    title=title,
                    word_count=str(word_count),
                    content_excerpt=text[:3000],
                    language=language,
                )
                result = await self.ollama.generate_json(
                    prompt=prompt, system=system, temperature=0.3, max_tokens=400,
                )
                ai_score = result.get("quality_score", 0)
                if ai_score < 5:
                    findings.append(RawFinding(
                        url=url, inspector=self.inspector_name,
                        category="low_content_quality_ai",
                        description=f"AI-assessed content quality score: {ai_score}/10. "
                                    f"Issues: {', '.join(result.get('issues', []))}",
                        raw_metadata=result,
                    ))
            except Exception as e:
                logger.debug(f"Ollama content analysis failed for {url}: {e}")

        return findings

    @staticmethod
    def _detect_language_from_content(text: str, url: str) -> str:
        # Quick heuristic: check for Japanese characters
        jp_chars = len(re.findall(r"[぀-ゟ゠-ヿ一-鿿]", text))
        if jp_chars > 10:
            return "jp"
        if "/jp/" in url.lower():
            return "jp"
        return "en"
