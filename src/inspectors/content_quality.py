from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.ollama_client import OllamaClient
from src.ai.prompt_manager import PromptManager
from src.ai.text_quality import assess_readability, detect_duplicate_risk
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

        # Check for blog articles with no images
        image_count = len(soup.find_all("img"))
        is_blog = "/blog/" in url.lower() or "/insights/" in url.lower()
        if is_blog and word_count >= 200 and image_count == 0:
            findings.append(RawFinding(
                url=url, inspector=self.inspector_name,
                category="article_no_images",
                description=(
                    f"Blog article has no images ({word_count} words, 0 images). "
                    f"Articles with relevant images get 94% more views, "
                    f"higher engagement, and better search rankings."
                ),
                current_value="0 images",
                suggested_value=(
                    "Add 2-3 relevant images (hero image + section illustrations). "
                    "Use free image APIs (Unsplash, Pexels, Pixabay) to find "
                    "high-quality photos matching the article topic."
                ),
                raw_metadata={"word_count": word_count, "image_count": 0},
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

        # Duplicate content detection
        if self._all_page_texts:
            dup_score = detect_duplicate_risk(text, self._all_page_texts)
            if dup_score > 0.85:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="duplicate_content",
                    description=f"Content is {dup_score:.0%} similar to another page",
                    raw_metadata={"similarity": dup_score},
                ))

        # Ollama-based nuance analysis (optional)
        if self.ollama and word_count >= 100:
            try:
                health = await self.ollama.health_check()
                if health:
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
