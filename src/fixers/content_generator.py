from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.deepseek_client import DeepSeekClient
from src.ai.ollama_client import OllamaClient
from src.ai.prompt_manager import PromptManager
from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource

logger = logging.getLogger(__name__)


class ContentGenerator(BaseFixer):
    """Generates full-length SEO content for thin or empty pages.

    Uses DeepSeek API for high-quality article generation (1000+ words).
    Falls back to local Ollama if DeepSeek is unavailable.

    Unlike ContentRewriter (which only provides diagnostic suggestions),
    ContentGenerator actually produces replacement body content —
    paragraphs, headings, lists, and CTAs — and writes it to the page.

    Fix type is "semi_auto" because generated content should be
    reviewed by a human before publishing.
    """

    fixer_name = "content_generator"
    fix_type = "semi_auto"
    supported_categories = [
        "thin_content",
        "low_content_quality_ai",
        "low_readability",
        "missing_content",
        "keyword_density_low",
    ]

    # Minimum word count to consider content "not thin"
    TARGET_WORD_COUNT = 800
    # Minimum existing words to attempt enhancement instead of replacement
    MIN_WORDS_FOR_ENHANCEMENT = 100

    def __init__(
        self,
        ollama: Optional[OllamaClient] = None,
        deepseek: Optional[DeepSeekClient] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        self.ollama = ollama
        self.deepseek = deepseek
        self.prompts = prompt_manager or PromptManager()

    def _get_ai(self):
        """Return the best available AI client: DeepSeek > Ollama."""
        if self.deepseek:
            return "deepseek", self.deepseek
        if self.ollama:
            return "ollama", self.ollama
        return None, None

    async def generate_fix(
        self,
        issue: dict,
        source: BaseSource,
        page_content: str,
    ) -> FixResult:
        url = issue.get("url", "")
        category = issue.get("category", "")
        file_path = issue.get("file_path", "")

        engine_name, ai = self._get_ai()
        if not ai:
            return FixResult(
                success=False,
                issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message="No AI engine available (DeepSeek or Ollama required)",
            )

        try:
            soup = BeautifulSoup(page_content, "html.parser")

            # Extract existing metadata
            body = soup.find("body")
            title_tag = soup.find("title")
            h1_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else url
            h1_text = h1_tag.get_text(strip=True) if h1_tag else title

            # Count existing words in body
            if body:
                for tag in body.find_all(["script", "style"]):
                    tag.decompose()
                existing_text = body.get_text(separator=" ", strip=True)
                existing_words = len(existing_text.split())
            else:
                existing_words = 0

            # Determine page type from URL
            page_type = self._classify_page_type(url)

            # Collect existing headings for context
            existing_headings = []
            if body:
                for h in body.find_all(["h1", "h2", "h3"]):
                    h_text = h.get_text(strip=True)
                    if h_text:
                        existing_headings.append(f"<{h.name}>{h_text}</{h.name}>")

            # Determine language
            language = "en"
            html_tag = soup.find("html")
            if html_tag:
                lang_attr = html_tag.get("lang", "")
                if lang_attr:
                    language = lang_attr[:2].lower()
            if "/jp/" in url or "/ja/" in url:
                language = "ja"

            # Extract target keywords from issue description or existing headings
            target_keywords = self._extract_keywords(issue, existing_headings, title)

            # Choose strategy: enhance existing or generate fresh
            if existing_words >= self.MIN_WORDS_FOR_ENHANCEMENT and category != "missing_content":
                strategy = "enhance"
                word_count_target = max(self.TARGET_WORD_COUNT, existing_words + 300)
            else:
                strategy = "generate"
                word_count_target = self.TARGET_WORD_COUNT

            logger.info(
                f"ContentGenerator: {strategy} content for {url} "
                f"(engine={engine_name}, existing={existing_words} words, "
                f"target={word_count_target}, keywords={target_keywords})"
            )

            # Build prompt and call AI (DeepSeek or Ollama)
            system, prompt = self.prompts.build_prompt(
                "content_generation",
                page_type=page_type,
                url=url,
                title=title,
                target_keywords=", ".join(target_keywords) if target_keywords else "silver trading, precious metals",
                existing_headings="\n".join(existing_headings[:10]) if existing_headings else "(none)",
                current_word_count=str(existing_words),
                word_count_target=str(word_count_target),
                language=language,
            )

            generated = await ai.generate_text(
                prompt=prompt,
                system=system,
                temperature=0.7,
                max_tokens=4000,
            )

            if not generated or len(generated.strip()) < 200:
                return FixResult(
                    success=False,
                    issue_id=issue.get("id", 0),
                    fixer_name=self.fixer_name,
                    fix_type=self.fix_type,
                    file_path=file_path,
                    before_content=page_content,
                    after_content=page_content,
                    error_message=f"{engine_name} returned insufficient content ({len(generated)} chars)",
                )

            # Sanitize generated content (remove markdown wrappers if any)
            generated = self._sanitize_output(generated)

            # Build the fixed page
            after_content = self._insert_content(
                soup, body, generated, strategy, existing_words,
            )

            # Generate diff
            diff = self._generate_diff(page_content, after_content, file_path)

            generated_words = len(generated.split())
            logger.info(
                f"ContentGenerator: generated {generated_words} words for {url}"
            )

            return FixResult(
                success=True,
                issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=after_content,
                diff=diff,
            )

        except Exception as e:
            logger.error(f"ContentGenerator failed for {url}: {e}")
            return FixResult(
                success=False,
                issue_id=issue.get("id", 0),
                fixer_name=self.fixer_name,
                fix_type=self.fix_type,
                file_path=file_path,
                before_content=page_content,
                after_content=page_content,
                error_message=str(e)[:500],
            )

    def _classify_page_type(self, url: str) -> str:
        """Classify the page type from URL structure."""
        url_lower = url.lower()
        if "/blog/" in url_lower or "/insights/" in url_lower:
            return "blog_article"
        if "/product" in url_lower:
            return "product"
        if "/about" in url_lower:
            return "about"
        if "/contact" in url_lower:
            return "contact"
        if url_lower.rstrip("/").endswith(("home", "index")) or url_lower == "/":
            return "homepage"
        return "landing_page"

    def _extract_keywords(
        self,
        issue: dict,
        existing_headings: list[str],
        title: str,
    ) -> list[str]:
        """Extract target keywords from available context."""
        keywords = []

        # Try to extract from issue description
        description = issue.get("description", "")
        # Look for keyword hints like "keywords: X, Y, Z"
        kw_match = re.search(
            r"(?:keywords?|target(?:ing)?)\s*:?\s*(.+?)(?:\.|$)",
            description, re.IGNORECASE,
        )
        if kw_match:
            keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

        # Extract from headings as fallback
        if not keywords:
            for h_text in existing_headings[:3]:
                clean = re.sub(r"<[^>]+>", "", h_text).strip()
                if clean and len(clean) > 10:
                    keywords.append(clean)

        # Extract from title
        if not keywords and title:
            # Use title without site name suffix
            clean_title = re.split(r"\s*[|–-]\s*", title)[0].strip()
            if clean_title:
                keywords.append(clean_title)

        return keywords[:5]

    def _sanitize_output(self, text: str) -> str:
        """Clean up LLM output — remove markdown wrappers, fix common issues."""
        # Remove markdown code block wrappers
        text = re.sub(r"^```(?:html)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text)

        # Remove "Here is the content..." preamble
        text = re.sub(
            r"^(?:Here\s+(?:is|are)\s+.*?(?:content|article|HTML).*?[:\n])",
            "", text, flags=re.IGNORECASE,
        ).strip()

        fragment = BeautifulSoup(text, "html.parser")
        for tag in fragment.find_all(["script", "iframe", "object", "embed", "form", "base"]):
            tag.decompose()
        for tag in fragment.find_all(True):
            for attr in list(tag.attrs):
                if attr.lower().startswith("on"):
                    del tag.attrs[attr]
            for attr in ("href", "src", "action"):
                value = tag.get(attr)
                if isinstance(value, str) and value.strip().lower().startswith(
                    ("javascript:", "data:text/html")
                ):
                    del tag.attrs[attr]
        return str(fragment)

    def _insert_content(
        self,
        soup: BeautifulSoup,
        body,
        generated: str,
        strategy: str,
        existing_words: int,
    ) -> str:
        """Insert generated content into the page body."""
        if not body:
            # No body tag — wrap in basic HTML structure
            html_tag = soup.find("html")
            if html_tag:
                new_body = soup.new_tag("body")
                new_body.append(BeautifulSoup(generated, "html.parser"))
                html_tag.append(new_body)
            else:
                return f"<html><body>\n{generated}\n</body></html>"
            return str(soup)

        # Find the main content area
        main = body.find("main") or body.find("article") or body.find(
            "div", class_=re.compile(r"content|article|blog|post|main", re.I)
        )

        if not main:
            # Try to find the content container after nav/header
            nav = body.find("nav")
            header = body.find("header")
            if nav or header:
                # Insert after nav/header
                insert_point = (nav or header)
                new_div = soup.new_tag("div")
                new_div["class"] = "blog-article"
                new_div.append(BeautifulSoup(generated, "html.parser"))
                insert_point.insert_after(new_div)
            else:
                # No semantic landmarks found — wrap generated content in a
                # div and append to body instead of clearing the whole page.
                wrapper = soup.new_tag("div")
                wrapper["class"] = "generated-content"
                wrapper.append(BeautifulSoup(generated, "html.parser"))
                body.append(wrapper)
        else:
            if strategy == "enhance":
                # Append to existing content
                main.append(BeautifulSoup(generated, "html.parser"))
            else:
                # Replace content area
                main.clear()
                main.append(BeautifulSoup(generated, "html.parser"))

        return str(soup)

    @staticmethod
    def _generate_diff(before: str, after: str, file_path: str) -> str:
        """Generate a simplified diff summary."""
        import difflib
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))
        return "\n".join(diff_lines[:200])  # Cap diff size
