from __future__ import annotations

import difflib
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.ollama_client import OllamaClient
from src.ai.prompt_manager import PromptManager
from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class ContentRewriter(BaseFixer):
    """Semi-auto fix thin content and low readability issues."""

    fixer_name = "content_rewriter"
    fix_type = "semi_auto"
    supported_categories = ["thin_content", "low_readability",
                           "duplicate_content", "low_content_quality_ai"]

    def __init__(self, ollama: Optional[OllamaClient] = None):
        self.ollama = ollama
        self.prompts = PromptManager()

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        category = issue.get("category", "")

        # This fixer doesn't directly modify — it generates suggestions
        # Actual content rewriting needs human approval
        suggestion = await self._generate_suggestion(issue, page_content)

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=page_content,
            after_content=page_content,  # No auto-modification
            diff=f"# Content improvement suggestion for {issue.get('url')}\n# {suggestion}",
            error_message="",
        )

    async def _generate_suggestion(self, issue: dict, page_content: str) -> str:
        url = issue.get("url", "")
        language = "jp" if "/jp/" in url.lower() else "en"

        if not self.ollama:
            return (f"Page at {url} needs content expansion. "
                    f"Current content is too thin. Consider adding more detailed "
                    f"information about the topic.")

        try:
            health = await self.ollama.health_check()
            if not health:
                return "Ollama unavailable for content suggestion."

            soup = BeautifulSoup(page_content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

            system, prompt = self.prompts.build_prompt(
                "content_quality",
                url=url,
                title=issue.get("title", ""),
                word_count=str(len(text.split())),
                content_excerpt=text[:3000],
                language=language,
            )
            result = await self.ollama.generate_json(
                prompt=prompt, system=system, temperature=0.5, max_tokens=500,
            )
            issues_list = result.get("issues", [])
            strengths = result.get("strengths", [])
            parts = []
            if issues_list:
                parts.append("Issues: " + "; ".join(issues_list))
            if strengths:
                parts.append("Strengths: " + "; ".join(strengths))
            return "\n".join(parts) if parts else "Content could be expanded."
        except Exception as e:
            return f"Unable to generate suggestion: {e}"
