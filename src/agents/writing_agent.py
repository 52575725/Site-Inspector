from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class WritingTask:
    prompt: str
    page_type: str
    content_direction: str
    language: str


class ArticleWritingAgent:
    """Execute a confirmed writing plan with direction-aware generation settings."""

    def __init__(self, settings, *, ai_client=None) -> None:
        from src.ai.deepseek_client import DeepSeekClient

        self.settings = settings
        self.ai = ai_client or DeepSeekClient(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            timeout=max(settings.deepseek_timeout, 180),
        )
        self._owns_ai = ai_client is None

    async def write(self, task: WritingTask) -> str:
        temperature = {
            "news": 0.35,
            "market_event": 0.4,
            "deep_analysis": 0.45,
            "buyer_question": 0.5,
            "industry_trend": 0.5,
            "evergreen_guide": 0.55,
        }.get(task.content_direction, 0.5)
        system = (
            "You are an expert SEO content writer operating inside an article agent. "
            "Follow the confirmed brief, answer validated user queries, cite only verified "
            "authority URLs, never copy reference wording, and output valid HTML only. "
            "Start with the reader's concrete question instead of generic scene-setting. "
            "Make every H2 add distinct facts, mechanisms, examples, trade-offs, or actions. "
            "Do not invent experience, interviews, quotations, statistics, or market claims. "
            f"Write a {task.page_type} article in {task.language}."
        )
        result = await self.ai.generate_text(
            task.prompt,
            system=system,
            temperature=temperature,
            max_tokens=7000,
        )
        if not str(result or "").strip():
            raise ValueError("The writing agent returned empty content")
        return str(result)

    async def revise(self, task: WritingTask, html: str, issues: list[str]) -> str:
        feedback = "\n".join(f"- {issue}" for issue in issues)
        prompt = f"""Revise the generated article so it passes every quality requirement.

Confirmed writing task:
{task.prompt}

Quality failures:
{feedback}

Current HTML:
<generated-article>
{html}
</generated-article>

Preserve correct facts and approved links. Add no URL that is absent from the confirmed
research task. Return one complete, valid HTML document only."""
        result = await self.ai.generate_text(
            prompt,
            system=(
                "You are the revision stage of an article quality agent. Fix only the listed "
                "failures, preserve the confirmed topic, and output valid HTML only."
            ),
            temperature=0.25,
            max_tokens=7000,
        )
        if not str(result or "").strip():
            raise ValueError("The writing agent returned an empty revision")
        return str(result)

    async def close(self) -> None:
        if self._owns_ai:
            await self.ai.close()
