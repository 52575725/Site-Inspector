from __future__ import annotations

import difflib
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.ai.ollama_client import OllamaClient
from src.ai.prompt_manager import PromptManager
from src.fixers.base import BaseFixer, FixResult
from src.sources.base import BaseSource


class AltTextGenerator(BaseFixer):
    """Auto-generate alt text for images using Ollama or rule-based fallback."""

    fixer_name = "alt_text_generator"
    fix_type = "fully_auto"
    supported_categories = ["missing_alt_text", "empty_alt_text", "image_missing_alt", "image_empty_alt"]

    def __init__(self, ollama: Optional[OllamaClient] = None,
                 deepseek_api_key: str = "", deepseek_model: str = "deepseek-chat"):
        self.ollama = ollama
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_model = deepseek_model
        self.prompts = PromptManager()

    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        soup = BeautifulSoup(page_content, "html.parser")
        element = issue.get("element", "")

        # Find the image element (may be a snippet)
        if element and element.strip():
            # Try to extract src from the element string
            src_match = re.search(r'src="([^"]*)"', element)
            if src_match:
                src = src_match.group(1)
                img = soup.find("img", src=src)
            else:
                img = None
        else:
            img = None

        if not img:
            # Fallback: find all images without alt text
            imgs_without_alt = soup.find_all("img", alt=None)
            imgs_without_alt.extend(
                soup.find_all("img", alt=lambda a: a is not None and a.strip() == "")
            )
            if not imgs_without_alt:
                return FixResult(
                    success=False, issue_id=issue.get("id", 0),
                    fixer_name=self.fixer_name, fix_type=self.fix_type,
                    file_path=issue.get("file_path", ""),
                    before_content=page_content, after_content=page_content,
                    error_message="No image without alt text found",
                )
            img = imgs_without_alt[0]

        filename = img.get("src", "image")
        # Get surrounding text context
        parent_text = img.parent.get_text(strip=True)[:200] if img.parent else ""
        url = issue.get("url", "")
        language = "jp" if "/jp/" in url.lower() else "en"

        # Try Ollama first, fall back to DeepSeek, then rule-based
        alt_text = None
        if self.ollama:
            try:
                health = await self.ollama.health_check()
                if health:
                    system, prompt = self.prompts.build_prompt(
                        "alt_text_generation",
                        page_type=self._guess_page_type(url),
                        surrounding_text=parent_text,
                        url=url,
                        filename=filename,
                        language=language,
                    )
                    result = await self.ollama.generate_json(
                        prompt=prompt, system=system, temperature=0.2, max_tokens=100,
                    )
                    alt_text = result.get("alt_text")
            except Exception:
                pass

        # DeepSeek fallback
        if (not alt_text or alt_text in ("DECORATIVE", "IMAGE_NEEDS_MANUAL_REVIEW")) \
                and self.deepseek_api_key:
            alt_text = self._deepseek_alt(filename, parent_text, language)

        # Rule-based fallback
        if not alt_text or alt_text in ("DECORATIVE", "IMAGE_NEEDS_MANUAL_REVIEW"):
            alt_text = self._rule_based_alt(filename, parent_text)

        img["alt"] = alt_text
        new_content = str(soup)
        diff = difflib.unified_diff(
            page_content.splitlines(True),
            new_content.splitlines(True),
            lineterm="",
        )

        return FixResult(
            success=True,
            issue_id=issue.get("id", 0),
            fixer_name=self.fixer_name,
            fix_type=self.fix_type,
            file_path=issue.get("file_path", ""),
            before_content=page_content,
            after_content=new_content,
            diff="\n".join(diff),
        )

    def _deepseek_alt(self, filename: str, surrounding_text: str,
                      language: str) -> str | None:
        """Generate alt text using DeepSeek API."""
        import asyncio
        try:
            import httpx
            lang_name = {"en": "English", "jp": "Japanese"}.get(language, "English")

            async def _call():
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.deepseek_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.deepseek_model,
                            "messages": [{
                                "role": "system",
                                "content": (
                                    f"You write concise, SEO-friendly image alt text in {lang_name}. "
                                    "Describe what the image likely shows based on the filename "
                                    "and surrounding context. 5-15 words max. Output only the alt text."
                                ),
                            }, {
                                "role": "user",
                                "content": (
                                    f"Image filename: {filename}\n"
                                    f"Surrounding text: {surrounding_text[:300]}\n"
                                    "Alt text:"
                                ),
                            }],
                            "temperature": 0.3, "max_tokens": 50,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()

            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _call())
                    data = future.result(timeout=25)
            else:
                data = asyncio.run(_call())

            text = data["choices"][0]["message"]["content"].strip().strip('"\'').strip()
            if 5 <= len(text) <= 150:
                return text
            return None
        except Exception:
            return None

    @staticmethod
    def _rule_based_alt(filename: str, surrounding_text: str) -> str:
        basename = filename.split("/")[-1].rsplit(".", 1)[0]
        basename = re.sub(r"[_-]", " ", basename).strip()

        if basename and len(basename) > 2 and not basename.isdigit():
            # Clean up: remove common prefixes/suffixes
            basename = re.sub(r"^(img|image|photo|pic)\d*", "", basename, flags=re.I).strip()
            if basename:
                return basename[:125]

        if surrounding_text:
            return surrounding_text[:125]

        return "Product image"

    @staticmethod
    def _guess_page_type(url: str) -> str:
        if "/blog/" in url:
            return "blog"
        if "/products/" in url:
            return "product"
        if "/about/" in url:
            return "about"
        return "general"
