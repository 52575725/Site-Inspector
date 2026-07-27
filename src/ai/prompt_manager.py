from __future__ import annotations

from pathlib import Path

import yaml


class PromptManager:
    """Load and manage prompt templates from YAML configuration.

    Supports business-config variables that are automatically merged into
    every prompt, so templates can use {business_description}, {tone},
    {industry}, and {target_audience} without every caller passing them.
    """

    def __init__(self, prompts_path: Path | None = None,
                 business_config: dict | None = None):
        self.prompts_path = prompts_path or Path("config/ollama_prompts.yaml")
        self._prompts: dict = {}
        self._business_defaults: dict = {
            "business_description": "a business website",
            "industry": "general",
            "tone": "professional",
            "target_audience": "general audience",
        }
        if business_config:
            self._business_defaults.update(business_config)
        self._load()

    def _load(self) -> None:
        if self.prompts_path.exists():
            with open(self.prompts_path, encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f) or {}

    def reload(self) -> None:
        self._load()

    def set_business_config(self, config: dict) -> None:
        """Update business defaults (e.g. after target config change)."""
        self._business_defaults.update(config)

    def get_prompt(self, task: str) -> dict:
        """Get system + prompt template for a task. Returns {system, prompt}."""
        task_config = self._prompts.get(task, {})
        return {
            "system": task_config.get("system", ""),
            "prompt": task_config.get("prompt", ""),
        }

    def build_prompt(self, task: str, **kwargs) -> tuple[str, str]:
        """Build (system, prompt) with all template variables filled in.

        Business-config defaults (business_description, tone, etc.) are
        automatically merged — callers only need to pass page-specific vars.
        Missing template variables are replaced with safe defaults instead of
        raising KeyError.
        """
        from string import Formatter

        # Merge: caller kwargs take priority over business defaults
        merged = {**self._business_defaults, **kwargs}
        task_config = self.get_prompt(task)

        # Safe format: replace missing keys with "{key}" placeholder
        def _safe_format(template: str) -> str:
            result = template
            for field_name, _, _, _ in Formatter().parse(template):
                if field_name and field_name not in merged:
                    merged.setdefault(field_name, f"{{{field_name}}}")
            return template.format(**merged)

        system = _safe_format(task_config["system"])
        prompt = _safe_format(task_config["prompt"])
        return system, prompt
