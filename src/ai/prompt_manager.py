from __future__ import annotations

from pathlib import Path

import yaml


class PromptManager:
    """Load and manage prompt templates from YAML configuration."""

    def __init__(self, prompts_path: Path | None = None):
        self.prompts_path = prompts_path or Path("config/ollama_prompts.yaml")
        self._prompts: dict = {}
        self._load()

    def _load(self) -> None:
        if self.prompts_path.exists():
            with open(self.prompts_path, encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f) or {}

    def reload(self) -> None:
        self._load()

    def get_prompt(self, task: str) -> dict:
        """Get system + prompt template for a task. Returns {system, prompt}."""
        task_config = self._prompts.get(task, {})
        return {
            "system": task_config.get("system", ""),
            "prompt": task_config.get("prompt", ""),
        }

    def build_prompt(self, task: str, **kwargs) -> tuple[str, str]:
        """Build (system, prompt) with all template variables filled in."""
        task_config = self.get_prompt(task)
        system = task_config["system"].format(**kwargs)
        prompt = task_config["prompt"].format(**kwargs)
        return system, prompt
