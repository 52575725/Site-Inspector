from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.sources.base import BaseSource


@dataclass
class FixResult:
    success: bool
    issue_id: int
    fixer_name: str
    fix_type: str  # "fully_auto" | "semi_auto" | "manual_only"
    file_path: str
    before_content: str
    after_content: str
    diff: str = ""
    error_message: str = ""


class BaseFixer(ABC):
    """Abstract base class for all fixers."""

    fixer_name: str = "base"
    fix_type: str = "manual_only"
    supported_categories: list[str] = []

    def can_fix(self, category: str) -> bool:
        return category in self.supported_categories

    @abstractmethod
    async def generate_fix(self, issue: dict, source: BaseSource,
                           page_content: str) -> FixResult:
        ...

    async def validate_fix(self, fix_result: FixResult,
                           original_url: str) -> float:
        """Default: return 0.0 (no visual diff capability without sandbox).
        Override for visual validation."""
        return 0.0
