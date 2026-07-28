from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class RawFinding:
    """Unprocessed finding from an inspector, before dedup/scoring."""

    url: str
    inspector: str
    category: str
    description: str
    element: str | None = None
    element_html: str | None = None
    current_value: str | None = None
    suggested_value: str | None = None
    raw_metadata: dict = field(default_factory=dict)
    scope: Literal["site", "template", "page", "element"] = "page"
    group_key: str | None = None
    confidence: float = 1.0


class BaseInspector(ABC):
    """Abstract base class for all inspectors."""

    inspector_name: str = "base"

    @abstractmethod
    async def setup(self) -> None:
        """One-time setup (e.g., start browser, verify binary)."""
        ...

    @abstractmethod
    async def inspect(self, url: str, html_content: str,
                      headers: dict | None = None) -> list[RawFinding]:
        """Inspect a single URL and return raw findings."""
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Cleanup resources."""
        ...
