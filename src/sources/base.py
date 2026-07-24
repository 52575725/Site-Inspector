from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def resolve_within(root: str | Path, relative_path: str) -> Path:
    """Resolve a relative path and guarantee that it remains below root."""
    root_path = Path(root).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute() or "\x00" in relative_path:
        raise ValueError(f"Unsafe source path: {relative_path!r}")
    resolved = (root_path / candidate).resolve()
    if not resolved.is_relative_to(root_path):
        raise ValueError(f"Source path escapes root: {relative_path!r}")
    return resolved


class BaseSource(ABC):
    """Abstract access to website source files for fix operations."""

    source_type: str = "base"

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def sync(self) -> Path:
        """Sync remote files to local working directory. Returns local path."""
        ...

    @abstractmethod
    async def read_file(self, relative_path: str) -> str:
        ...

    @abstractmethod
    async def write_file(self, relative_path: str, content: str) -> None:
        ...

    @abstractmethod
    async def list_files(self, pattern: str = "*") -> list[str]:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()
