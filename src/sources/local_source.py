from __future__ import annotations

from pathlib import Path

from src.sources.base import BaseSource, resolve_within


class LocalSource(BaseSource):
    """Access website files from a local directory."""

    source_type = "local"

    def __init__(self, root_path: str | Path):
        self.root = Path(root_path)
        self._connected = False

    async def connect(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Local source path not found: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}")
        self._connected = True

    async def sync(self) -> Path:
        return self.root

    async def read_file(self, relative_path: str) -> str:
        file_path = resolve_within(self.root, relative_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.read_text(encoding="utf-8")

    async def write_file(self, relative_path: str, content: str) -> None:
        file_path = resolve_within(self.root, relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    async def list_files(self, pattern: str = "*") -> list[str]:
        import glob
        files = glob.glob(pattern, root_dir=self.root, recursive=True)
        return files

    async def disconnect(self) -> None:
        self._connected = False
