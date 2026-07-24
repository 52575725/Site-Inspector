from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from src.sources.base import BaseSource, resolve_within


class GitSource(BaseSource):
    """Clone/pull website from a Git repository."""

    source_type = "git"

    def __init__(self, repo_url: str, branch: str = "main",
                 work_dir: Path | None = None):
        self.repo_url = repo_url
        self.branch = branch
        self._work_dir = work_dir

    async def connect(self) -> None:
        pass

    async def sync(self) -> Path:
        if not self._work_dir:
            self._work_dir = Path("data/site_sources/git_repo")

        if self._work_dir.exists() and (self._work_dir / ".git").exists():
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(self._work_dir), "pull", "origin", self.branch,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else f"git pull exited {proc.returncode}"
                raise RuntimeError(f"Git pull failed: {err}")
        else:
            if self._work_dir.exists():
                shutil.rmtree(self._work_dir)
            # Try with -b first, fall back to default branch
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "-b", self.branch, self.repo_url, str(self._work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                # If branch not found, try default branch
                err = stderr.decode().strip() if stderr else ""
                if "remote branch" in err.lower() or "not found" in err.lower():
                    proc2 = await asyncio.create_subprocess_exec(
                        "git", "clone", self.repo_url, str(self._work_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout2, stderr2 = await proc2.communicate()
                    if proc2.returncode != 0:
                        err2 = stderr2.decode().strip() if stderr2 else f"exit {proc2.returncode}"
                        raise RuntimeError(f"Git clone failed: {err2}")
                    # Detect actual default branch
                    proc3 = await asyncio.create_subprocess_exec(
                        "git", "-C", str(self._work_dir),
                        "rev-parse", "--abbrev-ref", "HEAD",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    out3, _ = await proc3.communicate()
                    self.branch = out3.decode().strip()
                else:
                    raise RuntimeError(f"Git clone failed: {err}")

        return self._work_dir

    async def read_file(self, relative_path: str) -> str:
        if not self._work_dir:
            raise RuntimeError("Must call sync() first")
        return resolve_within(self._work_dir, relative_path).read_text(encoding="utf-8")

    async def write_file(self, relative_path: str, content: str) -> None:
        if not self._work_dir:
            raise RuntimeError("Must call sync() first")
        file_path = resolve_within(self._work_dir, relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    async def list_files(self, pattern: str = "*") -> list[str]:
        if not self._work_dir:
            return []
        import glob
        return glob.glob(pattern, root_dir=self._work_dir, recursive=True)

    async def disconnect(self) -> None:
        pass
