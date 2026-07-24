from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from config.settings import Settings
from src.sources.base import BaseSource, resolve_within


class HttpSource(BaseSource):
    """Download website files via HTTP for fixing purposes."""

    source_type = "http"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.target_base_url
        self.client = httpx.AsyncClient(
            timeout=30,
            headers={"User-Agent": settings.crawl_user_agent},
            follow_redirects=True,
        )
        self._work_dir: Path | None = None

    async def connect(self) -> None:
        resp = await self.client.head(self.base_url)
        resp.raise_for_status()

    async def sync(self) -> Path:
        self._work_dir = self.settings.data_dir / "site_sources" / self.settings.target_name
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Download all HTML pages discovered via sitemap
        from src.crawler.sitemap_parser import SitemapParser
        parser = SitemapParser(self.client)
        pages = await parser.parse(f"{self.base_url}/sitemap.xml")

        for page in pages:
            path = urlparse(page.url).path.strip("/") or "index"
            if not path.endswith(".html"):
                path = path.rstrip("/") + "/index.html" if "/" in path else path + ".html"

            file_path = resolve_within(self._work_dir, path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                resp = await self.client.get(page.url)
                if resp.status_code == 200:
                    file_path.write_text(resp.text, encoding="utf-8")
            except Exception:
                continue

        return self._work_dir

    async def read_file(self, relative_path: str) -> str:
        if not self._work_dir:
            raise RuntimeError("Must call sync() before reading files")
        return resolve_within(self._work_dir, relative_path).read_text(encoding="utf-8")

    async def write_file(self, relative_path: str, content: str) -> None:
        if not self._work_dir:
            raise RuntimeError("Must call sync() before writing files")
        resolve_within(self._work_dir, relative_path).write_text(content, encoding="utf-8")

    async def list_files(self, pattern: str = "*") -> list[str]:
        if not self._work_dir:
            return []
        import glob
        return glob.glob(pattern, root_dir=self._work_dir, recursive=True)

    async def disconnect(self) -> None:
        await self.client.aclose()
