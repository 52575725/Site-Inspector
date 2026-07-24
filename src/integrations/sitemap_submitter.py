from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class SitemapSubmitter:
    """Submits sitemaps to search engines via their APIs.

    Google deprecated the public ping endpoint in 2023. The recommended
    replacement is Google Search Console API (needs OAuth) or simply
    listing the sitemap in robots.txt.

    This submitter tries API submission first, then falls back to
    providing manual submission links.
    """

    MANUAL_SUBMIT_LINKS = {
        "google": "https://search.google.com/search-console/sitemaps",
        "bing": "https://www.bing.com/webmasters/sitemaps",
    }

    BAIDU_SUBMIT_URL = "https://data.zz.baidu.com/urls"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def submit_to_google(
        self, sitemap_url: str, gsc_property: str = "",
    ) -> tuple[bool, str]:
        """Submit sitemap to Google.

        Tries Search Console API if credentials are available,
        otherwise provides manual submission link.
        """
        if gsc_property:
            return await self._gsc_submit(sitemap_url, gsc_property)

        # Ping is deprecated but try anyway, then provide manual link
        ok, msg = await self._ping_google(sitemap_url)
        if ok:
            return True, msg
        return False, (
            f"请手动提交: {self.MANUAL_SUBMIT_LINKS['google']} "
            f"(旧 ping 接口已废弃: {msg})"
        )

    async def submit_to_bing(self, sitemap_url: str) -> tuple[bool, str]:
        """Submit sitemap to Bing. The ping endpoint returns 410 (gone).
        Provide manual link as fallback."""
        ok, msg = await self._ping_bing(sitemap_url)
        if ok:
            return True, msg
        return False, (
            f"请手动提交: {self.MANUAL_SUBMIT_LINKS['bing']} "
            f"(Bing ping 接口已关闭)"
        )

    async def submit_to_baidu(
        self, site_url: str, urls: list[str], token: str,
    ) -> tuple[bool, str]:
        """Submit URLs to Baidu link-submit API."""
        if not token:
            return False, "需要 Baidu token — 从 ziyuan.baidu.com 获取"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.BAIDU_SUBMIT_URL,
                    params={"site": site_url, "token": token},
                    data="\n".join(urls),
                    headers={"Content-Type": "text/plain"},
                )
                if resp.status_code == 200:
                    result = resp.json()
                    return True, (
                        f"已提交 {result.get('success', 0)} 条 URL，"
                        f"剩余配额: {result.get('remain', 'unknown')}"
                    )
                return False, f"Baidu API 返回 {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return False, f"Baidu 提交失败: {e}"

    async def submit_all(
        self, sitemap_url: str, site_url: str = "",
        baidu_token: str = "", baidu_urls: list[str] | None = None,
        gsc_property: str = "",
    ) -> dict[str, tuple[bool, str]]:
        """Submit sitemap to all supported search engines."""
        results = {}

        # Google
        ok, msg = await self.submit_to_google(sitemap_url, gsc_property)
        results["google"] = (ok, msg)

        # Bing
        ok, msg = await self.submit_to_bing(sitemap_url)
        results["bing"] = (ok, msg)

        # Baidu (optional)
        if baidu_token and baidu_urls:
            ok, msg = await self.submit_to_baidu(site_url, baidu_urls, baidu_token)
            results["baidu"] = (ok, msg)

        return results

    # ── Private ──────────────────────────────────────────────────────

    async def _ping_google(self, sitemap_url: str) -> tuple[bool, str]:
        """Deprecated Google ping. Returns True only if it somehow works."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    "https://www.google.com/ping",
                    params={"sitemap": sitemap_url},
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    return True, "提交成功"
                return False, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, f"网络错误: {e}"

    async def _ping_bing(self, sitemap_url: str) -> tuple[bool, str]:
        """Deprecated Bing ping."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    "https://www.bing.com/ping",
                    params={"sitemap": sitemap_url},
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    return True, "提交成功"
                return False, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, f"网络错误: {e}"

    async def _gsc_submit(
        self, sitemap_url: str, gsc_property: str,
    ) -> tuple[bool, str]:
        """Submit sitemap via Google Search Console API (requires OAuth)."""
        return False, "GSC API 需要 OAuth 认证，请手动在 Search Console 提交"
