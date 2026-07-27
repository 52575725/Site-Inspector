"""External reference analysis — EEAT backlinks & business registrations.

Two dimensions:
1. OUTBOUND: Does the site cite authoritative external sources?  (EEAT signal)
2. INBOUND:  Is the business registered on key platforms?        (backlink audit)

Both are critical for EEAT and local/industry SEO.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

from src.inspectors.base import BaseInspector, RawFinding

logger = logging.getLogger(__name__)

# ── Authoritative domains worth citing ──────────────────────────────

AUTHORITATIVE_DOMAINS = {
    "government": [
        ".gov", ".gov.cn", ".go.jp", ".gov.uk", ".gov.hk",
        "trade.gov", "commerce.gov", "customs.gov",
    ],
    "education": [
        ".edu", ".ac.jp", ".ac.uk", ".edu.cn", ".edu.hk",
    ],
    "industry_bodies": [
        "lbma.org.uk",          # London Bullion Market Association
        "worldgoldcouncil.org",
        "silverinstitute.org",
        "kitco.com",
        "reuters.com",
        "bloomberg.com",
        "cmegroup.com",
        "lme.com",              # London Metal Exchange
        "hkex.com.hk",          # Hong Kong Exchange
        "sge.com.cn",           # Shanghai Gold Exchange
        "iso.org",
    ],
    "trade_publications": [
        "metalbulletin.com",
        "fastmarkets.com",
        "spglobal.com",
        "mining.com",
        "bullionvault.com",
    ],
}

# ── Business registration platforms ─────────────────────────────────

# ── 可以免费注册的平台（小微公司实用清单）─────────────────────

BUSINESS_PLATFORMS = [
    # ── 搜索引擎 ──
    {
        "name": "Google Business Profile",
        "url": "https://www.google.com/business/",
        "category": "search_engine",
        "importance": "critical",
        "free": True,
        "description": "免费注册，让公司出现在 Google 搜索和地图",
    },
    {
        "name": "Bing Places",
        "url": "https://www.bingplaces.com/",
        "category": "search_engine",
        "importance": "critical",
        "free": True,
        "description": "免费注册，Bing 搜索引擎的商家页面",
    },
    # ── 社交媒体 ──
    {
        "name": "LinkedIn Company Page",
        "url": "https://www.linkedin.com/company/",
        "category": "social",
        "importance": "critical",
        "free": True,
        "description": "免费创建公司主页，B2B 客户信任度高",
    },
    {
        "name": "Facebook Business Page",
        "url": "https://www.facebook.com/business/",
        "category": "social",
        "importance": "medium",
        "free": True,
        "description": "免费创建商家页面",
    },
    # ── B2B 贸易平台（免费供应商入驻）──
    {
        "name": "Alibaba Supplier (免费版)",
        "url": "https://supplier.alibaba.com/",
        "category": "b2b",
        "importance": "critical",
        "free": True,
        "description": "免费注册供应商账号，全球买家搜索",
    },
    {
        "name": "Made-in-China (免费版)",
        "url": "https://www.made-in-china.com/",
        "category": "b2b",
        "importance": "high",
        "free": True,
        "description": "免费供应商入驻，面向全球采购商",
    },
    {
        "name": "Global Sources (免费版)",
        "url": "https://www.globalsources.com/",
        "category": "b2b",
        "importance": "high",
        "free": True,
        "description": "免费供应商展示，香港公司首选",
    },
    # ── 商业目录（免费基础列表）──
    {
        "name": "Hong Kong Yellow Pages",
        "url": "https://www.yellowpages.com.hk/",
        "category": "directory",
        "importance": "high",
        "free": True,
        "description": "香港黄页，免费收录本地公司",
    },
    {
        "name": "HK TDC (香港贸发局)",
        "url": "https://www.hktdc.com/",
        "category": "directory",
        "importance": "critical",
        "free": True,
        "description": "香港贸发局免费供应商目录，国际贸易买家常查",
    },
    {
        "name": "Kompass (免费版)",
        "url": "https://hk.kompass.com/",
        "category": "directory",
        "importance": "high",
        "free": True,
        "description": "全球 B2B 商业目录，免费基础公司页面",
    },
    {
        "name": "Dun & Bradstreet (免费基础)",
        "url": "https://www.dnb.com/",
        "category": "directory",
        "importance": "medium",
        "free": True,
        "description": "D-U-N-S 编号免费申请，商业信用基础",
    },
]

# ── 需要认证/付费的平台（量力而行）─────────────────────────────

PREMIUM_PLATFORMS = [
    {
        "name": "Alibaba Gold Supplier",
        "url": "https://supplier.alibaba.com/",
        "category": "b2b",
        "importance": "high",
        "cost": "约 USD 3000-5000/年",
        "description": "付费认证供应商，买家信任度高很多",
    },
    {
        "name": "ThomasNet",
        "url": "https://www.thomasnet.com/",
        "category": "b2b",
        "importance": "medium",
        "cost": "约 USD 2000+/年",
        "description": "北美工业采购目录，适合做美国市场",
    },
    {
        "name": "Trustpilot Business",
        "url": "https://business.trustpilot.com/",
        "category": "reviews",
        "importance": "medium",
        "cost": "免费基础版可用",
        "description": "客户评价平台，提升信任度",
    },
]

# ── 行业权威来源（引用数据，不是注册）───────────────────────────

# 这些不是让你"注册"的——是建议在你的网站文章中引用它们的
# 公开数据、价格、标准作为权威背书。
# 例如：写银价分析文章时，引用 LBMA 公开的银价数据。
AUTHORITY_SOURCES_TO_CITE = [
    {
        "name": "LBMA (伦敦金银市场协会)",
        "url": "https://www.lbma.org.uk/prices-and-data/precious-metal-prices",
        "what_to_cite": "公开的银价数据、交割标准文档",
        "why": "贵金属行业最权威的价格基准",
    },
    {
        "name": "LME (伦敦金属交易所)",
        "url": "https://www.lme.com/Metals/Precious-metals/LME-Silver",
        "what_to_cite": "白银期货价格、库存数据",
        "why": "全球金属定价基准",
    },
    {
        "name": "World Gold Council",
        "url": "https://www.gold.org/",
        "what_to_cite": "贵金属市场趋势报告",
        "why": "行业权威研究机构",
    },
    {
        "name": "USGS (美国地质调查局)",
        "url": "https://www.usgs.gov/centers/national-minerals-information-center/silver-statistics-and-information",
        "what_to_cite": "全球银产量/消费量统计数据",
        "why": "政府权威数据，免费公开",
    },
    {
        "name": "Hong Kong Customs",
        "url": "https://www.customs.gov.hk/",
        "what_to_cite": "贵金属进出口法规、清关要求",
        "why": "香港本地政府来源，权威性最高",
    },
]


class ExternalReferencesInspector(BaseInspector):
    """Check outbound citations and external platform registrations.

    OUTBOUND: Does the site link to authoritative sources?
    INBOUND:  Is the business discoverable on key platforms?
    """

    inspector_name = "external_references"

    def __init__(self, target_config: dict | None = None):
        super().__init__()
        self._target_config = target_config or {}
        self._business_name: str = ""
        self._checked_outbound: bool = False
        self._outbound_findings: list[RawFinding] = []

    async def setup(self) -> None:
        org = self._target_config.get("organization", {})
        self._business_name = (
            org.get("name", "")
            or org.get("alternate_name", "")
            or ""
        )

    async def teardown(self) -> None:
        pass

    async def inspect(
        self, url: str, html_content: str, headers: dict | None = None,
    ) -> list[RawFinding]:
        findings: list[RawFinding] = []

        if not html_content:
            return findings

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # ── 1. Outbound authority check (runs once per scan) ────────

        if not self._checked_outbound:
            self._checked_outbound = True

            # Extract all external links
            external_domains: set[str] = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                parsed = urlparse(href)
                if parsed.netloc and parsed.scheme in ("http", "https"):
                    external_domains.add(parsed.netloc.lower())

            if not external_domains:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="eeat_no_external_links",
                    description=(
                        "No external links found on the site. Citing authoritative "
                        "sources (government agencies, industry bodies, trade "
                        "publications) is a strong EEAT signal."
                    ),
                    current_value="0 external links",
                    suggested_value=(
                        "Add 3-5 links to authoritative sources: government trade "
                        "data, LBMA standards, industry reports, etc."
                    ),
                    raw_metadata={
                        "recommended_sources": [
                            "https://www.lbma.org.uk/",
                            "https://www.lme.com/",
                            "https://www.trade.gov/",
                            "https://www.worldgoldcouncil.org/",
                        ],
                    },
                ))

            # Check which authoritative domains are cited
            cited_authoritative: list[str] = []
            for category, domains in AUTHORITATIVE_DOMAINS.items():
                for pattern in domains:
                    for ext_domain in external_domains:
                        if pattern in ext_domain:
                            cited_authoritative.append(
                                f"{ext_domain} ({category})"
                            )

            if not cited_authoritative:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="eeat_no_authoritative_sources",
                    description=(
                        "No authoritative external sources cited. For a precious "
                        "metals trading company, citing LBMA, LME, government "
                        "trade data, and industry reports builds credibility."
                    ),
                    current_value="No authoritative citations",
                    suggested_value=(
                        "Cite data from: LBMA (lbma.org.uk), LME (lme.com), "
                        "government trade statistics, World Gold Council, "
                        "industry publications"
                    ),
                    raw_metadata={
                        "cited_authoritative": cited_authoritative,
                        "recommended_domains": {
                            "LBMA standards": "https://www.lbma.org.uk/",
                            "LME pricing": "https://www.lme.com/",
                            "World Gold Council": "https://www.worldgoldcouncil.org/",
                            "US Trade Data": "https://www.trade.gov/",
                            "Silver Institute": "https://www.silverinstitute.org/",
                        },
                    },
                ))
            elif len(cited_authoritative) < 3:
                findings.append(RawFinding(
                    url=url, inspector=self.inspector_name,
                    category="eeat_few_authoritative_sources",
                    description=(
                        f"Only {len(cited_authoritative)} authoritative sources "
                        f"cited: {', '.join(cited_authoritative)}. "
                        f"More citations build stronger EEAT signals."
                    ),
                    current_value=f"{len(cited_authoritative)} sources",
                    suggested_value="Add 2-3 more authoritative citations",
                    raw_metadata={"cited": cited_authoritative},
                ))

            self._outbound_findings = findings

        return findings

    def get_registration_checklist(self) -> dict:
        """Return categorized recommendations for a small business.

        Returns:
            free_platforms:  cost-free platforms anyone can register on
            premium_platforms:  paid/verified platforms (optional)
            authority_sources:  authoritative sources worth citing (not registering)
            business_name:  configured business name
        """
        free = []
        for p in BUSINESS_PLATFORMS:
            free.append({
                "platform": p["name"],
                "url": p["url"],
                "category": p["category"],
                "importance": p["importance"],
                "description": p.get("description", ""),
                "action": (
                    f"Register {self._business_name} on {p['name']}"
                    if self._business_name
                    else f"Visit {p['url']} to register"
                ),
            })

        premium = []
        for p in PREMIUM_PLATFORMS:
            premium.append({
                "platform": p["name"],
                "url": p["url"],
                "importance": p["importance"],
                "cost": p.get("cost", ""),
                "description": p.get("description", ""),
            })

        cite = []
        for s in AUTHORITY_SOURCES_TO_CITE:
            cite.append({
                "name": s["name"],
                "url": s["url"],
                "what_to_cite": s["what_to_cite"],
                "why": s["why"],
            })

        return {
            "free_platforms": free,
            "premium_platforms": premium,
            "authority_sources_to_cite": cite,
            "business_name": self._business_name,
            "summary": (
                f"For a small business like {self._business_name}, start with: "
                f"1) Google Business Profile, 2) LinkedIn Company Page, "
                f"3) HK TDC directory, 4) one B2B platform (Alibaba or Global Sources). "
                f"Then add authoritative citations to LBMA/LME public data in your content."
            ) if self._business_name else "",
        }
