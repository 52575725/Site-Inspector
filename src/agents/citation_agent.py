from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.agents.models import QualityCheck, QualityReport


class ArticleCitationAgent:
    """Verify generated links and citation coverage against researched sources."""

    FACT_PATTERN = re.compile(
        r"\b20\d{2}\b|\b(?:effective|regulation|regulated|requirement|penalty|"
        r"percent|percentage|million|billion|tariff|duty|report|data|study|"
        r"record-breaking|record high|all-time high|historic high|typically|average|"
        r"consecutive years?|industry standard|market leader|fastest-growing)\b|"
        r"(?:法规|监管|生效|要求|处罚|数据|报告|研究|百分之|"
        r"历史新高|创纪录|通常|平均|连续\S{0,4}年|行业标准)|"
        r"(?:規制|施行|要件|罰則|データ|報告|調査|"
        r"過去最高|記録的|通常|平均)",
        re.IGNORECASE,
    )

    def inspect(self, html: str, research_report: dict | None) -> QualityReport:
        report = research_report or {}
        allowed_urls = self._allowed_urls(report)
        profile_url = str((report.get("profile") or {}).get("website_url", ""))
        site_host = self._host(profile_url)
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article") or soup.find("main") or soup.body or soup

        external_urls: list[str] = []
        approved_urls: list[str] = []
        unapproved_urls: list[str] = []
        for link in article.find_all("a", href=True):
            if link.find_parent("figure", class_="article-media"):
                continue
            url = str(link.get("href", "")).strip()
            host = self._host(url)
            if not host or host == site_host:
                continue
            external_urls.append(url)
            if url in allowed_urls:
                approved_urls.append(url)
            else:
                unapproved_urls.append(url)

        factual_blocks = []
        uncited_factual_blocks = []
        for block in article.find_all(["p", "li", "td"]):
            if block.find_parent("figure", class_="article-media"):
                continue
            text = block.get_text(" ", strip=True)
            if len(text) < 25 or not self.FACT_PATTERN.search(text):
                continue
            factual_blocks.append(text)
            links = {str(link.get("href", "")).strip() for link in block.find_all("a", href=True)}
            if not links & allowed_urls:
                uncited_factual_blocks.append(text[:180])

        has_researched_sources = bool(allowed_urls)
        coverage = (
            (len(factual_blocks) - len(uncited_factual_blocks)) / len(factual_blocks)
            if factual_blocks else 1.0
        )
        checks = [
            QualityCheck(
                name="approved_external_links",
                passed=not unapproved_urls,
                severity="error",
                message=(
                    "All external links come from researched sources."
                    if not unapproved_urls
                    else f"Found {len(set(unapproved_urls))} external link(s) not present in the research evidence."
                ),
            ),
            QualityCheck(
                name="researched_sources_used",
                passed=not has_researched_sources or bool(approved_urls),
                severity="error",
                message=(
                    "No authoritative source was approved, so no citation is required."
                    if not has_researched_sources
                    else (
                        "The article uses researched authoritative links."
                        if approved_urls
                        else "Researched sources exist, but the article does not cite any of them."
                    )
                ),
            ),
            QualityCheck(
                name="factual_claim_citations",
                passed=not uncited_factual_blocks,
                severity="error",
                message=(
                    "Time-sensitive and regulatory claims have inline researched citations."
                    if not uncited_factual_blocks
                    else f"Found {len(uncited_factual_blocks)} factual block(s) without an inline researched citation."
                ),
            ),
        ]
        return QualityReport(
            passed=all(check.passed for check in checks),
            checks=checks,
            metrics={
                "researched_source_count": len(allowed_urls),
                "external_link_count": len(external_urls),
                "approved_link_count": len(approved_urls),
                "factual_block_count": len(factual_blocks),
                "uncited_factual_block_count": len(uncited_factual_blocks),
                "citation_coverage": round(coverage, 3),
            },
        )

    @staticmethod
    def _host(url: str) -> str:
        host = (urlparse(url).hostname or "").casefold()
        return host[4:] if host.startswith("www.") else host

    @staticmethod
    def _allowed_urls(report: dict) -> set[str]:
        urls = {
            str(item.get("url", "")).strip()
            for item in report.get("authority_sources", [])
            if isinstance(item, dict)
        }
        decision = report.get("editorial_decision") or {}
        urls.update(str(url).strip() for url in decision.get("authority_source_urls", []))
        urls.update(str(url).strip() for url in decision.get("event_source_urls", []))
        return {url for url in urls if url.startswith(("http://", "https://"))}
