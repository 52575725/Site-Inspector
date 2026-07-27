from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.ai.deepseek_client import DeepSeekClient
from src.ai.ollama_client import OllamaClient
from src.ai.prompt_manager import PromptManager
from src.crawler.crawler import Crawler
from src.inspectors.accessibility import AccessibilityInspector
from src.inspectors.base import RawFinding
from src.inspectors.broken_links import BrokenLinksInspector
from src.inspectors.cannibalization import CannibalizationDetector
from src.inspectors.competitor_gap import CompetitorGapInspector
from src.inspectors.content_clustering import ContentClusterInspector
from src.inspectors.content_freshness import ContentFreshnessInspector
from src.inspectors.content_gap import ContentGapDetector
from src.inspectors.content_quality import ContentQualityInspector
from src.inspectors.crawl_budget import CrawlBudgetInspector
from src.inspectors.eeat import EEATInspector
from src.inspectors.external_references import ExternalReferencesInspector
from src.inspectors.headers import HeadersInspector
from src.inspectors.image_seo import ImageSEOInspector
from src.inspectors.js_seo import JSSeoInspector
from src.inspectors.keyword_analyzer import KeywordAnalyzer
from src.inspectors.link_graph import LinkGraphInspector
from src.inspectors.mobile import MobileInspector
from src.inspectors.robots_txt import RobotsTxtInspector
from src.inspectors.performance import PerformanceInspector
from src.inspectors.platform_seo import PlatformSEOInspector
from src.inspectors.semantic_content import SemanticContentInspector
from src.inspectors.seo import SEOInspector
from src.inspectors.sitemap import SitemapInspector
from src.inspectors.structured_data import StructuredDataValidator
from src.inspectors.url_audit import URLAuditor
from src.storage.chroma_store import ChromaStore
from src.storage.models import Issue, PageScan, Scan
from src.storage.repositories import (
    AuditLogRepository,
    IssueRepository,
    PageScanRepository,
    ScanRepository,
    TargetRepository,
)

logger = logging.getLogger(__name__)


class ScanOrchestrator:
    """Orchestrates a full scan: crawl → inspect → save."""

    def __init__(self, settings: Settings, session: AsyncSession,
                 ollama: Optional[OllamaClient] = None,
                 deepseek: Optional[DeepSeekClient] = None):
        self.settings = settings
        self.session = session
        self.ollama = ollama
        self.deepseek = deepseek
        target_config = self.settings.__class__.load_target(settings.target_name)
        business_config = target_config.get("business", {})
        self.prompts = PromptManager(business_config=business_config)

        # Repositories
        self.target_repo = TargetRepository(session)
        self.scan_repo = ScanRepository(session)
        self.page_repo = PageScanRepository(session)
        self.issue_repo = IssueRepository(session)
        self.audit_repo = AuditLogRepository(session)

        # ChromaDB for dedup
        self.chroma = ChromaStore(settings)

        # Cached crawled pages for quick-scan fix reuse
        self._last_crawled_pages: list = []

    async def run_full_scan(self, target_name: str | None = None,
                           target_base_url: str | None = None,
                           target_languages: list[str] | None = None,
                           existing_scan_id: int | None = None) -> Scan:
        """Execute a full inspection of the target website."""
        logger.info("Starting full site scan...")

        _name = target_name or self.settings.target_name
        _base_url = target_base_url or self.settings.target_base_url
        _languages = target_languages or self.settings.target_languages
        target_config = self.settings.__class__.load_target(_name)

        # 1. Get or create target
        target = await self.target_repo.get_or_create(
            name=_name,
            base_url=_base_url,
            source_type=self.settings.source_type,
            languages=_languages,
        )

        # 2. Create or reuse scan record
        if existing_scan_id is not None:
            scan = await self.scan_repo.get_by_id(existing_scan_id)
            if scan is None:
                raise ValueError(f"Scan {existing_scan_id} not found")
        else:
            scan = await self.scan_repo.create(target_id=target.id, scan_type="daily")
        await self.audit_repo.log("scan_started", "scan", scan.id)
        await self.scan_repo.set_phase(scan.id, "crawling")
        await self.session.commit()

        # 3. Crawl all pages
        target_crawl_config = target_config.get("crawl", {})
        use_browser = target_crawl_config.get("use_browser", False)
        crawler = Crawler(self.settings, base_url=_base_url, use_browser=use_browser)
        try:
            discovered = await crawler.discover_pages()
            if not discovered:
                logger.warning("No pages discovered!")
                await self.scan_repo.fail(scan.id)
                await self.session.commit()
                return scan

            crawled_pages = await crawler.crawl_all(discovered)
            self._last_crawled_pages = crawled_pages
        finally:
            await crawler.close()

        # 4. Save page scans
        page_records = []
        for cp in crawled_pages:
            ps = PageScan(
                scan_id=scan.id,
                url=cp.url,
                language=cp.language,
                title=cp.title,
                http_status=cp.http_status,
                load_time_ms=cp.load_time_ms,
                html_size_bytes=cp.html_size_bytes,
            )
            self.session.add(ps)
            page_records.append(ps)
        await self.session.flush()
        page_ids = {
            self._normalize_url(record.url): record.id
            for record in page_records
        }
        fallback_page_id = page_records[0].id

        # 5. Run all inspectors
        all_findings: list[RawFinding] = []

        # Prepare shared state for content quality inspector
        all_texts = [
            cp.html_content for cp in crawled_pages if cp.html_content
        ]

        # Initialize inspectors
        inspectors = self._create_inspectors()
        for insp in inspectors:
            try:
                await insp.setup()
            except Exception as e:
                logger.warning(f"Failed to setup {insp.inspector_name}: {e}")

        await self.scan_repo.set_phase(scan.id, "inspecting")
        await self.session.commit()

        # Set content texts for dedup
        crawled_urls = [cp.url for cp in crawled_pages]
        page_htmls = {cp.url: cp.html_content for cp in crawled_pages}
        sitemap_url = target_config.get(
            "sitemap_url",
            f"{_base_url}/sitemap.xml",
        )
        language_pairs = target_config.get("language_pairs", [])
        language_paths = target_config.get("language_paths", {"en": "/"})
        for insp in inspectors:
            if isinstance(insp, ContentQualityInspector):
                insp.set_all_texts(all_texts)
            if isinstance(insp, SitemapInspector):
                insp.set_crawled_urls(crawled_urls)
                insp.set_sitemap_url(sitemap_url)
                insp.set_language_paths(language_paths)
            if isinstance(insp, ContentGapDetector):
                insp.set_page_pairs(language_pairs)
                insp.set_page_htmls(page_htmls)
            if isinstance(insp, SEOInspector):
                insp.set_all_urls(crawled_urls)
                insp.set_target_languages(language_paths)
            if isinstance(insp, CannibalizationDetector):
                insp.set_page_data([
                    {"url": cp.url, "title": cp.title, "html_content": cp.html_content}
                    for cp in crawled_pages
                ])
            if isinstance(insp, EEATInspector):
                insp.set_crawled_urls(crawled_urls)
            if isinstance(insp, URLAuditor):
                insp.set_crawled_urls(crawled_urls)
            if isinstance(insp, CrawlBudgetInspector):
                insp.set_crawled_urls(crawled_urls)
                insp.set_page_data([
                    {"url": cp.url, "title": cp.title, "html_content": cp.html_content}
                    for cp in crawled_pages
                ])
                # Build incoming links map from page HTMLs for orphan detection
                incoming: dict[str, set[str]] = {}
                for cp in crawled_pages:
                    from urllib.parse import urljoin
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(cp.html_content or "", "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        if href.startswith("/") and not href.startswith("//"):
                            link_target = urljoin(cp.url, href).rstrip("/")
                            incoming.setdefault(link_target, set()).add(cp.url)
                insp.set_incoming_links(incoming)
            if isinstance(insp, CompetitorGapInspector):
                insp.set_page_data([
                    {"url": cp.url, "title": cp.title, "h1": ""}
                    for cp in crawled_pages
                ])

        # Run inspectors concurrently per page
        http_client = httpx.AsyncClient(timeout=15)
        semaphore = asyncio.Semaphore(self.settings.crawl_max_concurrent)

        async def inspect_page(cp) -> list[RawFinding]:
            findings: list[RawFinding] = []
            async with semaphore:
                for insp in inspectors:
                    try:
                        page_findings = await insp.inspect(
                            cp.url, cp.html_content, cp.headers
                        )
                        findings.extend(page_findings)
                    except Exception as e:
                        logger.error(f"{insp.inspector_name} failed on {cp.url}: {e}")
                        findings.append(RawFinding(
                            url=cp.url, inspector=insp.inspector_name,
                            category="inspector_error",
                            description=f"Inspector crashed: {str(e)[:200]}",
                        ))
            return findings

        tasks = [inspect_page(cp) for cp in crawled_pages]
        results = await asyncio.gather(*tasks)
        for r in results:
            all_findings.extend(r)

        # Teardown inspectors and collect cross-page findings
        for insp in inspectors:
            try:
                await insp.teardown()
            except Exception as e:
                logger.warning(f"Failed to teardown {insp.inspector_name}: {e}", exc_info=True)

            # Collect cross-page findings from cluster inspector
            if isinstance(insp, ContentClusterInspector):
                try:
                    cluster_findings = insp.get_findings()
                    all_findings.extend(cluster_findings)
                    logger.info(
                        f"ContentClusterInspector produced {len(cluster_findings)} "
                        f"cross-page findings"
                    )
                except Exception as e:
                    logger.warning(f"Failed to collect cluster findings: {e}")

            # Collect cross-page findings from link graph inspector
            if isinstance(insp, LinkGraphInspector):
                try:
                    graph_findings = insp.get_findings()
                    all_findings.extend(graph_findings)
                    logger.info(
                        f"LinkGraphInspector produced {len(graph_findings)} "
                        f"cross-page findings"
                    )
                except Exception as e:
                    logger.warning(f"Failed to collect link graph findings: {e}")

        await http_client.aclose()

        # 6. Save artifacts (Lighthouse JSON, screenshots)
        lighthouse_dir = self.settings.data_dir / "scans" / str(scan.id)
        lighthouse_dir.mkdir(parents=True, exist_ok=True)

        # 7. Save all issues
        new_issues = 0
        for finding in all_findings:
            # Deduplication check
            fingerprint = self.chroma.build_fingerprint(
                finding.url, finding.inspector, finding.category,
                finding.element, finding.description,
            )
            url_hash = hashlib.md5(finding.url.encode()).hexdigest()
            element_hash = hashlib.md5(
                (finding.element or "").encode()
            ).hexdigest()
            doc_id = self.chroma.build_doc_id(
                target.id, url_hash, finding.inspector, finding.category, element_hash
            )

            existing_id = await self.chroma.find_similar(fingerprint, target.id)
            if existing_id:
                # Update existing issue's last_seen
                await self.chroma.upsert_issue(
                    doc_id, fingerprint, target.id, finding.url,
                    finding.inspector, finding.category, 0.5, "active",
                )
                continue

            # Create new issue
            await self.chroma.upsert_issue(
                doc_id, fingerprint, target.id, finding.url,
                finding.inspector, finding.category, 0.5, "active",
            )

            # Build enriched description with metadata
            full_description = finding.description
            if finding.raw_metadata:
                import json as json_mod
                full_description += f"\n[metadata: {json_mod.dumps(finding.raw_metadata)}]"

            page_scan_id = page_ids.get(self._normalize_url(finding.url), fallback_page_id)
            issue = Issue(
                scan_id=scan.id,
                page_scan_id=page_scan_id,
                url=finding.url,
                inspector=finding.inspector,
                category=finding.category,
                title=finding.description[:200],
                description=full_description,
                element=finding.element,
                current_value=finding.current_value,
                suggested_value=finding.suggested_value,
                embedding_id=doc_id,
                status="open",
            )
            self.session.add(issue)
            new_issues += 1

        logger.info(f"Scan found {new_issues} new issues "
                    f"(total raw findings: {len(all_findings)})")

        # 8. Complete scan (only for standalone daily scans; quick-scan caller handles its own completion)
        if existing_scan_id is None:
            await self.scan_repo.complete(
                scan.id, len(crawled_pages), new_issues,
            )
        else:
            # Update ORM object so caller sees real counts
            scan.pages_crawled = len(crawled_pages)
            scan.total_issues_found = new_issues
        await self.audit_repo.log(
            "scan_completed", "scan", scan.id,
            {"pages": len(crawled_pages), "new_issues": new_issues,
             "total_findings": len(all_findings)},
        )

        await self.session.commit()
        # NOTE: Do NOT clear _last_crawled_pages here — engine.run_quick_scan()
        # reads it after run_full_scan() returns to build page_contents for
        # inline fixes. Clearing it causes all quick-scan fixes to be silently
        # skipped with "no content" errors.
        return scan

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL spelling for reliable PageScan association."""
        parts = urlsplit(url)
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))

    def _create_inspectors(self) -> list:
        http_client = httpx.AsyncClient(timeout=15, follow_redirects=False)
        target_config = self.settings.__class__.load_target(self.settings.target_name)
        competitor_urls = target_config.get("competitors", [])
        business_config = target_config.get("business", {})
        return [
            SEOInspector(),
            BrokenLinksInspector(client=http_client),
            CompetitorGapInspector(
                competitor_urls=competitor_urls,
                deepseek=self.deepseek,
                business_config=business_config,
            ),
            HeadersInspector(),
            CannibalizationDetector(),
            JSSeoInspector(),
            CrawlBudgetInspector(),
            EEATInspector(),
            ExternalReferencesInspector(target_config=target_config),
            AccessibilityInspector(),
            PerformanceInspector(
                lighthouse_path=self.settings.lighthouse_path,
                lighthouse_flags=self.settings.lighthouse_flags,
            ),
            MobileInspector(),
            ContentQualityInspector(
                ollama=self.ollama,
                prompt_manager=self.prompts,
            ),
            ContentFreshnessInspector(),
            SitemapInspector(),
            StructuredDataValidator(),  # field value validation is built-in, no AI required
            ContentGapDetector(),
            KeywordAnalyzer(deepseek=self.deepseek),
            ImageSEOInspector(ollama=self.ollama),
            URLAuditor(),
            RobotsTxtInspector(),
            PlatformSEOInspector(),
            # ── New AI-powered / graph inspectors ─────────────────
            SemanticContentInspector(deepseek=self.deepseek),
            ContentClusterInspector(ollama=self.ollama),
            LinkGraphInspector(),
        ]
