from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.ai.ollama_client import OllamaClient
from src.fixers.alt_text_generator import AltTextGenerator
from src.fixers.base import BaseFixer
from src.fixers.breadcrumb_fixer import BreadcrumbFixer
from src.fixers.canonical_fixer import CanonicalFixer
from src.fixers.content_generator import ContentGenerator
from src.fixers.eeat_fixer import EEATFixer
from src.fixers.freshness_fixer import FreshnessFixer
from src.fixers.content_rewriter import ContentRewriter
from src.fixers.headers_fixer import HeadersFixer
from src.fixers.keywords_fixer import KeywordsFixer
from src.fixers.hreflang_fixer import HreflangFixer
from src.fixers.htag_restructurer import HTagRestructurer
from src.fixers.image_optimizer import ImageOptimizer
from src.fixers.jsonld_generator import JsonLdGenerator
from src.fixers.link_fixer import LinkFixer
from src.fixers.meta_fixer import MetaFixer
from src.fixers.mobile_css_fixer import MobileCssFixer
from src.fixers.og_image_fixer import OgImageFixer
from src.fixers.robots_txt_fixer import RobotsTxtFixer
from src.fixers.sitemap_fixer import SitemapFixer
from src.git.pr_template import generate_pr_body, generate_pr_title
from src.git.workflow import GitWorkflow
from src.core.fix_validator import validate_html, summarize_validation
from src.presentation.issue_explainer import describe_issue
from src.sources.base import BaseSource, resolve_within
from src.sources.http_source import HttpSource
from src.sources.local_source import LocalSource
from src.storage.models import Fix, Issue
from src.storage.repositories import (
    AuditLogRepository,
    FixRepository,
    IssueRepository,
)

logger = logging.getLogger(__name__)


class FixOrchestrator:
    """Orchestrates the auto-fix pipeline: generate → validate → git → PR."""

    def __init__(self, settings: Settings, session: AsyncSession,
                 ollama: Optional[OllamaClient] = None):
        self.settings = settings
        self.session = session
        self.ollama = ollama

        self.issue_repo = IssueRepository(session)
        self.fix_repo = FixRepository(session)
        self.audit_repo = AuditLogRepository(session)

        # Load target-specific config
        target_config = settings.__class__.load_target(settings.target_name)
        org = target_config.get("organization", {})
        org_name = org.get("name", "")
        org_alt = org.get("alternate_name", "")
        org_addr = org.get("address", {})
        org_address = {
            "@type": "PostalAddress",
            "streetAddress": org_addr.get("street_address", ""),
            "addressLocality": org_addr.get("locality", ""),
            "addressRegion": org_addr.get("region", ""),
            "addressCountry": org_addr.get("country", ""),
        } if org_addr else None
        domain = target_config.get("base_url", settings.target_base_url)
        languages = target_config.get("language_paths", {"en": "/"})
        slug_translations = target_config.get("slug_translations", {})
        og_image = target_config.get("default_og_image", "")
        og_w = target_config.get("default_og_image_width", 1200)
        og_h = target_config.get("default_og_image_height", 630)
        geo_config = target_config.get("geo", {})

        # Initialize fixers with target-specific config
        self.fixers: list[BaseFixer] = [
            MetaFixer(default_og_image=og_image,
                       default_og_width=str(og_w),
                       default_og_height=str(og_h),
                       deepseek_api_key=settings.deepseek_api_key,
                       deepseek_model=settings.deepseek_model),
            JsonLdGenerator(org_name=org_name, org_alt_name=org_alt,
                            org_address=org_address, domain=domain,
                            geo_lat=geo_config.get("latitude"),
                            geo_lon=geo_config.get("longitude")),
            AltTextGenerator(ollama=ollama),
            ImageOptimizer(),
            LinkFixer(),
            HreflangFixer(languages=languages),
            HTagRestructurer(ollama=ollama),
            ContentGenerator(ollama=ollama),
            ContentRewriter(ollama=ollama),
            MobileCssFixer(),
            OgImageFixer(default_image=og_image, default_width=str(og_w),
                          default_height=str(og_h)),
            SitemapFixer(language_paths=languages),
            RobotsTxtFixer(),
            BreadcrumbFixer(language_paths=languages,
                           slug_translations=slug_translations),
            HeadersFixer(),
            CanonicalFixer(),
            EEATFixer(),
            FreshnessFixer(),
            KeywordsFixer(),
        ]

    async def run_fixes(self, issues: Sequence[Issue],
                        dry_run: bool = False,
                        inline_contents: dict[str, str] | None = None,
                        output_dir: str | None = None,
                        source_override: BaseSource | None = None) -> list[Fix]:
        """Run auto-fixes on prioritized issues.

        Three modes:
        - inline_contents only: quick-scan without repo (suggestions only)
        - source_override: quick-scan WITH repo (real fixes + git + PR)
        - neither: daily scan with configured source

        When output_dir is provided with inline_contents, writes
        fixed files to that directory.
        """
        # Only fix P0 and P1
        fixable = [i for i in issues if i.priority_tier in ("P0", "P1")]
        if not fixable:
            logger.info("No fixable issues (P0/P1) found")
            return []

        # Cap max auto-fixes per scan
        max_fixes = self.settings.auto_fix_max_per_scan
        if len(fixable) > max_fixes:
            logger.warning(f"Capping fixes at {max_fixes} (found {len(fixable)})")
            fixable = fixable[:max_fixes]

        # Quick-scan fast path: use in-memory content, skip source/git
        if inline_contents is not None and source_override is None:
            return await self._run_fixes_inline(fixable, inline_contents, output_dir=output_dir)

        # Full path: source connection, git workflow, file I/O
        # When source_override is provided, caller has already prepared it
        if source_override is not None:
            source = source_override
            work_dir = source._work_dir if hasattr(source, '_work_dir') else None
            skip_git = True  # engine/caller handles git operations
        else:
            source = await self._create_source()
            try:
                await source.connect()
                work_dir = await source.sync()
            except Exception as e:
                logger.error(f"Failed to connect to source: {e}")
                return []
            skip_git = False

        # Try to set up git (only when we own the source)
        git = None
        branch = None
        is_git = False
        if not skip_git:
            git = GitWorkflow(self.settings, work_dir)
            is_git = await git.is_git_repo()

            if is_git and not dry_run:
                try:
                    scan_id = fixable[0].scan_id if fixable else 0
                    branch = await git.create_fix_branch(scan_id)
                except Exception as e:
                    logger.warning(f"Could not create git branch: {e}")
                    is_git = False

        # Process each issue — cache page content so fixers chain on same file
        applied_fixes: list[Fix] = []
        changed_files: set[str] = set()
        page_cache: dict[str, str] = {}
        skip_reasons: dict[str, int] = {}

        # ── Pre-fix snapshot: save original content BEFORE any fixer runs ──
        original_snapshots: dict[str, str] = {}
        for issue in fixable:
            file_path = await self._url_to_file_path(issue.url, source)
            if not file_path or file_path in original_snapshots:
                continue
            try:
                page_cache[file_path] = await source.read_file(file_path)
                original_snapshots[file_path] = page_cache[file_path]
            except FileNotFoundError:
                logger.debug(f"Snapshot: file not found '{file_path}', will skip fixes on it")
        logger.debug(
            f"Prefetched {len(original_snapshots)} file snapshots "
            f"for {len(fixable)} fixable issues"
        )

        # ── Group issues by file path so all fixes for a page ──
        #     are chained through a single BeautifulSoup instance.
        #     This prevents HTML degradation from repeated parse→serialize cycles.
        MAX_FIXES_PER_FILE = 3  # Tightened: fewer chained fixers reduce corruption risk
        page_groups: dict[str, list[tuple]] = {}  # file_path → [(issue, fixer), ...]

        for issue in fixable:
            fixer = self._find_fixer(issue.category)
            if not fixer:
                reason = f"no fixer for category '{issue.category}'"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.info(f"Skipping issue #{issue.id} ({issue.url}): {reason}")
                continue

            file_path = await self._url_to_file_path(issue.url, source)
            if not file_path:
                reason = f"no file_path for URL '{issue.url}'"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            if file_path not in original_snapshots:
                reason = f"file not found in repo: '{file_path}'"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.info(f"Skipping issue #{issue.id} ({issue.url}): {reason}")
                continue

            if file_path not in page_groups:
                page_groups[file_path] = []
            page_groups[file_path].append((issue, fixer))

        # Process each page — chain fixers on a single content string
        for file_path, items in page_groups.items():
            # Enforce per-file cap
            if len(items) > MAX_FIXES_PER_FILE:
                reason = (
                    f"max fixes ({MAX_FIXES_PER_FILE}) reached for '{file_path}' "
                    f"({len(items)} issues, capping)"
                )
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                items = items[:MAX_FIXES_PER_FILE]

            # Start from the original snapshot — every fixer on this page
            # chains from the previous fixer's after_content
            page_content = original_snapshots[file_path]

            for issue, fixer in items:
                issue_dict = {
                    "id": issue.id,
                    "category": issue.category,
                    "url": issue.url,
                    "element": issue.element,
                    "description": issue.description,
                    "file_path": file_path,
                    "before_content": page_content,
                }

                try:
                    result = await fixer.generate_fix(issue_dict, source, page_content)
                except Exception as e:
                    reason = f"fixer '{fixer.fixer_name}' crashed on {issue.url}: {e}"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    logger.error(reason)
                    continue

                if not result.success:
                    reason = (
                        f"fixer '{fixer.fixer_name}' returned success=False "
                        f"for category '{issue.category}' on {issue.url}"
                    )
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    logger.warning(reason)
                    continue

                # Chain: next fixer sees this fixer's output
                page_content = result.after_content

                # Record fix
                explanation = describe_issue(issue.category, issue.description or "")
                fix_status = "proposed" if dry_run else "applied"
                fix = await self.fix_repo.create(
                    issue_id=issue.id,
                    scan_id=issue.scan_id,
                    fixer=fixer.fixer_name,
                    fix_type=fixer.fix_type,
                    status=fix_status,
                    plain_summary=explanation["summary"],
                    impact_explanation=explanation["impact"],
                    change_explanation=explanation["action"],
                    risk_level=explanation["risk"],
                    file_path=file_path,
                    before_content=result.before_content,
                    after_content=result.after_content,
                    diff=result.diff,
                    applied_at=None if dry_run else datetime.utcnow(),
                )

                await self.issue_repo.update_status(issue.id, fix_status)
                issue.fix_applied_at = None if dry_run else datetime.utcnow()
                applied_fixes.append(fix)
                logger.info(
                    f"Applied {fixer.fix_type} fix: {fixer.fixer_name} -> {issue.url}"
                )

            # Write to disk ONCE after all fixers for this page are done
            page_cache[file_path] = page_content
            if not dry_run and page_content != original_snapshots.get(file_path, ""):
                await source.write_file(file_path, page_content)
                changed_files.add(file_path)

        # Commit and create PR (only when we own the source)
        if is_git and changed_files and not dry_run:
            # ── Post-fix validation: rollback corrupted files ──────
            # Use pre-fix snapshots (saved BEFORE any fixer modified the files)
            validated_files = set()
            failed_files = set()
            # Re-read current on-disk state for newly created files
            for fp in changed_files:
                if fp not in original_snapshots:
                    try:
                        original_snapshots[fp] = await source.read_file(fp)
                    except Exception:
                        pass
            for fp in changed_files:
                if fp in page_cache and fp in original_snapshots:
                    result = validate_html(fp, original_snapshots[fp], page_cache[fp])
                    if result.passed:
                        validated_files.add(fp)
                    else:
                        failed_files.add(fp)
                        logger.error(f"Rolling back {fp}: {'; '.join(result.errors)}")
                        # Restore original
                        try:
                            await source.write_file(fp, original_snapshots[fp])
                            page_cache[fp] = original_snapshots[fp]
                        except Exception:
                            pass
                else:
                    validated_files.add(fp)  # new files are fine

            if failed_files:
                logger.warning(
                    f"Validation: {len(validated_files)} passed, "
                    f"{len(failed_files)} rolled back: {', '.join(sorted(failed_files))}"
                )
                changed_files = validated_files
                # Remove fixes for failed files
                applied_fixes = [
                    f for f in applied_fixes
                    if f.file_path not in failed_files
                ]

            if not changed_files:
                logger.warning("All changed files failed validation — aborting commit")
                return applied_fixes

            auto_count = sum(1 for f in applied_fixes if f.fix_type == "fully_auto")
            pr_title = generate_pr_title(
                fixable[0].scan_id, len(applied_fixes), auto_count,
            )
            fixes_data = [
                {
                    "category": i.category,
                    "description": i.description,
                    "url": i.url,
                    "fix_type": f.fix_type,
                    "diff": f.diff,
                }
                for i, f in zip(fixable, applied_fixes) if f.diff
            ]
            pr_body = generate_pr_body(fixes_data, fixable[0].scan_id,
                                       self.settings.target_name)

            try:
                commit_msg = (
                    f"fix: apply {len(applied_fixes)} automated site fixes "
                    f"({auto_count} auto, {len(applied_fixes) - auto_count} semi-auto)"
                )
                commit_hash = await git.stage_and_commit(
                    list(changed_files), commit_msg,
                )

                if commit_hash:
                    pushed = await git.push_branch(branch)
                    if pushed:
                        pr_url = await git.create_pr(branch, pr_title, pr_body)
                        if pr_url:
                            for fix in applied_fixes:
                                await self.fix_repo.mark_pr_created(fix.id, pr_url)
                            await self.audit_repo.log(
                                "pr_created", "fix",
                                details={
                                    "branch": branch, "pr_url": pr_url,
                                    "fix_count": len(applied_fixes),
                                    "auto_count": auto_count,
                                },
                            )
                        else:
                            logger.warning(
                                "Branch pushed but PR creation failed. "
                                "Check gh auth status and repo permissions."
                            )
            except Exception as e:
                logger.error(f"Git workflow error: {e}")

        if skip_reasons:
            logger.warning(f"Fix summary: {len(applied_fixes)} applied, "
                           f"{sum(skip_reasons.values())} skipped. Reasons: {dict(skip_reasons)}")
        if not skip_git:
            await source.disconnect()
        await self.session.commit()
        return applied_fixes

    async def _run_fixes_inline(self, fixable: list, page_contents: dict[str, str],
                                output_dir: str | None = None) -> list[Fix]:
        """Quick-scan fix path: generate fixes from in-memory content only.

        When output_dir is provided, writes fixed files to that directory.
        """
        applied_fixes: list[Fix] = []
        page_cache: dict[str, str] = {}
        skip_reasons: dict[str, int] = {}

        for issue in fixable:
            fixer = self._find_fixer(issue.category)
            if not fixer:
                reason = f"no fixer for category '{issue.category}'"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            file_path = await self._url_to_file_path(issue.url)
            if not file_path:
                reason = f"no file_path for URL '{issue.url}'"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            # Look up content: try URL key first, then file_path
            if file_path not in page_cache:
                # Also try URL variants (with/without trailing slash)
                url_variants = [
                    issue.url,
                    issue.url.rstrip("/"),
                    issue.url.rstrip("/") + "/",
                ]
                content = ""
                for variant in url_variants:
                    content = page_contents.get(variant, "")
                    if content:
                        break
                if not content:
                    content = page_contents.get(file_path, "")
                if not content:
                    reason = f"no content for '{file_path}' (url={issue.url})"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue
                page_cache[file_path] = content

            page_content = page_cache[file_path]

            issue_dict = {
                "id": issue.id,
                "category": issue.category,
                "url": issue.url,
                "element": issue.element,
                "description": issue.description,
                "file_path": file_path,
                "before_content": page_content,
            }

            try:
                result = await fixer.generate_fix(issue_dict, None, page_content)
            except Exception as e:
                reason = f"fixer '{fixer.fixer_name}' crashed on {issue.url}: {e}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.error(reason)
                continue

            if not result.success:
                reason = f"fixer '{fixer.fixer_name}' returned success=False for category '{issue.category}' on {issue.url}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.warning(reason)
                continue

            page_cache[file_path] = result.after_content

            explanation = describe_issue(issue.category, issue.description or "")
            writes_files = output_dir is not None
            fix_status = "applied" if writes_files else "proposed"
            fix = await self.fix_repo.create(
                issue_id=issue.id,
                scan_id=issue.scan_id,
                fixer=fixer.fixer_name,
                fix_type=fixer.fix_type,
                status=fix_status,
                plain_summary=explanation["summary"],
                impact_explanation=explanation["impact"],
                change_explanation=explanation["action"],
                risk_level=explanation["risk"],
                file_path=file_path,
                before_content=result.before_content,
                after_content=result.after_content,
                diff=result.diff,
                applied_at=datetime.utcnow() if writes_files else None,
            )

            await self.issue_repo.update_status(issue.id, fix_status)
            issue.fix_applied_at = datetime.utcnow() if writes_files else None
            applied_fixes.append(fix)
            logger.info(f"Suggested {fixer.fix_type} fix: {fixer.fixer_name} -> {issue.url}")

        if skip_reasons:
            logger.warning(
                f"Inline fix summary: {len(applied_fixes)} applied, "
                f"{sum(skip_reasons.values())} skipped. "
                f"Top reasons: {dict(sorted(skip_reasons.items(), key=lambda x: -x[1])[:5])}"
            )

        # Write fixed files to output_dir if requested
        if output_dir and page_cache:
            root = Path(output_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            for file_path, content in page_cache.items():
                full = resolve_within(root, file_path)
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")
            logger.info(f"Wrote {len(page_cache)} fixed files to {output_dir}")

        await self.session.commit()
        return applied_fixes

    def _find_fixer(self, category: str) -> Optional[BaseFixer]:
        for fixer in self.fixers:
            if fixer.can_fix(category):
                return fixer
        return None

    async def _create_source(self) -> BaseSource:
        # Check target-specific config first
        target_config = self.settings.load_target(self.settings.target_name)
        source_type = target_config.get("source", {}).get("type", self.settings.source_type)
        local_path = target_config.get("source", {}).get("local_path")

        if source_type == "local" and local_path:
            from pathlib import Path
            full_path = Path(local_path)
            if not full_path.is_absolute():
                full_path = Path(local_path)
            return LocalSource(full_path)
        if source_type == "http":
            return HttpSource(self.settings)
        return HttpSource(self.settings)

    @staticmethod
    async def _url_to_file_path(url: str, source: BaseSource | None = None) -> str | None:
        from pathlib import PurePosixPath
        from urllib.parse import unquote, urlparse

        raw_path = unquote(urlparse(url).path)
        if "\\" in raw_path or "\x00" in raw_path:
            logger.warning(f"Rejected unsafe URL path: {raw_path!r}")
            return None
        path_obj = PurePosixPath(raw_path)
        if ".." in path_obj.parts:
            logger.warning(f"Rejected URL path traversal: {raw_path!r}")
            return None
        path = str(path_obj).strip("/")
        if not path:
            return "index.html"
        if "." not in path.split("/")[-1]:
            path = path.rstrip("/") + "/index.html"

        # If no source to search, return the mapped path as-is
        if source is None:
            return path

        # Try exact path first
        try:
            await source.read_file(path)
            return path
        except (FileNotFoundError, RuntimeError):
            pass

        # Search for matching files: try common web file patterns
        slug = path.rsplit("/", 1)[-1].replace(".html", "").replace(".htm", "")
        slugs = {slug}
        if "-" in slug:
            slugs.add(slug.replace("-", ""))

        try:
            all_files = await source.list_files("**/*.*")
        except Exception:
            return path

        # Score candidates by slug match
        web_exts = {".html", ".htm", ".md", ".mdx", ".jsx", ".tsx", ".vue", ".svelte", ".php", ".astro"}
        best = None
        best_score = 0
        for candidate in all_files:
            c = candidate.replace("\\", "/")
            ext = c.rsplit(".", 1)[-1] if "." in c else ""
            base_name = c.rsplit("/", 1)[-1].rsplit(".", 1)[0] if "." in c.rsplit("/", 1)[-1] else c.rsplit("/", 1)[-1]

            score = 0
            if base_name in slugs:
                score = 10
            elif any(s in base_name for s in slugs):
                score = 5
            elif slug in c:
                score = 3

            if f".{ext}" in web_exts:
                score += 2
            if "/index." in c and base_name == "index":
                score += 1

            if score > best_score:
                best_score = score
                best = c

        if best and best_score >= 5:
            logger.info(f"File path '{path}' not found, mapped to '{best}' (score={best_score})")
            return best

        logger.warning(f"Could not find matching file for URL path '{path}' in repo")
        return path
