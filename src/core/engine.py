from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings
from src.ai.deepseek_client import DeepSeekClient
from src.ai.ollama_client import OllamaClient
from src.core.analyzer import Analyzer
from src.core.fix_orchestrator import FixOrchestrator
from src.core.scan_orchestrator import ScanOrchestrator
from src.core.verifier import Verifier
from src.reporters.daily_report import DailyReportGenerator
from src.reporters.weekly_report import WeeklyReportGenerator
from src.storage.repositories import (
    FixRepository,
    IssueRepository,
    ScanRepository,
    TargetRepository,
)

logger = logging.getLogger(__name__)


class Engine:
    """Main orchestrator for daily and weekly inspection routines."""

    def __init__(self, settings: Settings, session_factory):
        self.settings = settings
        self.session_factory = session_factory
        self._ollama: Optional[OllamaClient] = None
        self._deepseek: Optional[DeepSeekClient] = None

    @property
    def ollama(self) -> OllamaClient:
        if self._ollama is None:
            self._ollama = OllamaClient(self.settings)
        return self._ollama

    @property
    def deepseek(self) -> DeepSeekClient | None:
        if not self.settings.deepseek_enabled or not self.settings.deepseek_api_key:
            return None
        if self._deepseek is None:
            self._deepseek = DeepSeekClient(
                api_key=self.settings.deepseek_api_key,
                model=self.settings.deepseek_model,
                timeout=self.settings.deepseek_timeout,
            )
        return self._deepseek

    async def close(self) -> None:
        if self._ollama:
            await self._ollama.close()
        if self._deepseek:
            await self._deepseek.close()

    async def run_daily_scan(self, *, dry_run: bool = True) -> dict:
        """Execute a full daily scan: detect → analyze → fix → report.

        When dry_run=True, fixes are only suggested (not written to disk).
        Dry-run is the default; callers must explicitly opt into writes.
        """
        logger.info("=== Starting daily scan ===")

        async with self.session_factory() as session:
            # Phase 1: Detect
            orchestrator = ScanOrchestrator(
                self.settings, session, ollama=self.ollama,
                deepseek=self.deepseek,
            )
            scan = await orchestrator.run_full_scan()

            # Phase 2: Analyze
            analyzer = Analyzer(self.settings, session)
            issues = await analyzer.analyze_scan(scan.id, scan.pages_crawled)

            # Phase 3: Plan, then propose or execute only plan-approved issues.
            plan = None
            if issues:
                fixer = FixOrchestrator(
                    self.settings, session, ollama=self.ollama,
                )
                from src.agents.planning_context import PlanningContextCollector
                from src.agents.seo_planning_agent import SEOPlanningAgent

                planner = SEOPlanningAgent(fixer.fixers)
                context = await PlanningContextCollector(
                    self.settings, session,
                ).collect(issues)
                plan = await planner.create_plan(
                    issues,
                    scan_id=scan.id,
                    target_name=self.settings.target_name,
                    context=context,
                )
                plan_path = (
                    self.settings.data_dir / "reports" / "plans"
                    / f"scan-{scan.id}-plan.json"
                )
                plan.write_json(plan_path)
                planned_issues = plan.select_issues(issues, dry_run=dry_run)
                fixes = await fixer.run_fixes(planned_issues, dry_run=dry_run)
            else:
                fixes = []
                plan_path = None

            # Phase 4: Verification setup
            verifier = Verifier(self.settings, session)
            applied_fixes = [fix for fix in fixes if fix.status == "applied"]
            if applied_fixes:
                await verifier.create_verifications(applied_fixes)

            # Generate report
            report_gen = DailyReportGenerator(self.settings.data_dir / "reports")
            report_path = report_gen.generate(
                self.settings.target_name, scan, issues, fixes,
                target_url=self.settings.target_base_url,
            )

            console_summary = report_gen.generate_console(
                self.settings.target_name, scan, issues, fixes,
            )

            await session.commit()

        logger.info(f"Daily scan complete. Report: {report_path}")
        return {
            "scan_id": scan.id,
            "pages": scan.pages_crawled,
            "new_issues": scan.total_issues_found,
            "fixes_applied": len(fixes),
            "report_path": str(report_path),
            "plan_path": str(plan_path) if plan_path else None,
            "planned_actions": len(plan.actions) if plan else 0,
            "actions_requiring_approval": (
                sum(action.approval_required for action in plan.actions) if plan else 0
            ),
        }

    async def run_quick_scan(self, url: str, name: str | None = None,
                            scan_id: int | None = None,
                            repo_url: str | None = None,
                            repo_branch: str = "main",
                            push_changes: bool = True,
                            require_approval: bool = True) -> dict:
        """On-demand scan for an arbitrary URL.

        When repo_url is provided, clones the repo and applies real fixes
        (creates branch, commits, pushes, opens PR).
        When repo_url is None, generates fix suggestions from crawled HTML only.
        """
        from pathlib import Path
        from urllib.parse import urlparse

        from src.sources.git_source import GitSource

        if not name:
            parsed = urlparse(url)
            name = parsed.netloc.replace("www.", "").replace(".", "-")

        has_repo = bool(repo_url and repo_url.strip())
        logger.info(f"=== Starting quick scan for {url} (repo={has_repo}) ===")

        async with self.session_factory() as session:
            scan_repo = ScanRepository(session)

            # Phase 1: Crawl + Inspect
            orchestrator = ScanOrchestrator(self.settings, session, ollama=self.ollama, deepseek=self.deepseek)
            scan = await orchestrator.run_full_scan(
                target_name=name,
                target_base_url=url,
                target_languages=["en"],
                existing_scan_id=scan_id,
            )

            if scan.status == "failed":
                logger.warning(f"Scan failed for {url}: no pages discovered")
                await session.commit()
                return {
                    "scan_id": scan.id,
                    "pages": 0,
                    "new_issues": 0,
                    "fixes_suggested": 0,
                    "report_path": "",
                }

            await scan_repo.set_phase(scan.id, "analyzing")
            await session.commit()

            # Phase 2: Analyze
            analyzer = Analyzer(self.settings, session)
            issues = await analyzer.analyze_scan(scan.id, scan.pages_crawled)

            await scan_repo.set_phase(scan.id, "fixing")
            await session.commit()

            # Phase 3: Fixes (inline suggestions OR real source-based fixes)
            pr_url_result = None
            if issues:
                page_contents: dict[str, str] = {}
                for cp in orchestrator._last_crawled_pages:
                    if cp.html_content:
                        page_contents[cp.url] = cp.html_content

                fixer = FixOrchestrator(self.settings, session, ollama=self.ollama)

                if has_repo:
                    # Real fixes: clone repo, fix source files, commit, push PR
                    work_dir = Path("data/site_sources") / f"quick_{scan.id}"
                    source = GitSource(repo_url.strip(), repo_branch, work_dir=work_dir)
                    try:
                        await source.connect()
                        await source.sync()
                        logger.info(f"Cloned {repo_url} to {work_dir}")
                    except Exception as e:
                        err_msg = f"Failed to clone repo: {e}"
                        logger.error(err_msg)
                        await scan_repo.set_fix_error(scan.id, err_msg)
                        await session.commit()
                        fixes = []
                    else:
                        # When push_changes is True, skip approval and apply fixes directly
                        fixes = await fixer.run_fixes(
                            issues, dry_run=False, source_override=source,
                        )
                        if fixes:
                            from src.git.workflow import GitWorkflow
                            git = GitWorkflow(self.settings, work_dir)
                            if await git.is_git_repo():
                                try:
                                    branch = await git.create_fix_branch(scan.id)
                                    for fix in fixes:
                                        fix.git_branch = branch
                                    changed = await git._run_git("diff", "--name-only")
                                    if changed:
                                        files = [f.strip() for f in changed.split("\n") if f.strip()]
                                        if files:
                                            auto_count = sum(1 for f in fixes if f.fix_type == "fully_auto")
                                            title = f"[Site Inspector] Auto-fix {len(fixes)} issues ({auto_count} auto)"
                                            body = f"Auto-generated fixes from scan #{scan.id}\n\n"
                                            body += "\n".join(
                                                f"- [{f.fixer}] {f.file_path}" for f in fixes[:20]
                                            )
                                            await git.stage_and_commit(files, title)
                                            if push_changes:
                                                if await git.push_branch(branch):
                                                    pr_url_result = await git.create_pr(branch, title, body)
                                                    if pr_url_result:
                                                        await scan_repo.set_pr_url(scan.id, pr_url_result)
                                                        for fix in fixes:
                                                            fix.git_pr_url = pr_url_result
                                                        logger.info(f"PR created: {pr_url_result}")
                                                    else:
                                                        await scan_repo.set_fix_error(
                                                            scan.id,
                                                            f"分支 '{branch}' 已推送，但创建 PR 时 GitHub API 超时。"
                                                            "网络恢复后系统会自动重试，或手动在 GitHub 上创建 PR。"
                                                        )
                                                else:
                                                    await scan_repo.set_fix_error(scan.id, "Push failed — changes committed locally on branch: " + branch)
                                            else:
                                                logger.info(f"Changes committed locally on branch: {branch} (not pushed)")
                                        else:
                                            await scan_repo.set_fix_error(scan.id, "No changed files after fix")
                                    else:
                                        await scan_repo.set_fix_error(scan.id, "No file changes detected")
                                except Exception as e:
                                    await scan_repo.set_fix_error(scan.id, f"Git/PR failed: {e}")
                            else:
                                await scan_repo.set_fix_error(scan.id, "Cloned dir is not a git repo")
                        else:
                            await scan_repo.set_fix_error(scan.id, "No fixable issues matched source files")
                        await session.commit()
                else:
                    # Inline suggestions from crawled HTML
                    fixes = await fixer.run_fixes(
                        issues, dry_run=True, inline_contents=page_contents,
                    )
            else:
                fixes = []

            await scan_repo.set_phase(scan.id, "reporting")
            await session.commit()

            # Phase 4: Report
            report_gen = DailyReportGenerator(self.settings.data_dir / "reports")
            report_path = report_gen.generate(
                name, scan, issues, fixes, target_url=url,
            )
            logger.info(report_gen.generate_console(name, scan, issues, fixes))

            await scan_repo.set_phase(scan.id, "done")
            await scan_repo.complete(scan.id, scan.pages_crawled, scan.total_issues_found)
            await session.commit()

        logger.info(f"Quick scan complete for {url}. Report: {report_path}")
        result = {
            "scan_id": scan.id,
            "pages": scan.pages_crawled,
            "new_issues": scan.total_issues_found,
            "fixes_suggested": len(fixes),
            "report_path": str(report_path),
        }

        # Send desktop notification
        from src.notifications.notifier import Notifier
        notifier = Notifier(self.settings)
        await notifier.notify_scan_complete(
            target_name=name,
            pages=result["pages"],
            issues=result["new_issues"],
            fixes=result["fixes_suggested"],
            pr_url=scan.pr_url,
        )

        return result

    async def run_daily_report(self) -> dict:
        """Generate a daily report from the latest scan."""
        logger.info("=== Generating daily report ===")

        async with self.session_factory() as session:
            target_repo = TargetRepository(session)
            target = await target_repo.get_by_name(self.settings.target_name)
            if not target:
                logger.warning("No target found, run a scan first")
                return {"report_path": None, "error": "No target found"}

            scan_repo = ScanRepository(session)
            scan = await scan_repo.get_latest(target.id)
            if not scan:
                logger.warning("No scans found, run a scan first")
                return {"report_path": None, "error": "No scans found"}

            issue_repo = IssueRepository(session)
            fix_repo = FixRepository(session)
            issues = await issue_repo.get_by_scan(scan.id)
            fixes = await fix_repo.get_by_scan(scan.id)

            report_gen = DailyReportGenerator(self.settings.data_dir / "reports")
            report_path = report_gen.generate(
                self.settings.target_name, scan, issues, fixes,
                target_url=self.settings.target_base_url,
            )
            console_summary = report_gen.generate_console(
                self.settings.target_name, scan, issues, fixes,
            )
            logger.info(console_summary)

        logger.info(f"Daily report generated: {report_path}")
        return {"report_path": str(report_path)}

    async def run_weekly_report(self) -> dict:
        """Generate a weekly trend report."""
        logger.info("=== Generating weekly report ===")

        async with self.session_factory() as session:
            target_repo = TargetRepository(session)
            target = await target_repo.get_by_name(self.settings.target_name)

            if not target:
                logger.warning("No target found, running daily scan first")
                return await self.run_daily_scan()

            report_gen = WeeklyReportGenerator(
                session, self.settings.data_dir / "reports",
            )
            report_path = await report_gen.generate(target.name, target.id)

        logger.info(f"Weekly report generated: {report_path}")
        return {"report_path": str(report_path)}

    async def run_verification_check(self) -> dict:
        """Check pending verifications and trigger rollbacks if needed."""
        logger.info("=== Running verification check ===")

        async with self.session_factory() as session:
            verifier = Verifier(self.settings, session)
            result = await verifier.check_pending_verifications()
            await session.commit()

        return result

    async def run_sitemap_submit(self) -> dict:
        """Submit sitemap to Google and Bing search engines."""
        logger.info("=== Submitting sitemap to search engines ===")
        from src.integrations.sitemap_submitter import SitemapSubmitter

        target_config = self.settings.__class__.load_target(self.settings.target_name)
        base_url = target_config.get("base_url") or self.settings.target_base_url
        sitemap_url = target_config.get("sitemap_url") or f"{base_url.rstrip('/')}/sitemap.xml"

        submitter = SitemapSubmitter()
        gsc_property = self.settings.gsc_property or ""
        results = await submitter.submit_all(
            sitemap_url=sitemap_url, site_url=base_url,
            gsc_property=gsc_property,
        )

        summary = {}
        for engine, (ok, msg) in results.items():
            summary[engine] = {"success": ok, "message": msg}
            logger.info(f"  {engine}: {'OK' if ok else 'FAIL'} — {msg}")

        return {"sitemap_url": sitemap_url, "results": summary}
