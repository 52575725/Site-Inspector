from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from config.settings import Settings
from src.core.engine import Engine
from src.core.scan_orchestrator import ScanOrchestrator
from src.storage.database import get_session_factory, init_db

logger = logging.getLogger(__name__)
console = Console()

scan_cmd = typer.Typer(name="scan", help="Run site inspection scans")
plan_cmd = typer.Typer(name="plan", help="Generate evidence-based optimization plans")
fix_cmd = typer.Typer(name="fix", help="Run auto-fixes")
verify_cmd = typer.Typer(name="verify", help="Check fix verifications")
report_cmd = typer.Typer(name="report", help="Generate reports")
schedule_cmd = typer.Typer(name="schedule", help="Manage scheduled jobs")
web_cmd = typer.Typer(name="web", help="Start the web dashboard")


def _create_engine() -> Engine:
    settings = Settings.load()
    factory = get_session_factory(settings)
    return Engine(settings, factory)


@scan_cmd.command("run")
def scan_run(
    target: str = typer.Option("helinsilver", help="Target name from config"),
    inspectors: str = typer.Option("all", help="Inspectors to run (comma-separated)"),
    pages: Optional[int] = typer.Option(None, help="Limit to N pages"),
    apply_fixes: bool = typer.Option(
        False, "--apply-fixes", help="Explicitly apply eligible fixes during the scan",
    ),
):
    """Run a full site inspection scan now."""
    async def _run():
        await init_db()
        engine = _create_engine()
        try:
            result = await engine.run_daily_scan(dry_run=not apply_fixes)
            console.print(f"[green]Scan completed![/green]")
            console.print(f"  Pages crawled: {result['pages']}")
            console.print(f"  Issues found: {result['new_issues']}")
            console.print(f"  Fixes applied: {result['fixes_applied']}")
            console.print(f"  Report: {result['report_path']}")
        finally:
            await engine.close()

    asyncio.run(_run())


@fix_cmd.command("run")
def fix_run(
    scan_id: Optional[int] = typer.Option(None, help="Fix issues from specific scan"),
    auto_only: bool = typer.Option(False, help="Only apply fully-auto (P0) fixes"),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Preview fixes unless --apply is explicitly set",
    ),
):
    """Analyze and auto-fix issues from a scan."""
    async def _run():
        await init_db()
        settings = Settings.load()
        factory = get_session_factory(settings)
        engine = Engine(settings, factory)

        async with factory() as session:
            from src.storage.repositories import ScanRepository
            scan_repo = ScanRepository(session)

            if scan_id:
                scan = await scan_repo.get_by_id(scan_id)
            else:
                from src.storage.repositories import TargetRepository
                target_repo = TargetRepository(session)
                target = await target_repo.get_by_name(settings.target_name)
                scan = await scan_repo.get_latest(target.id) if target else None

            if not scan:
                console.print("[red]No scan found. Run a scan first.[/red]")
                return

            from src.core.analyzer import Analyzer
            from src.core.fix_orchestrator import FixOrchestrator
            from src.agents.planning_context import PlanningContextCollector
            from src.agents.seo_planning_agent import SEOPlanningAgent

            analyzer = Analyzer(settings, session)
            issues = await analyzer.analyze_scan(scan.id, scan.pages_crawled)

            fixer = FixOrchestrator(settings, session, ollama=engine.ollama)
            planner = SEOPlanningAgent(fixer.fixers)
            context = await PlanningContextCollector(settings, session).collect(issues)
            plan = await planner.create_plan(
                issues,
                scan_id=scan.id,
                target_name=settings.target_name,
                context=context,
            )
            plan_path = (
                settings.data_dir / "reports" / "plans"
                / f"scan-{scan.id}-plan.json"
            )
            plan.write_json(plan_path)
            planned_issues = plan.select_issues(issues, dry_run=dry_run)
            fixes = await fixer.run_fixes(planned_issues, dry_run=dry_run)

            if dry_run:
                console.print(f"[yellow]Dry run: {len(fixes)} fixes would be applied[/yellow]")
            else:
                console.print(f"[green]{len(fixes)} fixes applied![/green]")
            console.print(f"  Plan: {plan_path}")
            console.print(
                f"  Approval required: "
                f"{sum(action.approval_required for action in plan.actions)} action(s)"
            )

            await session.commit()

        await engine.close()

    asyncio.run(_run())


@plan_cmd.command("generate")
def plan_generate(
    scan_id: Optional[int] = typer.Option(None, help="Plan issues from a specific scan"),
    objective: str = typer.Option(
        "Improve qualified organic visibility without unreviewed business changes",
        help="Narrative objective recorded in the plan",
    ),
    goal: Optional[str] = typer.Option(
        None,
        help="Override scoring goal: organic_visibility, qualified_inquiries, or technical_health",
    ),
    output: Optional[Path] = typer.Option(None, help="Output JSON path"),
    use_ai: bool = typer.Option(False, "--ai", help="Add bounded DeepSeek planning notes"),
):
    """Generate a read-only optimization plan from scan evidence."""
    async def _run():
        allowed_goals = {
            "organic_visibility",
            "qualified_inquiries",
            "technical_health",
        }
        if goal and goal not in allowed_goals:
            console.print(
                "[red]--goal must be organic_visibility, "
                "qualified_inquiries, or technical_health[/red]"
            )
            return

        await init_db()
        settings = Settings.load()
        factory = get_session_factory(settings)
        reasoner = None

        if use_ai:
            if not settings.deepseek_api_key:
                console.print("[red]SI_DEEPSEEK_API_KEY is required for --ai[/red]")
                return
            from src.ai.deepseek_client import DeepSeekClient
            reasoner = DeepSeekClient(
                api_key=settings.deepseek_api_key,
                model=settings.deepseek_model,
                timeout=settings.deepseek_timeout,
            )

        try:
            async with factory() as session:
                from src.agents.seo_planning_agent import SEOPlanningAgent
                from src.agents.planning_context import (
                    BusinessObjective,
                    PlanningContextCollector,
                )
                from src.core.analyzer import Analyzer
                from src.core.fix_orchestrator import FixOrchestrator
                from src.storage.repositories import ScanRepository, TargetRepository

                scan_repo = ScanRepository(session)
                if scan_id:
                    scan = await scan_repo.get_by_id(scan_id)
                else:
                    target_repo = TargetRepository(session)
                    target = await target_repo.get_by_name(settings.target_name)
                    scan = await scan_repo.get_latest(target.id) if target else None
                if not scan:
                    console.print("[red]No scan found. Run a scan first.[/red]")
                    return

                analyzer = Analyzer(settings, session)
                issues = await analyzer.analyze_scan(scan.id, scan.pages_crawled)
                fixer = FixOrchestrator(settings, session)
                planner = SEOPlanningAgent(fixer.fixers, reasoner=reasoner)
                context = await PlanningContextCollector(settings, session).collect(issues)
                if goal:
                    objective_data = context.objective.model_dump()
                    objective_data["goal"] = goal
                    context.objective = BusinessObjective.model_validate(objective_data)
                plan = await planner.create_plan(
                    issues,
                    scan_id=scan.id,
                    target_name=settings.target_name,
                    objective=objective,
                    context=context,
                    use_ai=use_ai,
                )
                destination = output or (
                    settings.data_dir / "reports" / "plans"
                    / f"scan-{scan.id}-plan.json"
                )
                plan.write_json(destination)
                await session.commit()

                table = Table(title=f"SEO Plan for scan {scan.id}")
                table.add_column("ID")
                table.add_column("Phase")
                table.add_column("Action")
                table.add_column("Risk")
                table.add_column("Score")
                table.add_column("Mode")
                table.add_column("Approval")
                for action in plan.actions:
                    table.add_row(
                        action.action_id,
                        str(action.phase),
                        action.title,
                        action.risk,
                        f"{action.decision_score:.2f}",
                        action.execution_mode,
                        "yes" if action.approval_required else "no",
                    )
                console.print(table)
                console.print(plan.executive_summary)
                console.print(f"[green]Plan written to {destination}[/green]")
        finally:
            if reasoner:
                await reasoner.close()

    asyncio.run(_run())


@verify_cmd.command("check")
def verify_check():
    """Check all pending verifications and trigger rollbacks if needed."""
    async def _run():
        await init_db()
        engine = _create_engine()
        try:
            result = await engine.run_verification_check()
            console.print(f"Verification check complete:")
            console.print(f"  Improved: {result.get('improved_count', 0)}")
            console.print(f"  Degraded: {result.get('degraded_count', 0)}")
            console.print(f"  Rollbacks: {result.get('rollback_count', 0)}")
        finally:
            await engine.close()

    asyncio.run(_run())


@report_cmd.command("generate")
def report_generate(
    report_type: str = typer.Option("daily", help="Report type: daily or weekly"),
    date: Optional[str] = typer.Option(None, help="Report date (YYYY-MM-DD)"),
):
    """Generate a daily or weekly report."""
    async def _run():
        await init_db()
        engine = _create_engine()
        try:
            if report_type == "weekly":
                result = await engine.run_weekly_report()
            else:
                result = await engine.run_daily_report()
            console.print(f"[green]Report generated: {result.get('report_path')}[/green]")
        finally:
            await engine.close()

    asyncio.run(_run())


@schedule_cmd.command("start")
def schedule_start():
    """Start the scheduler for recurring jobs."""
    async def _run():
        await init_db()
        settings = Settings.load()
        factory = get_session_factory(settings)
        engine = Engine(settings, factory)

        from src.scheduler.runner import SchedulerRunner
        runner = SchedulerRunner(settings, engine)
        runner.start()

        console.print("[green]Scheduler started. Press Ctrl+C to stop.[/green]")
        console.print(f"  Daily scan: {settings.daily_scan_time} UTC")
        console.print(f"  Weekly report: {settings.weekly_report_day} at {settings.weekly_report_time} UTC")

        try:
            await asyncio.Event().wait()  # Keep running
        except KeyboardInterrupt:
            runner.stop()
            await engine.close()

    asyncio.run(_run())


@schedule_cmd.command("list")
def schedule_list():
    """List all scheduled jobs."""
    async def _run():
        await init_db()
        settings = Settings.load()
        factory = get_session_factory(settings)
        engine = Engine(settings, factory)

        from src.scheduler.runner import SchedulerRunner
        runner = SchedulerRunner(settings, engine)
        jobs = runner.list_jobs()

        table = Table(title="Scheduled Jobs")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Next Run")
        for job in jobs:
            table.add_row(job["id"], job["name"], job["next_run"])
        console.print(table)

        await engine.close()

    asyncio.run(_run())


@web_cmd.command("start")
def web_start(
    host: str = typer.Option("127.0.0.1", help="Bind address (use 0.0.0.0 only with auth)"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the web dashboard server."""
    import uvicorn
    from src.web.app import create_app

    console.print("[bold blue]Site Inspector Dashboard[/bold blue]")
    console.print(f"  Starting at [bold]http://{host}:{port}[/bold]")
    console.print("  Press Ctrl+C to stop.")

    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")


@scan_cmd.command()
def submit_sitemap(
    sitemap_url: str = typer.Option(
        "https://www.helinsilver.com/sitemap.xml", help="Full URL of the sitemap",
    ),
    baidu_token: str = typer.Option("", help="Baidu linksubmit API token"),
):
    """Submit sitemap to Google and Bing for faster indexing."""
    from src.integrations.sitemap_submitter import SitemapSubmitter

    async def _submit():
        submitter = SitemapSubmitter()
        console.print(f"[bold]Submitting sitemap:[/bold] {sitemap_url}")
        results = await submitter.submit_all(
            sitemap_url=sitemap_url,
            site_url=sitemap_url.rsplit("/sitemap", 1)[0],
            baidu_token=baidu_token,
            baidu_urls=[sitemap_url] if baidu_token else None,
        )
        for engine, (ok, msg) in results.items():
            icon = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
            console.print(f"  {engine}: {icon} — {msg}")

    asyncio.run(_submit())
