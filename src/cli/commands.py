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
):
    """Run a full site inspection scan now."""
    async def _run():
        await init_db()
        engine = _create_engine()
        try:
            result = await engine.run_daily_scan()
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
    dry_run: bool = typer.Option(False, help="Generate fixes without applying"),
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

            analyzer = Analyzer(settings, session)
            issues = await analyzer.analyze_scan(scan.id, scan.pages_crawled)

            fixer = FixOrchestrator(settings, session, ollama=engine.ollama)
            fixes = await fixer.run_fixes(issues, dry_run=dry_run)

            if dry_run:
                console.print(f"[yellow]Dry run: {len(fixes)} fixes would be applied[/yellow]")
            else:
                console.print(f"[green]{len(fixes)} fixes applied![/green]")

            await session.commit()

        await engine.close()

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
    host: str = typer.Option("0.0.0.0", help="Bind address"),
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
