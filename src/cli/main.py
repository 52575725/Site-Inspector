from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

from src.cli.commands import (
    fix_cmd,
    report_cmd,
    scan_cmd,
    schedule_cmd,
    verify_cmd,
    web_cmd,
)
from src.storage.database import init_db

console = Console()

app = typer.Typer(
    name="site-inspector",
    help="Website Intelligent Inspection & Optimization Agent",
    no_args_is_help=True,
)

app.add_typer(scan_cmd, name="scan")
app.add_typer(fix_cmd, name="fix")
app.add_typer(verify_cmd, name="verify")
app.add_typer(report_cmd, name="report")
app.add_typer(schedule_cmd, name="schedule")
app.add_typer(web_cmd, name="web")


@app.command()
def init():
    """First-time setup wizard."""
    console.print("[bold blue]Site Inspector - First Time Setup[/bold blue]")
    console.print()

    # Check dependencies
    console.print("Checking dependencies...")

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 12):
        console.print(f"  [green][OK][/green] Python {py_ver}")
    else:
        console.print(f"  [red][FAIL][/red] Python {py_ver} (need 3.12+)")

    # Ollama
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            console.print("  [green][OK][/green] Ollama server")
        else:
            console.print("  [yellow][WARN][/yellow] Ollama not running")
    except Exception:
        console.print("  [yellow][WARN][/yellow] Ollama not available")

    # Node.js (for Lighthouse and axe-core)
    import shutil
    if shutil.which("node"):
        console.print("  [green][OK][/green] Node.js")
    else:
        console.print("  [yellow][WARN][/yellow] Node.js not found (Lighthouse/axe-core will be limited)")

    # Init database
    from config.settings import Settings
    settings = Settings.load()
    asyncio.run(init_db(settings))
    console.print("  [green][OK][/green] Database initialized")

    # Create data directories
    for subdir in ["scans", "screenshots", "reports", "site_sources"]:
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)
    console.print("  [green][OK][/green] Data directories created")

    console.print()
    console.print("[bold green]Setup complete![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Configure Google APIs: site-inspector config --set google_credentials_path=...")
    console.print("  2. Test a scan: site-inspector scan run")
    console.print("  3. Enable scheduling: site-inspector schedule start")


@app.command()
def status(
    target: str = typer.Option("helinsilver", help="Target name"),
):
    """Show current inspection status."""
    async def _run():
        await init_db()
        from config.settings import Settings
        from src.storage.database import get_session_factory
        from src.storage.repositories import ScanRepository, TargetRepository

        settings = Settings.load()
        factory = get_session_factory(settings)
        async with factory() as session:
            target_repo = TargetRepository(session)
            scan_repo = ScanRepository(session)

            t = await target_repo.get_by_name(target)
            if not t:
                console.print(f"[red]Target '{target}' not configured.[/red]")
                console.print("Run 'site-inspector init' first.")
                return

            latest = await scan_repo.get_latest(t.id)
            if latest:
                console.print(f"[bold]Target:[/bold] {t.name} ({t.base_url})")
                console.print(f"[bold]Last scan:[/bold] {latest.started_at} ({latest.scan_type})")
                console.print(f"  Pages: {latest.pages_crawled}")
                console.print(f"  Issues: {latest.total_issues_found}")
                console.print(f"  Status: {latest.status}")
            else:
                console.print(f"[bold]Target:[/bold] {t.name} ({t.base_url})")
                console.print("[yellow]No scans yet. Run 'site-inspector scan run' first.[/yellow]")

    asyncio.run(_run())


@app.command()
def config(
    show: bool = typer.Option(False, help="Show current configuration"),
    set_value: str = typer.Option(None, "--set", help="Set a config value (key=value)"),
):
    """View and edit configuration."""
    if show:
        from config.settings import Settings
        s = Settings.load()
        console.print("[bold]Current Configuration:[/bold]")
        for field, value in s.model_dump().items():
            if not field.startswith("_"):
                console.print(f"  {field} = {value}")

    if set_value:
        if "=" not in set_value:
            console.print("[red]Use format: --set key=value[/red]")
            return
        key, value = set_value.split("=", 1)
        console.print(f"[yellow]Setting {key} = {value} (stored in .env for now)[/yellow]")
        # Write to .env file
        env_path = Path(".env")
        env_key = f"SI_{key.upper()}"
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        updated = False
        for index, line in enumerate(lines):
            if line.partition("=")[0].strip() == env_key:
                lines[index] = f"{env_key}={value}"
                updated = True
                break
        if not updated:
            lines.append(f"{env_key}={value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print("[green]Configuration saved.[/green]")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app()


if __name__ == "__main__":
    main()
