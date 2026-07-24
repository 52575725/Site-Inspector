from __future__ import annotations

import inspect

from src.core.engine import Engine
from src.core.fix_orchestrator import FixOrchestrator


def test_daily_scan_defaults_to_dry_run():
    dry_run = inspect.signature(Engine.run_daily_scan).parameters["dry_run"]
    assert dry_run.default is True


def test_per_page_fix_cap_stays_conservative():
    assert FixOrchestrator.MAX_FIXES_PER_FILE == 3
