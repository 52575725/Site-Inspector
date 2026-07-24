from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from src.git.workflow import GitWorkflow


def _workflow() -> GitWorkflow:
    return GitWorkflow(Settings(), Path("repo"))


@pytest.mark.asyncio
async def test_ensure_clean_fails_closed_when_git_status_errors(monkeypatch):
    workflow = _workflow()

    async def fail(*args):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(workflow, "_run_git", fail)

    assert not await workflow.ensure_clean()


@pytest.mark.asyncio
async def test_create_fix_branch_refuses_dirty_repository(monkeypatch):
    workflow = _workflow()
    calls = []

    async def run(*args):
        calls.append(args)
        if args == ("status", "--porcelain"):
            return " M index.html"
        return ""

    monkeypatch.setattr(workflow, "_run_git", run)

    with pytest.raises(RuntimeError, match="dirty repository"):
        await workflow.create_fix_branch(42)

    assert not any(args[:2] == ("checkout", "-b") for args in calls)


@pytest.mark.asyncio
async def test_stage_and_commit_uses_option_separator_and_scoped_identity(monkeypatch):
    workflow = _workflow()
    calls = []

    async def run(*args):
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return "0123456789abcdef"
        return ""

    monkeypatch.setattr(workflow, "_run_git", run)

    commit = await workflow.stage_and_commit(
        ["--suspicious-name.html", "about/index.html"],
        "fix: metadata",
    )

    assert ("add", "--", "--suspicious-name.html") in calls
    assert ("add", "--", "about/index.html") in calls
    commit_call = next(args for args in calls if "commit" in args)
    assert "user.name=Site Inspector Bot" in commit_call
    assert "user.email=bot@site-inspector.local" in commit_call
    assert not any(args and args[0] == "config" for args in calls)
    assert commit == "01234567"
