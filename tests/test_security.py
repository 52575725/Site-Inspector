from __future__ import annotations

import socket

import pytest
from fastapi import HTTPException

from src.sources.base import resolve_within
from src.web.security import validate_github_repo, validate_public_http_url


def test_resolve_within_accepts_child(tmp_path):
    assert resolve_within(tmp_path, "pages/index.html") == (
        tmp_path / "pages" / "index.html"
    ).resolve()


@pytest.mark.parametrize("path", ["../secret.txt", "pages/../../secret.txt"])
def test_resolve_within_rejects_traversal(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_within(tmp_path, path)


@pytest.mark.asyncio
async def test_scan_url_rejects_loopback():
    with pytest.raises(HTTPException) as exc:
        await validate_public_http_url("http://127.0.0.1/admin")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_scan_url_rejects_private_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))],
    )
    with pytest.raises(HTTPException):
        await validate_public_http_url("https://internal.example")


def test_repo_write_rejects_local_or_ambiguous_sources():
    with pytest.raises(HTTPException):
        validate_github_repo("file:///C:/repo", "main")
    with pytest.raises(HTTPException):
        validate_github_repo("https://github.com/acme/site.git", "../main")


def test_repo_write_accepts_explicit_github_https_url():
    assert validate_github_repo("https://github.com/acme/site.git", "main") == (
        "https://github.com/acme/site.git",
        "main",
    )
