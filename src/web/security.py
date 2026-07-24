from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlparse

from fastapi import HTTPException


_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
_GITHUB_REPO_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$",
    re.IGNORECASE,
)


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def validate_public_http_url(value: str) -> str:
    """Validate a scan URL and reject local, private, and ambiguous hosts."""
    raw = value.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="A valid HTTP(S) URL is required")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Credentials in scan URLs are not allowed")
    if parsed.port not in {None, 80, 443}:
        raise HTTPException(status_code=400, detail="Only ports 80 and 443 are allowed")

    try:
        literal = ipaddress.ip_address(parsed.hostname)
        if not _is_public_address(str(literal)):
            raise HTTPException(status_code=400, detail="Private or local addresses are not allowed")
    except ValueError:
        try:
            results = await asyncio.to_thread(
                socket.getaddrinfo,
                parsed.hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise HTTPException(status_code=400, detail="Host could not be resolved") from exc
        addresses = {item[4][0] for item in results}
        if not addresses or any(not _is_public_address(address) for address in addresses):
            raise HTTPException(status_code=400, detail="Private or local addresses are not allowed")

    return raw


def validate_github_repo(repo_url: str, branch: str) -> tuple[str, str]:
    repo = repo_url.strip()
    if not _GITHUB_REPO_RE.fullmatch(repo):
        raise HTTPException(
            status_code=400,
            detail="Repository writes only support explicit HTTPS GitHub repository URLs",
        )
    if not _BRANCH_RE.fullmatch(branch) or ".." in branch or branch.startswith(("/", "-")):
        raise HTTPException(status_code=400, detail="Invalid repository branch")
    return repo, branch
