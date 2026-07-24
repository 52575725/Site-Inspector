"""Settings API — read and update .env configuration from the Web UI."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])

ENV_PATH = Path(".env")


class EmailSettings(BaseModel):
    recipients: str = ""  # JSON array string or comma-separated
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""


@router.get("/api/settings/email")
async def get_email_settings(request: Request):
    """Get current email notification settings from .env."""
    settings = request.app.state.settings
    recipients = settings.report_recipients or []
    return {
        "recipients": ", ".join(recipients),
        "smtp_host": settings.smtp_host or "",
        "smtp_port": settings.smtp_port or 587,
        "smtp_username": settings.smtp_username or "",
        "has_password": bool(settings.smtp_password),
    }


@router.post("/api/settings/email")
async def update_email_settings(request: Request, body: EmailSettings):
    """Update email notification settings in .env file."""
    # Parse recipients
    recipients = [r.strip() for r in body.recipients.replace("[", "").replace("]", "").replace('"', "").split(",") if r.strip()]

    # Read current .env
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    updates = {
        "SI_REPORT_RECIPIENTS": json.dumps(recipients, ensure_ascii=False),
        "SI_SMTP_HOST": body.smtp_host,
        "SI_SMTP_PORT": str(body.smtp_port),
        "SI_SMTP_USERNAME": body.smtp_username,
    }
    if body.smtp_password:
        updates["SI_SMTP_PASSWORD"] = body.smtp_password

    # Update or append
    updated_keys = set()
    new_lines = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    # Append new keys
    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {"status": "ok", "message": "设置已保存，重启服务后生效", "recipients": recipients}
