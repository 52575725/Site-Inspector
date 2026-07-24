"""Scan completion notifications: desktop toast + email.

Integrates into scheduler and Quick Scan flow so users are notified
when scans complete without needing to keep the dashboard open.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from config.settings import Settings

logger = logging.getLogger(__name__)


class Notifier:
    """Sends scan completion notifications via desktop toast and/or email."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def notify_scan_complete(
        self,
        target_name: str,
        pages: int,
        issues: int,
        fixes: int,
        pr_url: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> dict:
        """Notify that a scan has completed. Returns dict of {channel: success_bool}."""
        results = {}

        # Desktop toast (always try)
        results["desktop"] = await self._send_desktop_notification(
            title=f"Site Inspector: {target_name}",
            message=(
                f"✅ 扫描完成\n"
                f"页面: {pages} | 问题: {issues} | 修复: {fixes}"
                + (f"\nPR: {pr_url}" if pr_url else "")
            ),
        )

        # Email (if configured)
        email_ok = await self._send_email(target_name, pages, issues, fixes, pr_url, report_path)
        if email_ok is not None:
            results["email"] = email_ok

        logger.info(f"Notifications sent: {results}")
        return results

    async def notify_error(
        self,
        target_name: str,
        error_message: str,
    ) -> None:
        """Notify about a scan failure."""
        await self._send_desktop_notification(
            title=f"⚠ Site Inspector: {target_name}",
            message=f"扫描失败: {error_message[:200]}",
        )

    async def notify_article_generated(
        self,
        title: str,
        pr_url: Optional[str] = None,
    ) -> None:
        """Notify that an article was generated and pushed."""
        msg = f"✍️ 文章已生成: {title}"
        if pr_url:
            msg += f"\nPR: {pr_url}"
        await self._send_desktop_notification(
            title="Site Inspector: 文章生成",
            message=msg,
        )

    # ── Desktop Notification ────────────────────────────────────────

    async def _send_desktop_notification(self, title: str, message: str) -> bool:
        """Send a cross-platform desktop toast notification.

        On Windows, uses PowerShell. On macOS, uses osascript.
        On Linux, uses notify-send.
        """
        system = platform.system()

        try:
            if system == "Windows":
                # Escape for PowerShell
                ps_title = title.replace('"', '""')
                ps_msg = message.replace('"', '""').replace('\n', ' ')

                # Try Windows.UI.Notifications first (Win10+), fall back to BurntToast
                ps_script = (
                    '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; '
                    '$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('
                    '[Windows.UI.Notifications.ToastTemplateType]::ToastText02); '
                    f'$template.GetElementsByTagName("text")[0].AppendChild($template.CreateTextNode("{ps_title}")) > $null; '
                    f'$template.GetElementsByTagName("text")[1].AppendChild($template.CreateTextNode("{ps_msg}")) > $null; '
                    '$toast = [Windows.UI.Notifications.ToastNotification]::new($template); '
                    '[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Site Inspector").Show($toast)'
                )
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-NoProfile", "-Command", ps_script,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                return proc.returncode == 0

            elif system == "Darwin":
                proc = await asyncio.create_subprocess_exec(
                    "osascript", "-e",
                    f'display notification "{message}" with title "{title}"',
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                return proc.returncode == 0

            else:  # Linux
                proc = await asyncio.create_subprocess_exec(
                    "notify-send", title, message,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                return proc.returncode == 0

        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")
            return False

    # ── Email Notification ──────────────────────────────────────────

    async def _send_email(
        self,
        target_name: str,
        pages: int,
        issues: int,
        fixes: int,
        pr_url: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> Optional[bool]:
        """Send scan summary via email if SMTP is configured. Returns None if skipped."""
        smtp_host = self.settings.smtp_host
        if not smtp_host:
            return None  # not configured, not a failure

        recipients = self.settings.report_recipients
        if not recipients:
            return None

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"Site Inspector: {target_name} 扫描完成 — {issues} 个问题"
            msg["From"] = self.settings.smtp_username or "bot@site-inspector.local"
            msg["To"] = ", ".join(recipients)

            pr_line = f'<li>🔗 PR: <a href="{pr_url}">{pr_url}</a></li>' if pr_url else ""
            html = f"""
            <html><body>
            <h2>🔍 Site Inspector 扫描报告</h2>
            <p><strong>站点:</strong> {target_name}</p>
            <ul>
                <li>📄 爬取页面: {pages}</li>
                <li>⚠️ 发现问题: {issues}</li>
                <li>🔧 自动修复: {fixes}</li>
                {pr_line}
            </ul>
            <p><small>由 Site Inspector 自动发送</small></p>
            </body></html>
            """
            msg.attach(MIMEText(html, "html"))

            # Run SMTP in thread to not block event loop
            def _send():
                with smtplib.SMTP(smtp_host, self.settings.smtp_port, timeout=15) as server:
                    server.starttls()
                    if self.settings.smtp_username and self.settings.smtp_password:
                        server.login(self.settings.smtp_username, self.settings.smtp_password)
                    server.sendmail(msg["From"], recipients, msg.as_string())

            await asyncio.to_thread(_send)
            logger.info(f"Email sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            logger.warning(f"Failed to send email notification: {e}")
            return False
