from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.database import Base


class Target(Base):
    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), default="http")
    source_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    languages: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    scans = relationship("Scan", back_populates="target", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(ForeignKey("targets.id"), nullable=False)
    scan_type: Mapped[str] = mapped_column(String(20), default="daily")
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    phase: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pages_crawled: Mapped[int] = mapped_column(Integer, default=0)
    total_issues_found: Mapped[int] = mapped_column(Integer, default=0)
    fix_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pr_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    target = relationship("Target", back_populates="scans")
    page_scans = relationship("PageScan", back_populates="scan", cascade="all, delete-orphan")
    issues = relationship("Issue", back_populates="scan", cascade="all, delete-orphan")
    fixes = relationship("Fix", back_populates="scan", cascade="all, delete-orphan")


class PageScan(Base):
    __tablename__ = "page_scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    http_status: Mapped[int] = mapped_column(Integer, default=200)
    load_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    html_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    lighthouse_json_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    axe_json_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    screenshot_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    scan = relationship("Scan", back_populates="page_scans")
    issues = relationship("Issue", back_populates="page_scan", cascade="all, delete-orphan")


class Issue(Base):
    __tablename__ = "issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), nullable=False)
    page_scan_id: Mapped[int] = mapped_column(ForeignKey("page_scans.id"), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    inspector: Mapped[str] = mapped_column(String(50), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    impact_scope: Mapped[float] = mapped_column(Float, default=0.0)
    fix_roi: Mapped[float] = mapped_column(Float, default=0.0)
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    priority_tier: Mapped[str] = mapped_column(String(5), default="P2")
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    element: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    current_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="open")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    fix_applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    scan = relationship("Scan", back_populates="issues")
    page_scan = relationship("PageScan", back_populates="issues")
    fixes = relationship("Fix", back_populates="issue", cascade="all, delete-orphan")


class Fix(Base):
    __tablename__ = "fixes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id"), nullable=False)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), nullable=False)
    fixer: Mapped[str] = mapped_column(String(50), nullable=False)
    fix_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="proposed", nullable=False)
    plain_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    impact_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    change_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_level: Mapped[str] = mapped_column(String(10), default="medium", nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    before_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    after_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diff: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    git_branch: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    git_pr_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sandbox_screenshot_before: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sandbox_screenshot_after: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sandbox_diff_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rejected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    merged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    issue = relationship("Issue", back_populates="fixes")
    scan = relationship("Scan", back_populates="fixes")
    verifications = relationship(
        "Verification", back_populates="fix", cascade="all, delete-orphan"
    )


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fix_id: Mapped[int] = mapped_column(ForeignKey("fixes.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    value_before: Mapped[float] = mapped_column(Float, nullable=False)
    value_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    fix = relationship("Fix", back_populates="verifications")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    entity_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
