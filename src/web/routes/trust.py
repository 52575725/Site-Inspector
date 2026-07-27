"""Trust & Verify dashboard — evidence quality, fix effectiveness, content factuality."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from src.core.trust_verifier import (
    EvidenceQuality,
    assess_evidence_quality,
    check_cross_consistency,
    compute_trust_report,
)
from src.storage.database import get_session_factory
from src.storage.models import Fix, Issue, Scan
from src.web.deps import templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trust"])


@router.get("/trust")
async def trust_page(request: Request):
    """Trust & Verify dashboard."""
    return templates.TemplateResponse(request, "trust.html")


@router.get("/api/trust/report")
async def trust_report_api(request: Request, scan_id: int | None = None):
    """Get a trust report for the latest scan (or specific scan)."""
    settings = request.app.state.settings
    session_factory = get_session_factory(settings)

    async with session_factory() as session:
        # Get scan
        if scan_id:
            result = await session.execute(
                select(Scan).where(Scan.id == scan_id)
            )
        else:
            result = await session.execute(
                select(Scan)
                .where(Scan.completed_at is not None)
                .order_by(Scan.completed_at.desc())
                .limit(1)
            )
        scan = result.scalar_one_or_none()
        if scan is None:
            return {"error": "No scan found", "badge": "none", "overall_trust": 0}

        # Get issues and fixes for this scan
        issues_result = await session.execute(
            select(Issue).where(Issue.scan_id == scan.id)
        )
        issues = issues_result.scalars().all()

        fixes_result = await session.execute(
            select(Fix).where(Fix.scan_id == scan.id)
        )
        fixes = fixes_result.scalars().all()

        # Assess evidence quality
        evidence_scores = assess_evidence_quality(issues)
        avg_evidence = (
            sum(e.evidence_score for e in evidence_scores) / max(len(evidence_scores), 1)
            if evidence_scores else 1.0
        )

        # Evidence breakdown by inspector
        by_inspector: dict[str, list[float]] = {}
        for issue, eq in zip(issues, evidence_scores):
            insp = getattr(issue, 'inspector', 'unknown')
            by_inspector.setdefault(insp, []).append(eq.evidence_score)
        inspector_scores = {
            k: sum(v) / len(v) for k, v in by_inspector.items()
        }

        # Fix trust
        fix_attempted = len(fixes)
        fix_failed = sum(1 for f in fixes if getattr(f, 'status', '') == "validation_failed")
        fix_success_rate = (
            1.0 - (fix_failed / max(fix_attempted, 1))
            if fix_attempted > 0 else 1.0
        )

        # Cross-consistency
        issues_by_url: dict[str, list] = {}
        for issue in issues:
            url = getattr(issue, 'url', '')
            issues_by_url.setdefault(url, []).append(issue)
        consistencies = check_cross_consistency(issues_by_url)
        avg_consistency = (
            sum(c.consistency_score for c in consistencies) / max(len(consistencies), 1)
            if consistencies else 1.0
        )

        # Contradictions found
        contradictions = [
            {"url": c.url, "issues": c.contradictions}
            for c in consistencies if c.contradictions
        ]

        # Compute trust report
        report = compute_trust_report(
            detection_trust=avg_evidence,
            fix_trust=fix_success_rate,
            generation_trust=1.0,
            cross_consistency=avg_consistency,
            scan_id=scan.id,
        )

        # Low-evidence findings: issues with score < 0.5 that need improvement
        low_evidence = [
            {
                "id": getattr(issue, 'id', 0),
                "category": eq.category,
                "inspector": getattr(issue, 'inspector', '?'),
                "score": eq.evidence_score,
                "missing": [
                    m for m in ["element", "current_value", "suggested_value", "raw_metadata"]
                    if not getattr(eq, f"has_{m}")
                ],
            }
            for issue, eq in zip(issues, evidence_scores)
            if eq.evidence_score < 0.5
        ][:20]

    return {
        "scan_id": scan.id,
        "scan_date": scan.completed_at.isoformat() if scan.completed_at else None,
        "badge": report.badge,
        "overall_trust": round(report.overall_trust, 3),
        "detection_trust": round(avg_evidence, 3),
        "fix_trust": round(fix_success_rate, 3),
        "cross_consistency": round(avg_consistency, 3),
        "total_issues": len(issues),
        "total_fixes": len(fixes),
        "fix_failures": fix_failed,
        "contradictions": contradictions[:10],
        "low_evidence_findings": low_evidence,
        "inspector_scores": inspector_scores,
        "details": report.details,
    }
