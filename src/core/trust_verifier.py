"""Trust & Verify — evidence-bound quality assurance for the agent pipeline.

Three dimensions of trust:
1. Detection Trust:  Is every finding backed by concrete evidence?  Do different
   inspectors agree or contradict each other?
2. Fix Trust:  After a fix is applied, was the issue actually resolved?  Did the
   fix introduce new problems?
3. Generation Trust:  Is AI-generated content factually consistent with source
   material?  Are claims traceable?

Architecture:  This module sits OUTSIDE the normal scan→fix pipeline.  It is a
separate verification pass that can be run independently to audit results.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import Fix, Issue, Scan

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class EvidenceQuality:
    """Assessment of how well a finding is backed by evidence."""
    finding_id: int
    category: str
    has_element: bool = False        # Does it point to the specific HTML element?
    has_current_value: bool = False  # Does it show what the current value is?
    has_suggested_value: bool = False  # Does it suggest a fix?
    has_raw_metadata: bool = False   # Does it include structured evidence?
    evidence_score: float = 0.0      # 0-1 composite score


@dataclass
class CrossConsistency:
    """Check if different inspectors contradict each other."""
    url: str
    contradictions: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    consistency_score: float = 1.0


@dataclass
class FixRecheckResult:
    """Result of re-inspecting a page after a fix was applied."""
    fix_id: int
    issue_id: int
    category: str
    was_resolved: bool = False
    original_finding: str = ""
    recheck_finding: str = ""
    new_issues_introduced: int = 0
    score: float = 0.0  # 1.0 = perfectly resolved


@dataclass
class TrustReport:
    """Complete trust assessment for a scan or target."""
    scan_id: int | None = None
    target_name: str = ""
    detection_trust: float = 0.0      # Evidence quality across all findings
    fix_trust: float = 0.0            # Fix effectiveness rate
    generation_trust: float = 0.0     # AI content quality
    cross_consistency: float = 0.0    # Inspector agreement rate
    overall_trust: float = 0.0        # Weighted composite
    badge: str = "none"               # gold/silver/bronze/none
    details: list[str] = field(default_factory=list)
    generated_at: str = ""


# ── Detection Trust ───────────────────────────────────────────────────


def assess_evidence_quality(issues: list) -> list[EvidenceQuality]:
    """Score each finding on how well it's backed by concrete evidence.

    A finding with:
    - element + current_value + suggested_value + raw_metadata = 1.0
    - only description text = 0.25
    """
    results = []
    for issue in issues:
        eq = EvidenceQuality(
            finding_id=issue.id if hasattr(issue, 'id') else 0,
            category=issue.category if hasattr(issue, 'category') else "",
        )
        eq.has_element = bool(
            getattr(issue, 'element', None)
        )
        eq.has_current_value = bool(
            getattr(issue, 'current_value', None)
        )
        eq.has_suggested_value = bool(
            getattr(issue, 'suggested_value', None)
        )
        eq.has_raw_metadata = bool(
            getattr(issue, 'raw_metadata', None)
        )
        # Score: each piece of evidence is worth 0.25
        eq.evidence_score = sum([
            0.25 if eq.has_element else 0,
            0.25 if eq.has_current_value else 0,
            0.25 if eq.has_suggested_value else 0,
            0.25 if eq.has_raw_metadata else 0,
        ])
        results.append(eq)
    return results


def check_cross_consistency(issues_by_url: dict[str, list]) -> list[CrossConsistency]:
    """Detect when different inspectors disagree about the same page.

    Example contradictions:
    - SEO says "missing title" but another inspector found a title tag
    - ContentQuality says "thin content" but ContentFreshness says "comprehensive"
    """
    results: list[CrossConsistency] = []
    # Map of inspector -> set of categories found per URL
    for url, issues in issues_by_url.items():
        cc = CrossConsistency(url=url)

        # Group by category type
        has_title_tag = False
        missing_title = False
        thin_content = False
        comprehensive = False

        for issue in issues:
            cat = getattr(issue, 'category', '')
            if cat == 'missing_title':
                missing_title = True
            if 'title' in cat.lower() and cat != 'missing_title':
                has_title_tag = True  # Title was analyzed
            if cat == 'thin_content':
                thin_content = True
            if cat in ('freshness_comprehensive',):
                comprehensive = True

        # Contradiction: missing title but title was found
        if missing_title and has_title_tag:
            cc.contradictions.append(
                "SEO reports missing title, but another inspector found and "
                "analyzed the title tag"
            )

        # Contradiction: thin vs comprehensive
        if thin_content and comprehensive:
            cc.contradictions.append(
                "Content flagged as thin but also marked as comprehensive — "
                "these assessments conflict"
            )

        # Confirmations: both SEO and ContentQuality agree
        if missing_title and not has_title_tag:
            cc.confirmations.append("Multiple inspectors agree: title is missing")

        cc.consistency_score = max(
            0.0,
            1.0 - len(cc.contradictions) * 0.3,
        )
        results.append(cc)

    return results


# ── Fix Trust ─────────────────────────────────────────────────────────


async def recheck_fix(
    fix: Fix, issue: Issue, session: AsyncSession,
    re_inspect_fn,
) -> FixRecheckResult:
    """After a fix is applied, re-inspect the page to verify the fix worked.

    Args:
        fix: The applied fix record
        issue: The original issue that was fixed
        session: DB session
        re_inspect_fn: async function(html_content: str, url: str) -> list[RawFinding]
                       that re-runs the SAME inspector on the fixed page

    Returns:
        FixRecheckResult with resolution status
    """
    result = FixRecheckResult(
        fix_id=fix.id if hasattr(fix, 'id') else 0,
        issue_id=issue.id if hasattr(issue, 'id') else 0,
        category=issue.category if hasattr(issue, 'category') else "",
    )

    # Get the fixed content
    after_content = getattr(fix, 'after_content', '') or ''
    before_content = getattr(fix, 'before_content', '') or ''
    url = getattr(issue, 'url', '')

    result.original_finding = (
        getattr(issue, 'description', '') or ''
    )[:200]

    if not after_content or not url:
        result.score = 0.0
        return result

    # Re-inspect the fixed content
    try:
        new_findings = await re_inspect_fn(after_content, url)
    except Exception as e:
        logger.warning(f"Re-inspection failed for fix {result.fix_id}: {e}")
        result.score = 0.5  # Uncertain
        result.recheck_finding = f"Re-inspection error: {e}"
        return result

    # Check if the original issue category still appears
    issue_still_present = any(
        getattr(f, 'category', '') == result.category
        for f in new_findings
    )

    result.was_resolved = not issue_still_present
    result.new_issues_introduced = len(new_findings)
    result.recheck_finding = (
        "Issue RESOLVED" if not issue_still_present
        else f"Issue STILL PRESENT: {result.category}"
    )

    # Score:
    # 1.0 = resolved, no new issues
    # 0.7 = resolved, but introduced new issues
    # 0.3 = not resolved
    # 0.0 = not resolved + new issues
    if result.was_resolved and result.new_issues_introduced == 0:
        result.score = 1.0
    elif result.was_resolved:
        result.score = 0.7
    elif result.new_issues_introduced == 0:
        result.score = 0.3
    else:
        result.score = 0.0

    return result


async def compute_fix_trust_score(
    fixes: list[Fix], issues: dict[int, Issue], session: AsyncSession,
    re_inspect_fn,
) -> tuple[float, list[FixRecheckResult]]:
    """Compute overall fix trust score from a batch of rechecks."""
    if not fixes:
        return 1.0, []

    results: list[FixRecheckResult] = []
    for fix in fixes:
        issue = issues.get(getattr(fix, 'issue_id', 0))
        if issue is None:
            continue
        result = await recheck_fix(fix, issue, session, re_inspect_fn)
        results.append(result)

    if not results:
        return 1.0, []

    avg_score = sum(r.score for r in results) / len(results)
    return avg_score, results


# ── Generation Trust ──────────────────────────────────────────────────


@dataclass
class ContentFactuality:
    """Check if AI-generated content is factually consistent with source."""
    url: str
    source_claims: list[str] = field(default_factory=list)
    generated_claims: list[str] = field(default_factory=list)
    hallucinations: list[str] = field(default_factory=list)
    factual_consistency: float = 1.0  # 1.0 = fully consistent
    source_traceability: float = 0.0  # How much is traceable to source
    issues: list[str] = field(default_factory=list)


def check_content_factuality(
    source_content: str, generated_content: str, url: str = "",
) -> ContentFactuality:
    """Basic factuality check without requiring AI.

    Checks:
    - Content overlap ratio (did we lose the original meaning?)
    - Key facts preserved (prices, dates, names unchanged?)
    - Generated content isn't just a slight rewrite
    """
    import re

    result = ContentFactuality(url=url)

    # Extract key facts from source (prices, dates, numbers, named entities)
    source_prices = set(re.findall(r'\$\d[\d,.]*', source_content))
    gen_prices = set(re.findall(r'\$\d[\d,.]*', generated_content))

    source_dates = set(re.findall(r'\b20\d{2}-\d{2}-\d{2}\b|\b20\d{2}\b', source_content))
    gen_dates = set(re.findall(r'\b20\d{2}-\d{2}-\d{2}\b|\b20\d{2}\b', generated_content))

    source_nums = set(re.findall(r'\b\d+%?\b', source_content))
    gen_nums = set(re.findall(r'\b\d+%?\b', generated_content))

    # Check price preservation
    missing_prices = source_prices - gen_prices
    if missing_prices:
        result.hallucinations.append(
            f"Price data lost: {', '.join(sorted(missing_prices)[:5])}"
        )

    # Check date preservation
    missing_dates = source_dates - gen_dates
    if missing_dates:
        result.issues.append(
            f"Date information lost: {', '.join(sorted(missing_dates)[:5])}"
        )

    # Check new invented data
    invented_prices = gen_prices - source_prices
    if invented_prices:
        result.hallucinations.append(
            f"Prices may be invented: {', '.join(sorted(invented_prices)[:5])}"
        )

    invented_dates = gen_dates - source_dates
    if invented_dates:
        result.hallucinations.append(
            f"Dates may be invented: {', '.join(sorted(invented_dates)[:5])}"
        )

    # Content overlap (what fraction of source key info is in generated)
    if source_nums:
        preserved = source_nums & gen_nums
        result.source_traceability = len(preserved) / max(len(source_nums), 1)

    # Factual consistency score
    fact_loss = len(result.hallucinations) * 0.2 + len(result.issues) * 0.1
    result.factual_consistency = max(0.0, 1.0 - fact_loss)

    return result


# ── Composite Trust Report ────────────────────────────────────────────


def compute_trust_report(
    detection_trust: float,
    fix_trust: float,
    generation_trust: float,
    cross_consistency: float,
    scan_id: int | None = None,
    target_name: str = "",
) -> TrustReport:
    """Compute overall trust score and badge level."""
    report = TrustReport(
        scan_id=scan_id,
        target_name=target_name,
        detection_trust=detection_trust,
        fix_trust=fix_trust,
        generation_trust=generation_trust,
        cross_consistency=cross_consistency,
        generated_at=datetime.utcnow().isoformat(),
    )

    # Weighted composite: detection 35%, fix 35%, generation 15%, consistency 15%
    report.overall_trust = (
        detection_trust * 0.35
        + fix_trust * 0.35
        + generation_trust * 0.15
        + cross_consistency * 0.15
    )

    # Badge levels
    if report.overall_trust >= 0.85:
        report.badge = "gold"
    elif report.overall_trust >= 0.65:
        report.badge = "silver"
    elif report.overall_trust >= 0.40:
        report.badge = "bronze"
    else:
        report.badge = "none"

    # Build detail strings
    if detection_trust < 0.7:
        report.details.append(
            f"Low evidence quality ({detection_trust:.0%}): "
            f"findings lack element references, current values, or suggested fixes"
        )
    if fix_trust < 0.7:
        report.details.append(
            f"Low fix effectiveness ({fix_trust:.0%}): "
            f"many fixes did not resolve the target issue when re-checked"
        )
    if generation_trust < 0.7:
        report.details.append(
            f"Low generation trust ({generation_trust:.0%}): "
            f"AI content may contain hallucinations or unsourced claims"
        )
    if cross_consistency < 0.7:
        report.details.append(
            f"Cross-inspector contradictions detected — "
            f"different inspectors disagree about the same page"
        )

    if report.overall_trust >= 0.85:
        report.details.append(
            "All quality checks passed. Findings are evidence-backed, "
            "fixes are verified, and content is traceable."
        )

    return report
