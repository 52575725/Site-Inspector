from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.reporters.daily_report import explain_issue


STATUS_LABELS = {
    "proposed": "等待批准",
    "approved": "已批准，等待应用",
    "applied": "已应用，等待验证",
    "pr_created": "已创建代码审查",
    "rejected": "已拒绝",
    "pending": "正在观察效果",
    "improved": "修复后有所改善",
    "degraded": "效果可能变差",
    "rollback_required": "需要人工恢复",
}

LOW_RISK = {
    "missing_title", "title_too_short", "title_too_long",
    "missing_meta_description", "meta_description_too_short",
    "meta_description_too_long", "missing_alt_text", "empty_alt_text",
    "missing_viewport_meta", "missing_canonical", "image_no_lazy_loading",
}
HIGH_RISK = {
    "thin_content", "duplicate_content", "low_content_quality_ai",
    "missing_content", "content_gap_section", "content_gap_word_count",
    "invalid_jsonld", "robots_txt_disallow_all",
}


def risk_for_category(category: str) -> str:
    if category in LOW_RISK:
        return "low"
    if category in HIGH_RISK:
        return "high"
    return "medium"


def describe_issue(category: str, description: str = "") -> dict[str, str]:
    explanation = explain_issue(category, description)
    return {
        "summary": explanation["what"],
        "impact": explanation["impact"],
        "action": explanation["how"],
        "outcome": explanation["outcome"],
        "risk": risk_for_category(category),
    }


def build_fix_preview(fix, *, include_developer: bool = True) -> dict:
    issue = fix.issue
    plain = describe_issue(issue.category, issue.description or "")
    added, removed = _diff_size(fix.diff or "")
    change_count = added + removed
    risk = fix.risk_level or plain["risk"]
    warning = None
    if change_count > 40:
        risk = "high"
        warning = f"这次建议会改动约 {change_count} 行，除了目标内容外还可能重排页面 HTML，请先查看开发者信息。"
    preview = {
        "id": fix.id,
        "issue_id": issue.id,
        "problem": fix.plain_summary or plain["summary"],
        "impact": fix.impact_explanation or plain["impact"],
        "action": fix.change_explanation or plain["action"],
        "outcome": plain["outcome"],
        "risk": risk,
        "warning": warning,
        "change_count": change_count,
        "before": _readable_value(issue.category, fix.before_content or "", before=True),
        "after": _readable_value(issue.category, fix.after_content or "", before=False),
        "file_path": fix.file_path,
        "page_url": issue.url,
        "status": fix.status or "proposed",
        "status_label": STATUS_LABELS.get(fix.status or "proposed", fix.status or "proposed"),
        "git_pr_url": fix.git_pr_url,
    }
    if include_developer:
        raw_diff = fix.diff or ""
        preview["diff"] = raw_diff[:20000]
        preview["diff_truncated"] = len(raw_diff) > 20000
    return preview


def _diff_size(diff: str) -> tuple[int, int]:
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed


def _readable_value(category: str, html: str, *, before: bool) -> str:
    if not html:
        return "没有内容" if before else "未生成内容"
    soup = BeautifulSoup(html, "html.parser")
    if "title" in category:
        tag = soup.find("title")
        return tag.get_text(" ", strip=True) if tag else "没有页面标题"
    if "meta_description" in category:
        tag = soup.find("meta", attrs={"name": "description"})
        return tag.get("content", "") if tag else "没有搜索结果描述"
    if "canonical" in category:
        tag = soup.find("link", rel="canonical")
        return tag.get("href", "") if tag else "没有规范页面地址"
    if category in {"missing_og_tags", "missing_og_image", "missing_twitter_cards"}:
        values = [
            f"{tag.get('property') or tag.get('name')}：{tag.get('content', '')}"
            for tag in soup.find_all("meta")
            if (tag.get("property", "").startswith("og:") or tag.get("name", "").startswith("twitter:"))
        ]
        return "\n".join(values[:10]) or "没有社交分享预览信息"
    if category in {"missing_h1", "multiple_h1", "h_tag_skip"}:
        headings = [f"{tag.name.upper()}：{tag.get_text(' ', strip=True)}" for tag in soup.find_all(re.compile("^h[1-6]$"))]
        return "\n".join(headings[:12]) or "没有页面主标题"
    if "alt" in category:
        images = [f"{tag.get('src', '图片')}：{tag.get('alt') or '没有说明'}" for tag in soup.find_all("img")]
        return "\n".join(images[:8]) or "没有图片"

    text = soup.get_text(" ", strip=True)
    return text[:500] + ("..." if len(text) > 500 else "")
