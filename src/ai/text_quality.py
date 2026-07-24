from __future__ import annotations

import re

import textstat


def assess_readability(text: str, language: str = "en") -> dict:
    """Assess text readability. Uses Flesch-Kincaid for EN, heuristic for JP."""
    if not text or len(text.split()) < 50:
        return {"score": 0, "level": "unknown", "is_thin": True}

    if language == "en":
        try:
            score = textstat.flesch_reading_ease(text)
            grade = textstat.flesch_kincaid_grade(text)
        except Exception:
            score = 50
            grade = 10
        return {
            "score": round(score, 1),
            "grade_level": round(grade, 1),
            "is_thin": len(text.split()) < 300,
        }

    # JP / other languages: character-based heuristic
    char_count = len(re.sub(r"\s", "", text))
    if char_count < 500:
        level = "thin"
    elif char_count < 1500:
        level = "light"
    elif char_count < 3000:
        level = "adequate"
    else:
        level = "comprehensive"

    return {
        "score": min(100, char_count / 30),
        "level": level,
        "is_thin": char_count < 500,
    }


def detect_duplicate_risk(text: str, existing_texts: list[str]) -> float:
    """Simple simhash-based duplicate detection. Returns similarity score 0-1."""
    try:
        from simhash import Simhash

        if not existing_texts:
            return 0.0

        current_hash = Simhash(text)
        max_similarity = 0.0
        for existing in existing_texts:
            existing_hash = Simhash(existing)
            distance = current_hash.distance(existing_hash)
            similarity = 1.0 - (distance / 64)
            max_similarity = max(max_similarity, similarity)
        return max_similarity
    except ImportError:
        return 0.0


def estimate_content_to_html_ratio(html: str) -> float:
    """Estimate text content vs HTML markup ratio."""
    text_len = len(re.sub(r"<[^>]+>", "", html).strip())
    if len(html) == 0:
        return 0.0
    return text_len / len(html)
