from __future__ import annotations

import math
import re
from typing import Optional

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


def detect_duplicate_risk_simhash(text: str, existing_texts: list[str]) -> float:
    """Simhash-based duplicate detection. Returns similarity score 0-1.

    Fast but only catches near-identical content.  For semantic duplicate
    detection (same meaning, different words), use detect_duplicate_risk_embedding.
    """
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


async def detect_duplicate_risk_embedding(
    text: str,
    existing_texts: list[str],
    embed_fn=None,
    threshold: float = 0.88,
) -> tuple[float, Optional[str]]:
    """Embedding-based semantic duplicate detection.

    Uses cosine similarity of text embeddings to detect content that is
    semantically similar even when worded differently.  Much better than
    Simhash for catching rewritten/spun content.

    Args:
        text: The text to check.
        existing_texts: All other page texts to compare against.
        embed_fn: Async function that takes text and returns a list[float].
        threshold: Cosine similarity above which content is considered duplicate.

    Returns:
        (max_similarity, most_similar_text_snippet_or_None)
    """
    if not existing_texts or embed_fn is None:
        return 0.0, None

    # Only check against texts of similar length to reduce comparisons
    text_len = len(text)
    candidates = [
        t for t in existing_texts
        if t != text and 0.2 < len(t) / max(text_len, 1) < 5.0
    ]
    if not candidates:
        return 0.0, None

    try:
        current_embedding = await embed_fn(text[:8000])
    except Exception:
        return 0.0, None

    max_similarity = 0.0
    best_match: Optional[str] = None

    # Check in batches to limit embedding API calls
    for candidate in candidates[:20]:  # Cap at 20 comparisons
        try:
            candidate_embedding = await embed_fn(candidate[:8000])
        except Exception:
            continue

        sim = _cosine_similarity(current_embedding, candidate_embedding)
        if sim > max_similarity:
            max_similarity = sim
            best_match = candidate[:200]

    return max_similarity, best_match


async def detect_duplicate_risk(
    text: str,
    existing_texts: list[str],
    embed_fn=None,
    embedding_threshold: float = 0.88,
) -> float:
    """Smart duplicate detection: embedding if available, Simhash fallback.

    Returns similarity score 0-1.
    """
    if embed_fn is not None and existing_texts:
        sim, _ = await detect_duplicate_risk_embedding(
            text, existing_texts, embed_fn, embedding_threshold,
        )
        if sim > 0:
            return sim

    return detect_duplicate_risk_simhash(text, existing_texts)


def estimate_content_to_html_ratio(html: str) -> float:
    """Estimate text content vs HTML markup ratio."""
    text_len = len(re.sub(r"<[^>]+>", "", html).strip())
    if len(html) == 0:
        return 0.0
    return text_len / len(html)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
