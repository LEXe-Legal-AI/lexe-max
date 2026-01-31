"""
Category Classifier for Massime

Keyword-based classification with confidence scoring.
- L1: Always assigned (highest confidence)
- L2: Assigned if confidence >= 0.70
"""

import re
from dataclasses import dataclass
from typing import Optional

from .categories import (
    ALL_CATEGORIES,
    CategoryDef,
    get_l1_categories,
    get_l2_by_parent,
)


@dataclass
class CategoryMatch:
    """Result of category matching."""
    category_id: str
    level: int
    parent_id: Optional[str]
    confidence: float
    evidence_terms: list[str]
    method: str = "keyword"


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _match_keywords(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """
    Match keywords against text.
    Returns (match_count, matched_terms).
    """
    text_norm = _normalize_text(text)
    matched = []

    for kw in keywords:
        kw_norm = kw.lower()
        # Use word boundaries for single words, substring for phrases
        if " " in kw_norm or "." in kw_norm:
            # Phrase or article reference - substring match
            if kw_norm in text_norm:
                matched.append(kw)
        else:
            # Single word - word boundary match
            pattern = rf"\b{re.escape(kw_norm)}\b"
            if re.search(pattern, text_norm):
                matched.append(kw)

    return len(matched), matched


def _calculate_confidence(match_count: int, total_keywords: int, level: int) -> float:
    """
    Calculate confidence score.

    Formula:
    - L1: match_count / min(total, 20) + 0.20 boost
    - L2: match_count / min(total, 8) (more lenient for subcategories)
    - Cap at 0.95

    Examples for L2 with 15 keywords:
    - 2 matches → 2/8 * 1.5 = 0.375
    - 3 matches → 3/8 * 1.5 = 0.563
    - 4 matches → 4/8 * 1.5 = 0.75
    - 5 matches → 5/8 * 1.5 = 0.94
    """
    if total_keywords == 0:
        return 0.0

    # Different denominators for L1 vs L2
    if level == 1:
        effective_total = min(total_keywords, 20)
        boost = 0.20
    else:
        # L2 is more lenient - need fewer matches to reach threshold
        effective_total = min(total_keywords, 8)
        boost = 0.0

    base = match_count / effective_total + boost

    # Apply scaling for smoother distribution
    scaled = min(base * 1.5, 0.95)

    return round(scaled, 3)


def classify_massima(text: str, min_l2_confidence: float = 0.50) -> list[CategoryMatch]:
    """
    Classify a massima text into categories.

    Returns list of CategoryMatch, sorted by level then confidence.
    - Always includes best L1 match
    - Includes L2 matches with confidence >= min_l2_confidence

    Args:
        text: Massima text (testo or testo_breve)
        min_l2_confidence: Minimum confidence for L2 assignment (default 0.70)

    Returns:
        List of CategoryMatch objects
    """
    if not text or len(text.strip()) < 50:
        return []

    results: list[CategoryMatch] = []

    # Match all L1 categories
    l1_matches: list[tuple[CategoryDef, int, list[str]]] = []
    for cat in get_l1_categories():
        match_count, matched_terms = _match_keywords(text, cat.keywords)
        if match_count > 0:
            l1_matches.append((cat, match_count, matched_terms))

    if not l1_matches:
        # No L1 match - unusual, return empty
        return []

    # Sort L1 by match count descending
    l1_matches.sort(key=lambda x: x[1], reverse=True)

    # Take best L1
    best_l1, best_count, best_terms = l1_matches[0]
    l1_confidence = _calculate_confidence(best_count, len(best_l1.keywords), 1)

    results.append(CategoryMatch(
        category_id=best_l1.id,
        level=1,
        parent_id=None,
        confidence=l1_confidence,
        evidence_terms=best_terms[:10],  # Top 10 terms
        method="keyword",
    ))

    # Match L2 subcategories under best L1
    l2_candidates = get_l2_by_parent(best_l1.id)

    for cat in l2_candidates:
        match_count, matched_terms = _match_keywords(text, cat.keywords)
        if match_count == 0:
            continue

        confidence = _calculate_confidence(match_count, len(cat.keywords), 2)

        if confidence >= min_l2_confidence:
            results.append(CategoryMatch(
                category_id=cat.id,
                level=2,
                parent_id=best_l1.id,
                confidence=confidence,
                evidence_terms=matched_terms[:10],
                method="keyword",
            ))

    # Sort by level, then confidence descending
    results.sort(key=lambda x: (x.level, -x.confidence))

    return results


def classify_massima_multi_l1(text: str, max_l1: int = 2) -> list[CategoryMatch]:
    """
    Classify with multiple L1 if confidence is close.

    Use this for massime that span multiple domains (e.g., civil + procedural).
    """
    if not text or len(text.strip()) < 50:
        return []

    results: list[CategoryMatch] = []

    # Match all L1 categories
    l1_matches: list[tuple[CategoryDef, int, list[str], float]] = []
    for cat in get_l1_categories():
        match_count, matched_terms = _match_keywords(text, cat.keywords)
        if match_count > 0:
            confidence = _calculate_confidence(match_count, len(cat.keywords), 1)
            l1_matches.append((cat, match_count, matched_terms, confidence))

    if not l1_matches:
        return []

    # Sort by confidence descending
    l1_matches.sort(key=lambda x: x[3], reverse=True)

    # Take top L1 matches (within 0.15 of best)
    best_confidence = l1_matches[0][3]
    threshold = best_confidence - 0.15

    selected_l1 = []
    for cat, count, terms, conf in l1_matches[:max_l1]:
        if conf >= threshold:
            selected_l1.append((cat, count, terms, conf))

    # Add L1 results
    for cat, count, terms, confidence in selected_l1:
        results.append(CategoryMatch(
            category_id=cat.id,
            level=1,
            parent_id=None,
            confidence=confidence,
            evidence_terms=terms[:10],
            method="keyword",
        ))

    # Add L2 for primary L1 only
    primary_l1 = selected_l1[0][0]
    for cat in get_l2_by_parent(primary_l1.id):
        match_count, matched_terms = _match_keywords(text, cat.keywords)
        if match_count == 0:
            continue

        confidence = _calculate_confidence(match_count, len(cat.keywords), 2)

        if confidence >= 0.70:
            results.append(CategoryMatch(
                category_id=cat.id,
                level=2,
                parent_id=primary_l1.id,
                confidence=confidence,
                evidence_terms=matched_terms[:10],
                method="keyword",
            ))

    results.sort(key=lambda x: (x.level, -x.confidence))
    return results
