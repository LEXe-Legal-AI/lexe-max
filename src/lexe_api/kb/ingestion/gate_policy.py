"""
LEXE Knowledge Base - Gate Policy with Structured Logging

Evaluates whether extracted text elements should be accepted or rejected
as massime, with structured reason codes and numeric details.

Usage:
    from lexe_api.kb.ingestion.gate_policy import evaluate_gate, GateConfig, GateResult

    config = GateConfig()
    result = evaluate_gate(text, element_type="NarrativeText", page=5, config=config)
    if result.accepted:
        # process massima
    else:
        # log rejection with result.reason and result.details
"""

import re
from dataclasses import dataclass, field


@dataclass
class GateConfig:
    """Configuration for gate policy evaluation."""

    min_length: int = 150
    max_citation_ratio: float = 0.03
    bad_starts: list[str] = field(
        default_factory=lambda: [", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO"]
    )
    skip_element_types: list[str] = field(
        default_factory=lambda: ["Header", "Footer", "PageNumber"]
    )
    skip_pages: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "min_length": self.min_length,
            "max_citation_ratio": self.max_citation_ratio,
            "bad_starts": self.bad_starts,
            "skip_element_types": self.skip_element_types,
            "skip_pages": sorted(self.skip_pages),
        }


@dataclass
class GateResult:
    """Result of gate policy evaluation."""

    accepted: bool
    reason: str | None = None
    details: dict | None = None

    @property
    def decision(self) -> str:
        """Returns 'accepted' or 'rejected' for DB ENUM."""
        return "accepted" if self.accepted else "rejected"


# Citation counting pattern (from ingest_staging.py)
_CITATION_COUNT_PATTERN = re.compile(r"Cass\.|Sez\.\s*\d|n\.\s*\d+|Rv\.\s*\d+")


def evaluate_gate(
    text: str,
    element_type: str = "NarrativeText",
    page: int = 0,
    config: GateConfig | None = None,
) -> GateResult:
    """
    Evaluate whether a text element passes the gate policy.

    Reason taxonomy (structured codes):
    - header_footer: element_type is Header/Footer/PageNumber
    - skip_page: page is in skip_pages set
    - too_short: char_count < min_length
    - too_citation_dense: citation_ratio > max_citation_ratio
    - bad_start: text starts with a known bad pattern

    Args:
        text: The text to evaluate
        element_type: Unstructured element category
        page: Page number (0-indexed)
        config: Gate configuration (uses defaults if None)

    Returns:
        GateResult with accepted/rejected, reason, and numeric details
    """
    if config is None:
        config = GateConfig()

    text = text.strip() if text else ""

    # 1. Header/Footer filter
    if element_type in config.skip_element_types:
        return GateResult(
            accepted=False,
            reason="header_footer",
            details={
                "element_type": element_type,
                "skip_types": config.skip_element_types,
            },
        )

    # 2. Skip pages
    if page > 0 and page in config.skip_pages:
        return GateResult(
            accepted=False,
            reason="skip_page",
            details={"page": page, "skip_pages_count": len(config.skip_pages)},
        )

    # 3. Too short
    char_count = len(text)
    if char_count < config.min_length:
        return GateResult(
            accepted=False,
            reason="too_short",
            details={
                "threshold": config.min_length,
                "actual": char_count,
            },
        )

    # 4. Citation density
    citations = len(_CITATION_COUNT_PATTERN.findall(text))
    words = len(text.split())
    citation_ratio = citations / words if words > 0 else 0.0
    if citation_ratio > config.max_citation_ratio:
        return GateResult(
            accepted=False,
            reason="too_citation_dense",
            details={
                "threshold": config.max_citation_ratio,
                "actual": round(citation_ratio, 4),
                "citation_count": citations,
                "word_count": words,
            },
        )

    # 5. Bad starts
    for bad in config.bad_starts:
        if text.startswith(bad):
            return GateResult(
                accepted=False,
                reason="bad_start",
                details={
                    "matched_pattern": bad,
                    "text_start": text[:50],
                },
            )

    # All checks passed
    return GateResult(accepted=True)
