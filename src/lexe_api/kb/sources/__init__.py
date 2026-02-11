# lexe_api/kb/sources/__init__.py
"""
Legal Source Adapters for KB Ingestion.

Source hierarchy (Trust Levels):
- TIER A (Canonical): Normattiva, Gazzetta Ufficiale
- TIER B (Editorial): Brocardi, StudioCataldi
- TIER C (Optional): Altalex (if available)
"""

from lexe_api.kb.sources.base_adapter import (
    BaseLegalSourceAdapter,
    TrustLevel,
)
from lexe_api.kb.sources.models import (
    ArticleExtract,
    BrocardiExtract,
    DizionarioExtract,
    ValidationResult,
)

__all__ = [
    "BaseLegalSourceAdapter",
    "TrustLevel",
    "ArticleExtract",
    "BrocardiExtract",
    "DizionarioExtract",
    "ValidationResult",
]
