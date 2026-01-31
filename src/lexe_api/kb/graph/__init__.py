"""
LEXE Knowledge Base - Graph Module

Citation Graph, Topic Classification, GraphRAG Reranking.

Modules:
    - citation_extractor: Two-step citation extraction (extract -> resolve)
    - edge_builder: Dual-write builder (SQL + AGE)
    - categories: Category definitions and hierarchy
    - classifier: Hybrid topic classification
    - norm_extractor: Norm canonicalization
    - overrule_detector: Turning points detection
    - graph_reranker: GraphRAG cached reranking
"""

from enum import Enum


class EdgeType(str, Enum):
    """Tipi di edge nel citation graph."""

    CITES = "CITES"  # Citazione generica a precedente
    CITES_NORM = "CITES_NORM"  # Citazione a norma


class RelationSubtype(str, Enum):
    """Sottotipi di relazione per CITES."""

    CONFIRMS = "CONFIRMS"  # Conforme, nello stesso senso
    DISTINGUISHES = "DISTINGUISHES"  # Distingue, fattispecie diversa
    OVERRULES = "OVERRULES"  # Contra, in senso contrario, supera


class TurningPointType(str, Enum):
    """Tipi di turning point."""

    SEZ_UNITE = "SEZ_UNITE"  # Sezioni Unite risolvono contrasto
    CONTRASTO_RISOLTO = "CONTRASTO_RISOLTO"  # Risoluzione contrasto
    MUTAMENTO = "MUTAMENTO"  # Mutamento orientamento
    ABBANDONO_INDIRIZZO = "ABBANDONO_INDIRIZZO"  # Abbandono indirizzo


class GraphEngine(str, Enum):
    """Engine per query graph (v3.2.1 Miglioria #8)."""

    SQL = "sql"  # Query via SQL tables (default, più performante)
    AGE = "age"  # Query via Apache AGE
    AUTO = "auto"  # Auto-select (currently SQL)


# Relation indicators patterns
RELATION_INDICATORS = {
    RelationSubtype.CONFIRMS: [
        r"conforme",
        r"nello stesso senso",
        r"v\. anche",
        r"cfr\.",
        r"in senso analogo",
        r"in termini",
    ],
    RelationSubtype.DISTINGUISHES: [
        r"diversamente",
        r"va distint[ao]",
        r"non si applica",
        r"fattispecie diversa",
        r"caso diverso",
    ],
    RelationSubtype.OVERRULES: [
        r"contra",
        r"in senso contrario",
        r"superando",
        r"muta(?:ndo)? (?:il )?(?:proprio )?orientamento",
        r"in senso difforme",
        r"abbandona(?:ndo)?",
    ],
}

# Export key types
__all__ = [
    "EdgeType",
    "RelationSubtype",
    "TurningPointType",
    "GraphEngine",
    "RELATION_INDICATORS",
]
