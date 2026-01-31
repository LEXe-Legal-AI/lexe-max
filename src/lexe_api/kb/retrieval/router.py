"""
LEXE Knowledge Base - Query Router

Routes queries to optimal search strategy:
1. Citation queries (Rv./Sez./n.) → Direct DB lookup (fast, precise)
2. Semantic queries → Hybrid search (dense + sparse + RRF)

Citation lookup cascade (same as graph/citation_extractor.py):
1. rv_exact: RV column match
2. rv_text: RV pattern in testo_normalizzato
3. sez_num_anno: (sezione, numero, anno) match
4. num_anno: (numero, anno) match
5. Fallback: hybrid search

v3.2.3: Leverages RV backfill (99% coverage)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# ROUTE TYPES
# ============================================================

class RouteType(str, Enum):
    """Query route classification."""
    CITATION_RV = "citation_rv"           # Has RV reference
    CITATION_SEZ_NUM_ANNO = "citation_sez_num_anno"  # Has Sez/Num/Anno
    CITATION_NUM_ANNO = "citation_num_anno"  # Has Num/Anno only
    SEMANTIC = "semantic"                  # General semantic query


class LookupResult(str, Enum):
    """Lookup resolution result."""
    RV_EXACT = "rv_exact"
    RV_TEXT = "rv_text"
    SEZ_NUM_ANNO = "sez_num_anno"
    NUM_ANNO = "num_anno"
    FALLBACK_HYBRID = "fallback_hybrid"


# ============================================================
# CITATION PATTERNS (aligned with graph/citation_extractor.py)
# ============================================================

# Rv. 639966 or Rv. 639966-01 or Rv.639966
RV_PATTERN = re.compile(r"[Rr]v\.?\s*(\d{5,7})(?:-\d+)?", re.IGNORECASE)

# Sez. Un., n. 12345/2020 or Sez. 1, n. 12345/2020
SEZ_NUM_ANNO_PATTERN = re.compile(
    r"[Ss]ez\.?\s*([A-Za-z0-9\-]+)[\.,\s]+[Nn]\.?\s*(\d+)[/\s]+(\d{4})",
    re.IGNORECASE,
)

# n. 12345/2020 or n. 12345 del 2020
NUM_ANNO_PATTERN = re.compile(
    r"[Nn]\.?\s*(\d+)\s*(?:/|del\s+)(\d{4})",
    re.IGNORECASE,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ParsedCitation:
    """Parsed citation from query."""
    rv: Optional[str] = None
    sezione: Optional[str] = None
    numero: Optional[str] = None
    anno: Optional[int] = None
    raw_match: str = ""

    def is_valid(self) -> bool:
        """Check if citation has enough info for lookup."""
        return bool(self.rv or (self.numero and self.anno))


@dataclass
class RouterResult:
    """Result of query routing."""
    route_type: RouteType
    citation: Optional[ParsedCitation] = None

    # If direct lookup succeeded
    lookup_result: Optional[LookupResult] = None
    massima_ids: list[UUID] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    # Metrics
    lookup_attempted: bool = False
    lookup_hit: bool = False


@dataclass
class RoutedSearchResult:
    """Combined search result with routing info."""
    massima_id: UUID
    score: float
    rank: int
    source: str  # rv_exact, rv_text, sez_num_anno, num_anno, hybrid

    # Component scores (for hybrid)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


# ============================================================
# CITATION PARSING
# ============================================================

def normalize_rv(rv: str) -> str:
    """Normalize RV: strip leading zeros, remove suffix."""
    if not rv:
        return rv
    # Extract just the core digits
    match = re.match(r"0*(\d{5,7})", rv)
    if match:
        return match.group(1)
    return rv


def normalize_sezione(sezione: str) -> str:
    """Normalize sezione format."""
    if not sezione:
        return sezione
    sezione = sezione.strip().upper()
    # Unite variants
    if sezione in ("UN", "UNITE", "U"):
        return "U"
    if sezione in ("LAV", "L"):
        return "L"
    return sezione


def normalize_numero(numero: str) -> str:
    """Normalize numero: strip leading zeros."""
    if not numero:
        return numero
    stripped = numero.lstrip("0")
    return stripped if stripped else "0"


def parse_citation(query: str) -> Optional[ParsedCitation]:
    """
    Parse citation reference from query.

    Returns ParsedCitation if found, None otherwise.
    """
    citation = ParsedCitation()

    # Try RV pattern first (most specific)
    rv_match = RV_PATTERN.search(query)
    if rv_match:
        citation.rv = normalize_rv(rv_match.group(1))
        citation.raw_match = rv_match.group(0)

    # Try Sez + n. + anno pattern
    sez_match = SEZ_NUM_ANNO_PATTERN.search(query)
    if sez_match:
        citation.sezione = normalize_sezione(sez_match.group(1))
        citation.numero = normalize_numero(sez_match.group(2))
        citation.anno = int(sez_match.group(3))
        if not citation.raw_match:
            citation.raw_match = sez_match.group(0)

    # Try n. + anno pattern (less specific)
    if not citation.numero:
        num_match = NUM_ANNO_PATTERN.search(query)
        if num_match:
            citation.numero = normalize_numero(num_match.group(1))
            citation.anno = int(num_match.group(2))
            if not citation.raw_match:
                citation.raw_match = num_match.group(0)

    return citation if citation.is_valid() else None


def classify_query(query: str) -> tuple[RouteType, Optional[ParsedCitation]]:
    """
    Classify query and extract citation if present.

    Returns (route_type, citation or None)
    """
    citation = parse_citation(query)

    if citation:
        if citation.rv:
            return RouteType.CITATION_RV, citation
        elif citation.sezione and citation.numero and citation.anno:
            return RouteType.CITATION_SEZ_NUM_ANNO, citation
        elif citation.numero and citation.anno:
            return RouteType.CITATION_NUM_ANNO, citation

    return RouteType.SEMANTIC, None


# ============================================================
# CITATION LOOKUP
# ============================================================

async def citation_lookup(
    conn: Any,
    citation: ParsedCitation,
    limit: int = 10,
) -> tuple[list[UUID], list[float], LookupResult]:
    """
    Direct database lookup for citation queries.

    Cascade (aligned with graph/citation_extractor.py):
    1. RV column exact match (score 1.0)
    2. RV in text fallback (score 0.98)
    3. Sez+Num+Anno match (score 0.95)
    4. Num+Anno match (score 0.90)

    Returns (massima_ids, scores, lookup_result)
    """
    massima_ids: list[UUID] = []
    scores: list[float] = []
    result_type = LookupResult.FALLBACK_HYBRID

    # Strategy 1: RV column exact match
    if citation.rv:
        rows = await conn.fetch(
            """
            SELECT id, 1.0 as score
            FROM kb.massime
            WHERE is_active = TRUE
            AND rv = $1
            LIMIT $2
            """,
            citation.rv,
            limit,
        )
        if rows:
            massima_ids = [row["id"] for row in rows]
            scores = [row["score"] for row in rows]
            result_type = LookupResult.RV_EXACT
            logger.debug("RV exact match", rv=citation.rv, count=len(rows))
            return massima_ids, scores, result_type

        # Strategy 2: RV in text fallback
        rv_pattern = rf"[Rr]v\.?\s*{citation.rv}"
        rows = await conn.fetch(
            """
            SELECT id, 0.98 as score
            FROM kb.massime
            WHERE is_active = TRUE
            AND testo_normalizzato ~* $1
            LIMIT $2
            """,
            rv_pattern,
            limit,
        )
        if rows:
            massima_ids = [row["id"] for row in rows]
            scores = [row["score"] for row in rows]
            result_type = LookupResult.RV_TEXT
            logger.debug("RV text fallback", rv=citation.rv, count=len(rows))
            return massima_ids, scores, result_type

    # Strategy 3: Sez + Num + Anno
    if citation.sezione and citation.numero and citation.anno:
        rows = await conn.fetch(
            """
            SELECT id, 0.95 as score
            FROM kb.massime
            WHERE is_active = TRUE
            AND UPPER(sezione) = $1
            AND numero = $2
            AND anno = $3
            LIMIT $4
            """,
            citation.sezione,
            citation.numero,
            citation.anno,
            limit,
        )
        if rows:
            massima_ids = [row["id"] for row in rows]
            scores = [row["score"] for row in rows]
            result_type = LookupResult.SEZ_NUM_ANNO
            logger.debug(
                "Sez+Num+Anno match",
                sez=citation.sezione,
                num=citation.numero,
                anno=citation.anno,
                count=len(rows),
            )
            return massima_ids, scores, result_type

    # Strategy 4: Num + Anno only (less precise)
    if citation.numero and citation.anno:
        rows = await conn.fetch(
            """
            SELECT id, 0.90 as score
            FROM kb.massime
            WHERE is_active = TRUE
            AND numero = $1
            AND anno = $2
            LIMIT $3
            """,
            citation.numero,
            citation.anno,
            limit,
        )
        if rows:
            massima_ids = [row["id"] for row in rows]
            scores = [row["score"] for row in rows]
            result_type = LookupResult.NUM_ANNO
            logger.debug(
                "Num+Anno match",
                num=citation.numero,
                anno=citation.anno,
                count=len(rows),
            )
            return massima_ids, scores, result_type

    logger.debug("Citation lookup miss", citation=citation)
    return massima_ids, scores, result_type


# ============================================================
# ROUTER
# ============================================================

async def route_query(
    query: str,
    conn: Any,
    limit: int = 10,
) -> RouterResult:
    """
    Route query to optimal search strategy.

    1. Parse citation from query
    2. If citation found, attempt direct lookup
    3. Return routing result (may include lookup results)

    Args:
        query: User query
        conn: Database connection
        limit: Max results for lookup

    Returns:
        RouterResult with route_type and optional lookup results
    """
    route_type, citation = classify_query(query)

    result = RouterResult(
        route_type=route_type,
        citation=citation,
    )

    # If citation detected, try direct lookup
    if citation:
        result.lookup_attempted = True
        massima_ids, scores, lookup_result = await citation_lookup(
            conn, citation, limit
        )

        if massima_ids:
            result.lookup_hit = True
            result.lookup_result = lookup_result
            result.massima_ids = massima_ids
            result.scores = scores

    logger.info(
        "Query routed",
        route_type=route_type.value,
        citation_detected=citation is not None,
        lookup_hit=result.lookup_hit,
        lookup_result=result.lookup_result.value if result.lookup_result else None,
    )

    return result


async def routed_search(
    query: str,
    query_embedding: list[float],
    conn: Any,
    hybrid_search_fn: Any,  # Callable for hybrid search
    hybrid_config: Any,
    limit: int = 10,
) -> list[RoutedSearchResult]:
    """
    Full routed search with fallback to hybrid.

    1. Route query
    2. If citation lookup hit → return lookup results
    3. Else → run hybrid search

    Args:
        query: User query
        query_embedding: Query embedding vector
        conn: Database connection
        hybrid_search_fn: Hybrid search function
        hybrid_config: Config for hybrid search
        limit: Max results

    Returns:
        List of RoutedSearchResult
    """
    # Route query
    route_result = await route_query(query, conn, limit)

    # If lookup hit, return lookup results
    if route_result.lookup_hit:
        return [
            RoutedSearchResult(
                massima_id=mid,
                score=score,
                rank=i + 1,
                source=route_result.lookup_result.value,
            )
            for i, (mid, score) in enumerate(
                zip(route_result.massima_ids, route_result.scores)
            )
        ]

    # Fallback to hybrid search
    hybrid_results = await hybrid_search_fn(
        query, query_embedding, hybrid_config, conn
    )

    return [
        RoutedSearchResult(
            massima_id=r.massima_id,
            score=r.rrf_score,
            rank=r.final_rank,
            source="hybrid",
            dense_score=r.dense_score,
            sparse_score=r.bm25_score,
        )
        for r in hybrid_results[:limit]
    ]
