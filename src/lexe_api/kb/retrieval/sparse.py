"""
LEXE Knowledge Base - Sparse Search

BM25 con pg_search (ParadeDB) + trgm typo-catch.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SparseSearchResult:
    """Risultato ricerca sparse."""

    massima_id: UUID
    score: float
    rank: int
    method: str  # "bm25" o "trgm"


@dataclass
class SparseSearchConfig:
    """Configurazione sparse search."""

    bm25_limit: int = 50
    trgm_limit: int = 20
    min_trgm_similarity: float = 0.3


async def bm25_search(
    query: str,
    limit: int,
    db_pool: Any,
    filters: dict[str, Any] | None = None,
) -> list[SparseSearchResult]:
    """
    Ricerca BM25 con pg_search (ParadeDB).

    Fallback a tsvector FTS se pg_search non disponibile.

    Args:
        query: Query testuale
        limit: Limite risultati
        db_pool: Connection pool
        filters: Filtri opzionali

    Returns:
        Lista risultati ordinati per BM25 score
    """
    # Prima prova pg_search
    pg_search_query = """
    SELECT
        m.id as massima_id,
        paradedb.score(m.id) as bm25_score
    FROM kb.massime m
    WHERE m.id @@@ paradedb.search(
        query => paradedb.parse($1),
        index => 'massime_bm25'
    )
    """

    # Fallback tsvector query
    tsvector_query = """
    SELECT
        m.id as massima_id,
        ts_rank_cd(m.testo_tsv, plainto_tsquery('italian', $1)) as bm25_score
    FROM kb.massime m
    WHERE m.testo_tsv @@ plainto_tsquery('italian', $1)
    """

    params = [query]
    param_idx = 2

    # Costruisci filtri
    filter_clause = ""
    if filters:
        if filters.get("anno_min"):
            filter_clause += f" AND m.anno >= ${param_idx}"
            params.append(filters["anno_min"])
            param_idx += 1

        if filters.get("anno_max"):
            filter_clause += f" AND m.anno <= ${param_idx}"
            params.append(filters["anno_max"])
            param_idx += 1

        if filters.get("tipo"):
            filter_clause += f" AND m.tipo = ${param_idx}"
            params.append(filters["tipo"])
            param_idx += 1

        if filters.get("sezione"):
            filter_clause += f" AND m.sezione = ${param_idx}"
            params.append(filters["sezione"])
            param_idx += 1

    order_limit = f"""
    ORDER BY bm25_score DESC
    LIMIT ${param_idx}
    """
    params.append(limit)

    try:
        async with db_pool.acquire() as conn:
            # Prova pg_search
            try:
                full_query = pg_search_query + filter_clause + order_limit
                rows = await conn.fetch(full_query, *params)
                method = "bm25"
            except Exception as pg_search_err:
                # Fallback a tsvector
                logger.debug(
                    "pg_search not available, using tsvector",
                    error=str(pg_search_err)[:100],
                )
                full_query = tsvector_query + filter_clause + order_limit
                rows = await conn.fetch(full_query, *params)
                method = "tsvector"

        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                SparseSearchResult(
                    massima_id=row["massima_id"],
                    score=float(row["bm25_score"]),
                    rank=rank,
                    method=method,
                )
            )

        logger.debug(
            "BM25 search completed",
            method=method,
            results=len(results),
            top_score=results[0].score if results else 0,
        )

        return results

    except Exception as e:
        logger.error("BM25 search failed", error=str(e))
        raise


async def trgm_search(
    query: str,
    limit: int,
    db_pool: Any,
    min_similarity: float = 0.3,
    filters: dict[str, Any] | None = None,
) -> list[SparseSearchResult]:
    """
    Ricerca typo-tolerant con pg_trgm.

    Utile per abbreviazioni e errori di battitura.

    Args:
        query: Query testuale
        limit: Limite risultati
        db_pool: Connection pool
        min_similarity: Soglia minima similarita' trgm
        filters: Filtri opzionali

    Returns:
        Lista risultati
    """
    # Normalizza query per trgm
    query_normalized = query.lower().strip()

    base_query = """
    SELECT
        m.id as massima_id,
        similarity(m.testo_normalizzato, $1) as trgm_score
    FROM kb.massime m
    WHERE similarity(m.testo_normalizzato, $1) >= $2
    """

    params = [query_normalized, min_similarity]
    param_idx = 3

    # Filtri
    if filters:
        if filters.get("anno_min"):
            base_query += f" AND m.anno >= ${param_idx}"
            params.append(filters["anno_min"])
            param_idx += 1

        if filters.get("anno_max"):
            base_query += f" AND m.anno <= ${param_idx}"
            params.append(filters["anno_max"])
            param_idx += 1

        if filters.get("tipo"):
            base_query += f" AND m.tipo = ${param_idx}"
            params.append(filters["tipo"])
            param_idx += 1

    base_query += f"""
    ORDER BY trgm_score DESC
    LIMIT ${param_idx}
    """
    params.append(limit)

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(base_query, *params)

        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                SparseSearchResult(
                    massima_id=row["massima_id"],
                    score=float(row["trgm_score"]),
                    rank=rank,
                    method="trgm",
                )
            )

        logger.debug(
            "Trgm search completed",
            results=len(results),
            top_score=results[0].score if results else 0,
        )

        return results

    except Exception as e:
        logger.error("Trgm search failed", error=str(e))
        raise


async def keyword_boost_search(
    query: str,
    keywords: list[str],
    limit: int,
    db_pool: Any,
    keyword_boost: float = 1.5,
) -> list[SparseSearchResult]:
    """
    BM25 con boost per keyword match.

    Args:
        query: Query principale
        keywords: Keyword da boostare
        limit: Limite risultati
        db_pool: Connection pool
        keyword_boost: Fattore boost per keyword match

    Returns:
        Risultati con boost applicato
    """
    # Prima esegui BM25 standard
    results = await bm25_search(query, limit * 2, db_pool)

    if not keywords:
        return results[:limit]

    # Applica boost per keyword
    keywords_lower = {k.lower() for k in keywords}

    async with db_pool.acquire() as conn:
        for result in results:
            # Fetch testo massima per check keywords
            row = await conn.fetchrow(
                "SELECT keywords FROM kb.massime WHERE id = $1",
                result.massima_id,
            )
            if row and row["keywords"]:
                massima_keywords = {k.lower() for k in row["keywords"]}
                overlap = len(keywords_lower & massima_keywords)
                if overlap > 0:
                    result.score *= keyword_boost**overlap

    # Riordina
    results.sort(key=lambda x: x.score, reverse=True)

    # Aggiorna rank
    for i, r in enumerate(results):
        r.rank = i + 1

    return results[:limit]
