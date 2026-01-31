"""
LEXE Knowledge Base - Norm Booster

Lightweight reranker that boosts results citing norms mentioned in query.

Unlike GraphRAG reranking, this is additive and doesn't expand the result set.
It only boosts existing hybrid results that cite the detected norm.

Usage:
    query = "responsabilità extracontrattuale art. 2043 c.c."
    results = await hybrid_search(...)
    boosted = await boost_by_norm(query, results, conn)

v1.0: Initial implementation
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from lexe_api.kb.graph.norm_extractor import (
    norm_to_canonical_id,
    parse_norm_query,
)
from .hybrid import HybridSearchResult

logger = structlog.get_logger(__name__)


@dataclass
class NormBoostConfig:
    """Configuration for norm boosting."""
    boost_factor: float = 0.10  # Additive boost (conservative)
    citation_count_weight: float = 0.02  # Boost per 100 citations
    max_boost: float = 0.20  # Cap on total boost


@dataclass
class NormBoostedResult:
    """Hybrid result with norm boost applied."""

    massima_id: UUID
    original_score: float
    original_rank: int
    norm_boost: float
    final_score: float
    final_rank: int

    # Norm info
    cites_norm: bool = False
    norm_id: str | None = None
    norm_citation_count: int = 0

    # Original component scores
    dense_score: float | None = None
    bm25_score: float | None = None


async def get_norm_citing_massime(
    norm_id: str,
    conn: Any,
) -> tuple[set[UUID], int]:
    """
    Get all massima IDs that cite a specific norm.

    Returns:
        (set of massima_ids, norm's total citation_count)
    """
    # Get norm citation count
    norm_row = await conn.fetchrow(
        "SELECT citation_count FROM kb.norms WHERE id = $1",
        norm_id,
    )

    if not norm_row:
        return set(), 0

    # Get all massime citing this norm
    rows = await conn.fetch(
        """
        SELECT massima_id
        FROM kb.massima_norms
        WHERE norm_id = $1
        """,
        norm_id,
    )

    massima_ids = {row["massima_id"] for row in rows}
    return massima_ids, norm_row["citation_count"]


async def boost_by_norm(
    query: str,
    hybrid_results: list[HybridSearchResult],
    conn: Any,
    config: NormBoostConfig | None = None,
) -> list[NormBoostedResult]:
    """
    Boost hybrid results that cite norms mentioned in query.

    This is a lightweight post-processor that:
    1. Detects norm references in query
    2. Finds which results cite that norm
    3. Applies additive boost to those results
    4. Re-sorts by final_score

    Args:
        query: User query
        hybrid_results: Results from hybrid_search()
        conn: Database connection
        config: Boost configuration

    Returns:
        Results with norm boost applied
    """
    if config is None:
        config = NormBoostConfig()

    if not hybrid_results:
        return []

    # Detect norm in query
    norm_dict = parse_norm_query(query)

    if not norm_dict:
        # No norm detected - return results as-is with NormBoostedResult wrapper
        return [
            NormBoostedResult(
                massima_id=r.massima_id,
                original_score=r.rrf_score,
                original_rank=r.final_rank,
                norm_boost=0.0,
                final_score=r.rrf_score,
                final_rank=r.final_rank,
                cites_norm=False,
                dense_score=r.dense_score,
                bm25_score=r.bm25_score,
            )
            for r in hybrid_results
        ]

    # Get norm info
    norm_id = norm_to_canonical_id(norm_dict)
    citing_massime, citation_count = await get_norm_citing_massime(norm_id, conn)

    if not citing_massime:
        logger.debug("Norm not found or no citations", norm_id=norm_id)
        return [
            NormBoostedResult(
                massima_id=r.massima_id,
                original_score=r.rrf_score,
                original_rank=r.final_rank,
                norm_boost=0.0,
                final_score=r.rrf_score,
                final_rank=r.final_rank,
                cites_norm=False,
                norm_id=norm_id,
                dense_score=r.dense_score,
                bm25_score=r.bm25_score,
            )
            for r in hybrid_results
        ]

    # Compute boost based on norm popularity
    # More cited norms get slightly higher boost (indicates important norm)
    popularity_bonus = min(citation_count / 100, 1.0) * config.citation_count_weight

    # Apply boost to citing results
    results = []
    boosted_count = 0

    for r in hybrid_results:
        cites = r.massima_id in citing_massime

        if cites:
            norm_boost = min(
                config.boost_factor + popularity_bonus,
                config.max_boost,
            )
            boosted_count += 1
        else:
            norm_boost = 0.0

        final_score = r.rrf_score + norm_boost

        results.append(
            NormBoostedResult(
                massima_id=r.massima_id,
                original_score=r.rrf_score,
                original_rank=r.final_rank,
                norm_boost=norm_boost,
                final_score=final_score,
                final_rank=0,  # Set after sorting
                cites_norm=cites,
                norm_id=norm_id,
                norm_citation_count=citation_count if cites else 0,
                dense_score=r.dense_score,
                bm25_score=r.bm25_score,
            )
        )

    # Re-sort by final score
    results.sort(key=lambda x: x.final_score, reverse=True)

    # Assign final ranks
    for i, r in enumerate(results, start=1):
        r.final_rank = i

    # Calculate reordering stats
    reordered = sum(1 for r in results if r.final_rank != r.original_rank)

    logger.info(
        "Norm boost applied",
        norm_id=norm_id,
        citation_count=citation_count,
        candidates=len(results),
        boosted=boosted_count,
        reordered=reordered,
        avg_boost=f"{sum(r.norm_boost for r in results) / len(results):.4f}" if results else "0",
    )

    return results


async def hybrid_search_with_norm_boost(
    query: str,
    query_embedding: list[float],
    hybrid_config: Any,
    conn: Any,
    norm_config: NormBoostConfig | None = None,
) -> list[NormBoostedResult]:
    """
    Full hybrid search + norm boost pipeline.

    Args:
        query: User query
        query_embedding: Query embedding vector
        hybrid_config: HybridSearchConfig
        conn: Database connection
        norm_config: NormBoostConfig (None = default config)

    Returns:
        Norm-boosted hybrid results
    """
    from .hybrid import hybrid_search

    # Step 1: Hybrid search
    hybrid_results = await hybrid_search(
        query,
        query_embedding,
        hybrid_config,
        conn,
    )

    if not hybrid_results:
        return []

    # Step 2: Apply norm boost
    return await boost_by_norm(query, hybrid_results, conn, norm_config)
