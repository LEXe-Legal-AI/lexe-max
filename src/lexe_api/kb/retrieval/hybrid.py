"""
LEXE Knowledge Base - Hybrid Search

RRF (Reciprocal Rank Fusion) per combinare Dense + BM25 + trgm.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from ..config import EmbeddingChannel, EmbeddingModel, HybridSearchConfig
from .dense import DenseSearchConfig, dense_search
from .sparse import bm25_search, trgm_search

logger = structlog.get_logger(__name__)


@dataclass
class HybridSearchResult:
    """Risultato ricerca ibrida."""

    massima_id: UUID
    rrf_score: float
    final_rank: int

    # Component scores
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    trgm_score: float | None = None
    trgm_rank: int | None = None

    # Metadata
    model: EmbeddingModel | None = None
    channel: EmbeddingChannel | None = None


def reciprocal_rank_fusion(
    *result_lists: list[Any],
    k: int = 60,
    id_extractor: callable = lambda x: x.massima_id,
) -> list[tuple[UUID, float, dict[str, int]]]:
    """
    RRF fusion di multiple liste risultati.

    Formula: RRF(d) = sum(1 / (k + rank(d)))

    Args:
        result_lists: Liste di risultati da fondere
        k: Parametro RRF (default 60)
        id_extractor: Funzione per estrarre ID da risultato

    Returns:
        Lista di (id, rrf_score, ranks_per_list)
    """
    scores: dict[UUID, float] = {}
    ranks: dict[UUID, dict[str, int]] = {}

    for list_idx, result_list in enumerate(result_lists):
        list_name = f"list_{list_idx}"
        for result in result_list:
            doc_id = id_extractor(result)
            rank = result.rank

            # RRF score
            rrf_contribution = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution

            # Track rank per list
            if doc_id not in ranks:
                ranks[doc_id] = {}
            ranks[doc_id][list_name] = rank

    # Sort by RRF score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_id, score, ranks.get(doc_id, {})) for doc_id, score in sorted_results]


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    config: HybridSearchConfig,
    db_pool: Any,
    filters: dict[str, Any] | None = None,
) -> list[HybridSearchResult]:
    """
    Ricerca ibrida: Dense + BM25 + trgm con RRF fusion.

    Args:
        query: Query testuale
        query_embedding: Embedding della query
        config: Configurazione hybrid search
        db_pool: Connection pool
        filters: Filtri opzionali

    Returns:
        Lista risultati fusi e ordinati
    """
    import asyncio

    # Esegui le 3 ricerche in parallelo
    dense_config = DenseSearchConfig(
        model=config.model,
        channel=config.channel,
        limit=config.dense_limit,
        min_similarity=config.min_similarity,
    )

    dense_task = dense_search(query_embedding, dense_config, db_pool, filters)
    bm25_task = bm25_search(query, config.bm25_limit, db_pool, filters)
    trgm_task = trgm_search(query, config.trgm_limit, db_pool, filters=filters)

    dense_results, bm25_results, trgm_results = await asyncio.gather(
        dense_task, bm25_task, trgm_task
    )

    logger.debug(
        "Component searches completed",
        dense=len(dense_results),
        bm25=len(bm25_results),
        trgm=len(trgm_results),
    )

    # RRF Fusion
    fused = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        trgm_results,
        k=config.rrf_k,
    )

    # Costruisci risultati con score components
    # Crea lookup per score
    dense_lookup = {r.massima_id: r for r in dense_results}
    bm25_lookup = {r.massima_id: r for r in bm25_results}
    trgm_lookup = {r.massima_id: r for r in trgm_results}

    results = []
    for rank, (doc_id, rrf_score, _) in enumerate(fused[: config.final_limit], start=1):
        dense_r = dense_lookup.get(doc_id)
        bm25_r = bm25_lookup.get(doc_id)
        trgm_r = trgm_lookup.get(doc_id)

        results.append(
            HybridSearchResult(
                massima_id=doc_id,
                rrf_score=rrf_score,
                final_rank=rank,
                dense_score=dense_r.score if dense_r else None,
                dense_rank=dense_r.rank if dense_r else None,
                bm25_score=bm25_r.score if bm25_r else None,
                bm25_rank=bm25_r.rank if bm25_r else None,
                trgm_score=trgm_r.score if trgm_r else None,
                trgm_rank=trgm_r.rank if trgm_r else None,
                model=config.model,
                channel=config.channel,
            )
        )

    logger.info(
        "Hybrid search completed",
        total_fused=len(fused),
        returned=len(results),
        top_rrf=results[0].rrf_score if results else 0,
    )

    return results


async def hybrid_search_multi_model(
    query: str,
    query_embeddings: dict[EmbeddingModel, list[float]],
    config: HybridSearchConfig,
    db_pool: Any,
    filters: dict[str, Any] | None = None,
) -> dict[EmbeddingModel, list[HybridSearchResult]]:
    """
    Hybrid search con multipli modelli embedding (per benchmark).

    Args:
        query: Query testuale
        query_embeddings: Dict model -> embedding
        config: Configurazione base
        db_pool: Connection pool
        filters: Filtri

    Returns:
        Dict model -> risultati hybrid
    """
    import asyncio

    async def search_with_model(model: EmbeddingModel, embedding: list[float]):
        model_config = HybridSearchConfig(
            dense_limit=config.dense_limit,
            bm25_limit=config.bm25_limit,
            trgm_limit=config.trgm_limit,
            rrf_k=config.rrf_k,
            final_limit=config.final_limit,
            min_similarity=config.min_similarity,
            model=model,
            channel=config.channel,
        )
        results = await hybrid_search(query, embedding, model_config, db_pool, filters)
        return model, results

    tasks = [search_with_model(model, emb) for model, emb in query_embeddings.items()]

    results = await asyncio.gather(*tasks)
    return dict(results)


def calculate_precision_at_k(
    results: list[HybridSearchResult],
    ground_truth_ids: set[UUID],
    k: int = 5,
) -> float:
    """
    Calcola Precision@K.

    Args:
        results: Risultati ricerca
        ground_truth_ids: ID rilevanti (ground truth)
        k: Top K da considerare

    Returns:
        Precision score (0-1)
    """
    if not results or k == 0:
        return 0.0

    top_k_ids = {r.massima_id for r in results[:k]}
    relevant_in_top_k = top_k_ids & ground_truth_ids

    return len(relevant_in_top_k) / k
