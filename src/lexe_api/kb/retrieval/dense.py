"""
LEXE Knowledge Base - Dense Search

Ricerca vettoriale con pgvector HNSW.
Supporta partial indexes per multi-model benchmark.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from ..config import EMBEDDING_DIMS, EmbeddingChannel, EmbeddingModel

logger = structlog.get_logger(__name__)


@dataclass
class DenseSearchResult:
    """Risultato ricerca dense."""

    massima_id: UUID
    score: float  # Cosine similarity (0-1)
    rank: int
    model: EmbeddingModel
    channel: EmbeddingChannel


@dataclass
class DenseSearchConfig:
    """Configurazione dense search."""

    model: EmbeddingModel = EmbeddingModel.QWEN3
    channel: EmbeddingChannel = EmbeddingChannel.TESTO
    limit: int = 50
    min_similarity: float = 0.5
    use_ivfflat: bool = False  # True per dataset grandi


async def dense_search(
    query_embedding: list[float],
    config: DenseSearchConfig,
    db_pool: Any,
    filters: dict[str, Any] | None = None,
) -> list[DenseSearchResult]:
    """
    Ricerca dense con pgvector.

    Usa partial HNSW index per modello/canale specificato.

    Args:
        query_embedding: Embedding della query
        config: Configurazione ricerca
        db_pool: Connection pool database
        filters: Filtri opzionali (anno, tipo, sezione, etc.)

    Returns:
        Lista risultati ordinati per similarita'
    """
    dims = EMBEDDING_DIMS.get(config.model, len(query_embedding))

    # Costruisci query con partial index hint
    # L'index viene usato automaticamente grazie a WHERE model = X
    base_query = """
    SELECT
        e.massima_id,
        1 - (e.embedding <=> $1::vector({dims})) as similarity
    FROM kb.embeddings e
    JOIN kb.massime m ON e.massima_id = m.id
    WHERE e.model = $2
      AND e.channel = $3
      AND 1 - (e.embedding <=> $1::vector({dims})) >= $4
    """.format(dims=dims)

    params = [
        query_embedding,
        config.model.value,
        config.channel.value,
        config.min_similarity,
    ]
    param_idx = 5

    # Aggiungi filtri
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

        if filters.get("sezione"):
            base_query += f" AND m.sezione = ${param_idx}"
            params.append(filters["sezione"])
            param_idx += 1

    # Order e limit
    base_query += f"""
    ORDER BY similarity DESC
    LIMIT ${param_idx}
    """
    params.append(config.limit)

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(base_query, *params)

        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                DenseSearchResult(
                    massima_id=row["massima_id"],
                    score=float(row["similarity"]),
                    rank=rank,
                    model=config.model,
                    channel=config.channel,
                )
            )

        logger.debug(
            "Dense search completed",
            model=config.model.value,
            channel=config.channel.value,
            results=len(results),
            top_score=results[0].score if results else 0,
        )

        return results

    except Exception as e:
        logger.error(
            "Dense search failed",
            model=config.model.value,
            error=str(e),
        )
        raise


async def dense_search_multi_model(
    query_embeddings: dict[EmbeddingModel, list[float]],
    channel: EmbeddingChannel,
    limit: int,
    min_similarity: float,
    db_pool: Any,
    filters: dict[str, Any] | None = None,
) -> dict[EmbeddingModel, list[DenseSearchResult]]:
    """
    Ricerca dense con multipli modelli (per benchmark A/B).

    Args:
        query_embeddings: Dict model -> embedding
        channel: Canale da usare
        limit: Limite risultati per modello
        min_similarity: Soglia minima
        db_pool: Connection pool
        filters: Filtri

    Returns:
        Dict model -> risultati
    """
    import asyncio

    async def search_single(model: EmbeddingModel, embedding: list[float]):
        config = DenseSearchConfig(
            model=model,
            channel=channel,
            limit=limit,
            min_similarity=min_similarity,
        )
        return model, await dense_search(embedding, config, db_pool, filters)

    tasks = [
        search_single(model, emb)
        for model, emb in query_embeddings.items()
    ]

    results = await asyncio.gather(*tasks)
    return dict(results)


def estimate_recall(
    results: list[DenseSearchResult],
    ground_truth_ids: set[UUID],
    k: int = 20,
) -> float:
    """
    Calcola Recall@K per valutazione.

    Args:
        results: Risultati ricerca
        ground_truth_ids: ID massime rilevanti (ground truth)
        k: Top K da considerare

    Returns:
        Recall score (0-1)
    """
    if not ground_truth_ids:
        return 0.0

    retrieved_ids = {r.massima_id for r in results[:k]}
    relevant_retrieved = retrieved_ids & ground_truth_ids

    return len(relevant_retrieved) / len(ground_truth_ids)


def estimate_mrr(
    results: list[DenseSearchResult],
    ground_truth_ids: set[UUID],
    k: int = 10,
) -> float:
    """
    Calcola MRR@K (Mean Reciprocal Rank).

    Args:
        results: Risultati ricerca
        ground_truth_ids: ID massime rilevanti
        k: Top K da considerare

    Returns:
        MRR score (0-1)
    """
    for i, result in enumerate(results[:k]):
        if result.massima_id in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0
