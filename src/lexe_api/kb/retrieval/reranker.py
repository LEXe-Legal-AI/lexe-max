"""
LEXE Knowledge Base - Reranker

bge-reranker-v2-m3 per reranking risultati hybrid search.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from .hybrid import HybridSearchResult

logger = structlog.get_logger(__name__)


@dataclass
class RerankedResult:
    """Risultato dopo reranking."""

    massima_id: UUID
    rerank_score: float
    final_score: float
    original_rank: int
    new_rank: int

    # Original scores
    rrf_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class BGEReranker:
    """
    Reranker basato su bge-reranker-v2-m3.

    Usa CrossEncoder per reranking preciso.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Inizializza reranker.

        Args:
            model_name: Nome modello HuggingFace
            device: Device (cpu/cuda/mps)
            batch_size: Batch size per inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy load del modello."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
                logger.info(
                    "Reranker model loaded",
                    model=self.model_name,
                )
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise

    async def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        massima_texts: dict[UUID, str],
        top_k: int = 10,
        rrf_weight: float = 0.4,
    ) -> list[RerankedResult]:
        """
        Rerank risultati hybrid search.

        Args:
            query: Query originale
            results: Risultati da hybrid search
            massima_texts: Dict massima_id -> testo massima
            top_k: Top K risultati da ritornare
            rrf_weight: Peso per RRF score nel final score

        Returns:
            Lista risultati reranked
        """
        import asyncio

        if not results:
            return []

        # Load model (lazy)
        self._load_model()

        # Prepara coppie (query, text) per reranking
        pairs = []
        valid_results = []

        for result in results:
            text = massima_texts.get(result.massima_id)
            if text:
                pairs.append((query, text))
                valid_results.append(result)
            else:
                logger.warning(
                    "Missing text for massima",
                    massima_id=str(result.massima_id),
                )

        if not pairs:
            return []

        # Run reranking in executor (blocking call)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, batch_size=self.batch_size),
        )

        # Combina score
        reranked = []
        for original_result, rerank_score in zip(valid_results, scores):
            # Normalizza rerank score a 0-1 (sigmoid gia' applicato dal modello)
            rerank_score = float(rerank_score)

            # Final score: combinazione weighted
            rrf_norm = original_result.rrf_score / max(r.rrf_score for r in valid_results)
            final_score = (1 - rrf_weight) * rerank_score + rrf_weight * rrf_norm

            reranked.append(
                RerankedResult(
                    massima_id=original_result.massima_id,
                    rerank_score=rerank_score,
                    final_score=final_score,
                    original_rank=original_result.final_rank,
                    new_rank=0,  # Assegnato dopo sort
                    rrf_score=original_result.rrf_score,
                    dense_score=original_result.dense_score,
                    bm25_score=original_result.bm25_score,
                )
            )

        # Sort by final score
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        # Assegna new_rank
        for i, r in enumerate(reranked):
            r.new_rank = i + 1

        logger.info(
            "Reranking completed",
            input_count=len(results),
            output_count=min(top_k, len(reranked)),
            top_rerank_score=reranked[0].rerank_score if reranked else 0,
        )

        return reranked[:top_k]


async def rerank_results(
    query: str,
    results: list[HybridSearchResult],
    db_pool: Any,
    reranker: BGEReranker | None = None,
    top_k: int = 10,
) -> list[RerankedResult]:
    """
    Convenience function per reranking.

    Fetcha testi dal DB e applica reranking.

    Args:
        query: Query
        results: Risultati hybrid
        db_pool: Connection pool
        reranker: Reranker (creato se None)
        top_k: Top K

    Returns:
        Risultati reranked
    """
    if not results:
        return []

    # Fetch testi massime
    massima_ids = [r.massima_id for r in results]

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, testo
            FROM kb.massime
            WHERE id = ANY($1)
            """,
            massima_ids,
        )

    massima_texts = {row["id"]: row["testo"] for row in rows}

    # Rerank
    if reranker is None:
        reranker = BGEReranker()

    return await reranker.rerank(query, results, massima_texts, top_k)


def calculate_rerank_lift(
    reranked: list[RerankedResult],
    ground_truth_ids: set[UUID],
    k: int = 10,
) -> float:
    """
    Calcola lift del reranking rispetto a ranking originale.

    Lift = (MRR_reranked - MRR_original) / MRR_original

    Args:
        reranked: Risultati reranked
        ground_truth_ids: ID rilevanti
        k: Top K

    Returns:
        Lift percentage
    """
    if not reranked or not ground_truth_ids:
        return 0.0

    # MRR con new_rank
    mrr_new = 0.0
    for r in reranked[:k]:
        if r.massima_id in ground_truth_ids:
            mrr_new = 1.0 / r.new_rank
            break

    # MRR con original_rank
    mrr_original = 0.0
    sorted_by_original = sorted(reranked, key=lambda x: x.original_rank)
    for r in sorted_by_original[:k]:
        if r.massima_id in ground_truth_ids:
            mrr_original = 1.0 / r.original_rank
            break

    if mrr_original == 0:
        return 0.0

    return (mrr_new - mrr_original) / mrr_original
