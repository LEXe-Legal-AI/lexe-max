"""
LEXE Knowledge Base - Reranker

Cross-encoder reranking per hybrid search results.

Modelli supportati:
- BAAI/bge-reranker-v2-m3 (default, multilingua, 1024 tokens)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (veloce, inglese)
- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingua, italiano)
- Alibaba-NLP/gte-multilingual-reranker-base (multilingua avanzato)
"""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog

from .hybrid import HybridSearchResult

logger = structlog.get_logger(__name__)


class RerankerModel(str, Enum):
    """Modelli reranker supportati."""

    BGE_M3 = "BAAI/bge-reranker-v2-m3"
    MS_MARCO_MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MMARCO_MULTILINGUAL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    GTE_MULTILINGUAL = "Alibaba-NLP/gte-multilingual-reranker-base"

    @classmethod
    def default(cls) -> "RerankerModel":
        """Modello default per italiano legale."""
        return cls.BGE_M3


@dataclass
class RerankerConfig:
    """Configurazione reranker."""

    model: RerankerModel = RerankerModel.BGE_M3
    device: str | None = None  # auto-detect: cuda/mps/cpu
    batch_size: int = 32
    max_length: int = 512  # Max tokens per input
    top_k: int = 10  # Top K risultati da ritornare
    rrf_weight: float = 0.4  # Peso RRF nel final score (0=solo rerank, 1=solo RRF)


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


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str | None = None):
    """
    Carica CrossEncoder con caching singleton.

    Il modello viene caricato una sola volta e riutilizzato.

    Args:
        model_name: Nome modello HuggingFace
        device: Device target (None = auto)

    Returns:
        CrossEncoder instance
    """
    from sentence_transformers import CrossEncoder

    logger.info("Loading CrossEncoder model", model=model_name, device=device or "auto")
    model = CrossEncoder(model_name, device=device)
    logger.info("CrossEncoder loaded successfully", model=model_name)
    return model


class CrossEncoderReranker:
    """
    Reranker generico basato su CrossEncoder.

    Supporta multipli modelli cross-encoder per reranking preciso.
    Ottimizzato per testo legale italiano.
    """

    def __init__(
        self,
        model: RerankerModel | str = RerankerModel.BGE_M3,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Inizializza reranker.

        Args:
            model: Modello cross-encoder (enum o stringa)
            device: Device (cpu/cuda/mps), None = auto
            batch_size: Batch size per inference
            max_length: Max token per coppia query-doc
        """
        self.model_name = model.value if isinstance(model, RerankerModel) else model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        """Lazy load del modello con caching."""
        if self._model is None:
            try:
                self._model = _load_cross_encoder(self.model_name, self.device)
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Calcola score per coppie query-documento.

        Args:
            pairs: Lista di (query, document) tuples

        Returns:
            Lista di score (higher = more relevant)
        """
        self._load_model()
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        massima_texts: dict[UUID, str],
        top_k: int = 10,
        rrf_weight: float = 0.4,
    ) -> list[RerankedResult]:
        """
        Rerank risultati hybrid search con cross-encoder.

        Args:
            query: Query originale
            results: Risultati da hybrid search
            massima_texts: Dict massima_id -> testo massima
            top_k: Top K risultati da ritornare
            rrf_weight: Peso per RRF score nel final score (0-1)

        Returns:
            Lista risultati reranked
        """
        import asyncio

        if not results:
            return []

        # Prepara coppie (query, text) per reranking
        pairs = []
        valid_results = []

        for result in results:
            text = massima_texts.get(result.massima_id)
            if text:
                # Tronca testo se troppo lungo (preserva inizio)
                truncated = text[: self.max_length * 4]  # ~4 chars per token
                pairs.append((query, truncated))
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
        scores = await loop.run_in_executor(None, lambda: self.predict(pairs))

        # Normalizza score e combina con RRF
        max_rrf = max(r.rrf_score for r in valid_results) if valid_results else 1.0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        score_range = max_score - min_score if max_score > min_score else 1.0

        reranked = []
        for original_result, rerank_score in zip(valid_results, scores):
            # Normalizza rerank score a 0-1
            norm_rerank = (rerank_score - min_score) / score_range

            # Normalizza RRF score a 0-1
            rrf_norm = original_result.rrf_score / max_rrf if max_rrf > 0 else 0

            # Final score: combinazione weighted
            final_score = (1 - rrf_weight) * norm_rerank + rrf_weight * rrf_norm

            reranked.append(
                RerankedResult(
                    massima_id=original_result.massima_id,
                    rerank_score=float(rerank_score),
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
            "Cross-encoder reranking completed",
            model=self.model_name,
            input_count=len(results),
            output_count=min(top_k, len(reranked)),
            top_rerank_score=reranked[0].rerank_score if reranked else 0,
        )

        return reranked[:top_k]


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


# =============================================================================
# KB Normativa Reranking (chunks)
# =============================================================================


@dataclass
class NormativaRerankedResult:
    """Risultato reranking per normativa chunks."""

    chunk_id: int
    work_code: str
    articolo: str
    chunk_no: int
    rerank_score: float
    final_score: float
    original_rank: int
    new_rank: int
    text_preview: str

    # Original scores
    rrf_score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


async def rerank_normativa_chunks(
    query: str,
    chunks: list[dict],
    reranker: CrossEncoderReranker | None = None,
    top_k: int = 10,
    rrf_weight: float = 0.3,
) -> list[NormativaRerankedResult]:
    """
    Rerank normativa chunks da hybrid search.

    Pipeline:
    1. Hybrid search ritorna top-20 chunks con RRF score
    2. Cross-encoder ricalcola score per ogni (query, chunk_text)
    3. Final score = (1-w)*rerank + w*rrf_norm

    Args:
        query: Query testuale
        chunks: Lista di dict con campi:
            - chunk_id: int
            - work_code: str (es. "CC", "CP")
            - articolo: str
            - chunk_no: int
            - text: str
            - rrf_score: float
            - dense_score: float (optional)
            - sparse_score: float (optional)
            - rank: int (original rank)
        reranker: CrossEncoderReranker (creato se None)
        top_k: Top K risultati
        rrf_weight: Peso RRF (0=solo rerank, 1=solo RRF)

    Returns:
        Lista NormativaRerankedResult ordinata per final_score
    """
    import asyncio

    if not chunks:
        return []

    # Crea reranker se non fornito
    if reranker is None:
        reranker = CrossEncoderReranker(
            model=RerankerModel.BGE_M3,
            batch_size=16,
            max_length=512,
        )

    # Prepara pairs per cross-encoder
    pairs = [(query, chunk.get("text", "")[:2000]) for chunk in chunks]

    # Run reranking
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, lambda: reranker.predict(pairs))

    # Normalizza e combina
    max_rrf = max(c.get("rrf_score", 0) for c in chunks) or 1.0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    score_range = max_score - min_score if max_score > min_score else 1.0

    reranked = []
    for chunk, rerank_score in zip(chunks, scores):
        rrf_score = chunk.get("rrf_score", 0)
        rrf_norm = rrf_score / max_rrf if max_rrf > 0 else 0
        norm_rerank = (rerank_score - min_score) / score_range

        final_score = (1 - rrf_weight) * norm_rerank + rrf_weight * rrf_norm

        text = chunk.get("text", "")
        preview = text[:150].replace("\n", " ") if text else ""

        reranked.append(
            NormativaRerankedResult(
                chunk_id=chunk.get("chunk_id", 0),
                work_code=chunk.get("work_code", ""),
                articolo=chunk.get("articolo", ""),
                chunk_no=chunk.get("chunk_no", 0),
                rerank_score=float(rerank_score),
                final_score=final_score,
                original_rank=chunk.get("rank", 0),
                new_rank=0,
                text_preview=preview,
                rrf_score=rrf_score,
                dense_score=chunk.get("dense_score"),
                sparse_score=chunk.get("sparse_score"),
            )
        )

    # Sort by final score
    reranked.sort(key=lambda x: x.final_score, reverse=True)

    # Assign new_rank
    for i, r in enumerate(reranked):
        r.new_rank = i + 1

    logger.info(
        "Normativa chunks reranked",
        input_count=len(chunks),
        output_count=min(top_k, len(reranked)),
        top_rerank=reranked[0].rerank_score if reranked else 0,
        rank_changes=sum(1 for r in reranked[:top_k] if r.new_rank != r.original_rank),
    )

    return reranked[:top_k]


def get_reranker(
    model: RerankerModel | str = RerankerModel.BGE_M3,
    device: str | None = None,
) -> CrossEncoderReranker:
    """
    Factory per ottenere reranker con modello specificato.

    Args:
        model: Modello da usare
        device: Device (None = auto)

    Returns:
        CrossEncoderReranker configurato
    """
    return CrossEncoderReranker(model=model, device=device)
