"""LEXE Knowledge Base - Library Hybrid Search Service.

Hybrid search (Dense + Sparse + RRF) for private library chunks.
Follows the same pattern as normativa_hybrid.py but operates on
``kb.library_chunk`` / ``kb.library_chunk_fts`` / ``kb.library_chunk_embeddings``
tables, always filtered by ``tenant_id`` for data isolation.

Database: lexe-max (schema kb)
Embeddings: Qwen3 1536d via LiteLLM
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

import asyncpg
import structlog

from ..config import EMBEDDING_DIMS, EmbeddingModel, KBSettings
from ..ingestion.embedder import EmbeddingClient

logger = structlog.get_logger(__name__)

# Library documents use Qwen3 1536d (same as ingestion).
_LIBRARY_MODEL = EmbeddingModel.QWEN3
_LIBRARY_DIMS = EMBEDDING_DIMS[_LIBRARY_MODEL]

# RRF / search defaults
_RRF_K = 60
_DENSE_LIMIT = 50
_SPARSE_LIMIT = 50


# ── Result dataclass ──────────────────────────────────────────────


@dataclass
class LibrarySearchResult:
    """Single result from library hybrid search."""

    document_id: UUID
    chunk_id: UUID
    chunk_no: int
    text: str
    rrf_score: float
    dense_score: float
    sparse_score: float
    filename: str | None = None


# ── Embedding helper ──────────────────────────────────────────────


async def _get_query_embedding(
    query: str,
    embedding_client: EmbeddingClient | None = None,
) -> tuple[list[float], float, EmbeddingClient | None]:
    """Generate query embedding via LiteLLM.

    Returns:
        (embedding, elapsed_ms, client_to_close_or_None)
    """
    own_client = False
    if embedding_client is None:
        settings = KBSettings()
        embedding_client = EmbeddingClient(litellm_url=settings.kb_litellm_url)
        own_client = True

    t0 = time.perf_counter()
    embedding = await embedding_client.embed_single(query, _LIBRARY_MODEL)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.debug(
        "Library query embedding generated",
        dims=len(embedding),
        time_ms=round(elapsed_ms, 1),
    )

    return embedding, elapsed_ms, embedding_client if own_client else None


# ── Hybrid search ─────────────────────────────────────────────────


async def library_hybrid_search(
    pool: asyncpg.Pool,
    query: str,
    tenant_id: UUID,
    *,
    top_k: int = 5,
    doc_type: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> tuple[list[LibrarySearchResult], float, float | None]:
    """Hybrid search on private library chunks.

    Implements Dense + Sparse + RRF fusion:
        1. Dense: vector cosine similarity on ``kb.library_chunk_embeddings``
        2. Sparse: BM25 via ``ts_rank_cd`` on ``kb.library_chunk_fts``
        3. RRF: Reciprocal Rank Fusion to merge rankings

    All queries are strictly scoped to *tenant_id* for data isolation.

    Args:
        pool:             asyncpg connection pool.
        query:            Search query text.
        tenant_id:        UUID of the owning tenant (mandatory filter).
        top_k:            Number of results to return.
        doc_type:         Optional document type filter (reserved for future use).
        embedding_client: Optional pre-initialised client; created from env if None.

    Returns:
        (results, query_time_ms, embedding_time_ms)
    """
    start = time.perf_counter()

    # 1. Query embedding
    client_to_close: EmbeddingClient | None = None
    try:
        query_embedding, embedding_time_ms, client_to_close = await _get_query_embedding(
            query, embedding_client
        )
    except Exception:
        logger.exception("Library search embedding failed — falling back to sparse-only")
        # Graceful degradation: sparse-only search
        results = await _sparse_only_search(pool, query, tenant_id, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return results, elapsed_ms, None

    try:
        results = await _hybrid_rrf_search(
            pool, query, query_embedding, tenant_id, top_k
        )
    finally:
        if client_to_close is not None:
            await client_to_close.close()

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Library hybrid search completed",
        tenant_id=str(tenant_id),
        results=len(results),
        top_k=top_k,
        time_ms=round(elapsed_ms, 1),
    )

    return results, elapsed_ms, embedding_time_ms


# ── Internal search implementations ──────────────────────────────


async def _hybrid_rrf_search(
    pool: asyncpg.Pool,
    query: str,
    query_embedding: list[float],
    tenant_id: UUID,
    top_k: int,
) -> list[LibrarySearchResult]:
    """Execute hybrid search with RRF fusion."""

    sql = f"""
WITH query_params AS (
    SELECT
        $1::vector({_LIBRARY_DIMS}) AS qemb,
        plainto_tsquery('italian', $2) AS qtsv
),
-- Dense search: top-N by cosine similarity
dense AS (
    SELECT
        c.id AS chunk_id,
        ROW_NUMBER() OVER (ORDER BY e.embedding <=> q.qemb) AS rank_dense,
        1 - (e.embedding <=> q.qemb) AS score_dense
    FROM kb.library_chunk c
    JOIN kb.library_chunk_embeddings e ON e.chunk_id = c.id
    CROSS JOIN query_params q
    WHERE c.tenant_id = $3
    ORDER BY e.embedding <=> q.qemb
    LIMIT $4
),
-- Sparse search: top-N by BM25 ts_rank_cd
sparse AS (
    SELECT
        f.chunk_id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC) AS rank_sparse,
        ts_rank_cd(f.tsv_it, q.qtsv) AS score_sparse
    FROM kb.library_chunk_fts f
    JOIN kb.library_chunk c ON c.id = f.chunk_id
    CROSS JOIN query_params q
    WHERE c.tenant_id = $3
      AND f.tsv_it @@ q.qtsv
    ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC
    LIMIT $5
),
-- RRF Fusion (k=60)
rrf AS (
    SELECT
        COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
        COALESCE(1.0 / ($6 + d.rank_dense), 0)
            + COALESCE(1.0 / ($6 + s.rank_sparse), 0) AS rrf_score,
        d.score_dense,
        s.score_sparse
    FROM dense d
    FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
)
SELECT
    c.document_id,
    c.id AS chunk_id,
    c.chunk_no,
    c.text,
    ROUND(r.rrf_score::numeric, 6) AS rrf_score,
    ROUND(COALESCE(r.score_dense, 0)::numeric, 6) AS dense_score,
    ROUND(COALESCE(r.score_sparse, 0)::numeric, 6) AS sparse_score
FROM rrf r
JOIN kb.library_chunk c ON c.id = r.chunk_id
ORDER BY r.rrf_score DESC
LIMIT $7;
    """

    params = [
        query_embedding,           # $1 - vector
        query,                     # $2 - tsquery text
        tenant_id,                 # $3 - tenant_id
        _DENSE_LIMIT,              # $4 - dense limit
        _SPARSE_LIMIT,             # $5 - sparse limit
        _RRF_K,                    # $6 - RRF k
        top_k,                     # $7 - final limit
    ]

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return _rows_to_results(rows)


async def _sparse_only_search(
    pool: asyncpg.Pool,
    query: str,
    tenant_id: UUID,
    top_k: int,
) -> list[LibrarySearchResult]:
    """Fallback sparse-only (BM25) search when embeddings are unavailable."""

    sql = """
WITH query_tsv AS (
    SELECT plainto_tsquery('italian', $1) AS qtsv
)
SELECT
    c.document_id,
    c.id AS chunk_id,
    c.chunk_no,
    c.text,
    ROUND(ts_rank_cd(f.tsv_it, q.qtsv)::numeric, 6) AS rrf_score,
    0::numeric AS dense_score,
    ROUND(ts_rank_cd(f.tsv_it, q.qtsv)::numeric, 6) AS sparse_score
FROM kb.library_chunk_fts f
JOIN kb.library_chunk c ON c.id = f.chunk_id
CROSS JOIN query_tsv q
WHERE c.tenant_id = $2
  AND f.tsv_it @@ q.qtsv
ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC
LIMIT $3;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, query, tenant_id, top_k)

    return _rows_to_results(rows)


# ── Row mapping ───────────────────────────────────────────────────


def _rows_to_results(rows: list[asyncpg.Record]) -> list[LibrarySearchResult]:
    """Convert database rows to LibrarySearchResult objects."""
    results: list[LibrarySearchResult] = []
    for row in rows:
        results.append(
            LibrarySearchResult(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                chunk_no=row["chunk_no"],
                text=row["text"] or "",
                rrf_score=float(row["rrf_score"]) if row["rrf_score"] else 0.0,
                dense_score=float(row["dense_score"]) if row["dense_score"] else 0.0,
                sparse_score=float(row["sparse_score"]) if row["sparse_score"] else 0.0,
            )
        )
    return results
