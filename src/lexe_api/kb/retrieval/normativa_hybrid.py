"""
LEXE Knowledge Base - Normativa Hybrid Search Service

Hybrid search (Dense + Sparse + RRF) for KB Normativa chunks.
Based on hybrid_search_staging.py script.

Database: lexe-max (port 5436 staging, schema kb)
Embeddings: OpenRouter API (text-embedding-3-small, 1536 dim)
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import asyncpg
import httpx
import structlog

from .schemas import NormativaSearchResult, SearchMode

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NormativaDBConfig:
    """Database configuration for KB Normativa."""

    host: str = "localhost"
    port: int = 5436
    user: str = "lexe_max"
    password: str = "lexe_max_dev_password"
    dbname: str = "lexe_max"

    @classmethod
    def from_env(cls) -> "NormativaDBConfig":
        """Load config from environment variables."""
        return cls(
            host=os.environ.get("LEXE_KB_HOST", "localhost"),
            port=int(os.environ.get("LEXE_KB_PORT", "5436")),
            user=os.environ.get("LEXE_KB_USER", "lexe_max"),
            password=os.environ.get("LEXE_KB_PASSWORD", "lexe_max_dev_password"),
            dbname=os.environ.get("LEXE_KB_DBNAME", "lexe_max"),
        )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    dense_limit: int = 50  # Top-K for dense search
    sparse_limit: int = 50  # Top-K for sparse search
    rrf_k: int = 60  # RRF parameter (standard value)
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    openrouter_url: str = "https://openrouter.ai/api/v1/embeddings"


# =============================================================================
# Database Pool Management
# =============================================================================

_normativa_pool: asyncpg.Pool | None = None


async def get_normativa_pool(config: NormativaDBConfig | None = None) -> asyncpg.Pool:
    """Get or create the KB Normativa database pool."""
    global _normativa_pool
    if _normativa_pool is None:
        cfg = config or NormativaDBConfig.from_env()
        _normativa_pool = await asyncpg.create_pool(
            cfg.dsn,
            min_size=2,
            max_size=5,
            command_timeout=30,
        )
        logger.info("Normativa DB pool created", dsn=cfg.dsn[:40] + "...")
    return _normativa_pool


async def close_normativa_pool() -> None:
    """Close the KB Normativa database pool."""
    global _normativa_pool
    if _normativa_pool:
        await _normativa_pool.close()
        _normativa_pool = None
        logger.info("Normativa DB pool closed")


# =============================================================================
# Embedding Service
# =============================================================================


async def get_embedding(
    text: str,
    config: HybridSearchConfig | None = None,
) -> tuple[list[float], float]:
    """
    Get embedding vector for text via OpenRouter API.

    Args:
        text: Text to embed
        config: Search configuration

    Returns:
        Tuple of (embedding vector, time_ms)

    Raises:
        ValueError: If OPENROUTER_API_KEY not set
        httpx.HTTPStatusError: If API call fails
    """
    cfg = config or HybridSearchConfig()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. Required for embedding generation."
        )

    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            cfg.openrouter_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": cfg.embedding_model,
                "input": [text],
            },
        )
        response.raise_for_status()

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = response.json()
    embedding = data["data"][0]["embedding"]

    logger.debug(
        "Embedding generated",
        model=cfg.embedding_model,
        dim=len(embedding),
        time_ms=round(elapsed_ms, 1),
    )

    return embedding, elapsed_ms


# =============================================================================
# Hybrid Search
# =============================================================================


async def hybrid_search_normativa(
    query: str,
    query_embedding: list[float],
    pool: asyncpg.Pool,
    *,
    top_k: int = 10,
    codes: list[str] | None = None,
    mode: SearchMode = SearchMode.HYBRID,
    config: HybridSearchConfig | None = None,
) -> tuple[list[NormativaSearchResult], float]:
    """
    Hybrid search on KB Normativa chunks.

    Implements Dense + Sparse + RRF fusion:
    1. Dense: Vector similarity (embedding <=> query_embedding)
    2. Sparse: BM25 via plainto_tsquery('italian', query)
    3. RRF: Reciprocal Rank Fusion to combine results

    Args:
        query: Search query text
        query_embedding: Pre-computed embedding vector
        pool: Database connection pool
        top_k: Number of results to return
        codes: Optional filter by work codes (e.g., ['CC', 'CPC'])
        mode: Search mode (hybrid, dense, sparse)
        config: Search configuration

    Returns:
        Tuple of (results, query_time_ms)
    """
    cfg = config or HybridSearchConfig()
    start = time.perf_counter()

    # Build code filter if provided
    code_filter = ""
    code_values: list[Any] = []
    if codes:
        placeholders = ", ".join(f"${i + 7}" for i in range(len(codes)))
        code_filter = f"AND w.code IN ({placeholders})"
        code_values = codes

    async with pool.acquire() as conn:
        if mode == SearchMode.DENSE:
            results = await _dense_only_search(
                conn, query_embedding, top_k, code_filter, code_values, cfg
            )
        elif mode == SearchMode.SPARSE:
            results = await _sparse_only_search(conn, query, top_k, code_filter, code_values)
        else:
            # Default: Hybrid (Dense + Sparse + RRF)
            results = await _hybrid_rrf_search(
                conn, query, query_embedding, top_k, code_filter, code_values, cfg
            )

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Normativa search completed",
        mode=mode.value,
        results=len(results),
        top_k=top_k,
        codes=codes,
        time_ms=round(elapsed_ms, 1),
    )

    return results, elapsed_ms


async def _hybrid_rrf_search(
    conn: asyncpg.Connection,
    query: str,
    query_embedding: list[float],
    top_k: int,
    code_filter: str,
    code_values: list[Any],
    cfg: HybridSearchConfig,
) -> list[NormativaSearchResult]:
    """Execute hybrid search with RRF fusion."""

    # Build the SQL query with RRF fusion
    sql = f"""
WITH query_params AS (
    SELECT
        $1::vector(1536) as qemb,
        plainto_tsquery('italian', $2) as qtsv
),
-- Dense search (top N)
dense AS (
    SELECT c.id as chunk_id,
           ROW_NUMBER() OVER (ORDER BY e.embedding <=> q.qemb) as rank_dense,
           1 - (e.embedding <=> q.qemb) as score_dense
    FROM kb.normativa_chunk c
    JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
    JOIN kb.work w ON w.id = c.work_id
    CROSS JOIN query_params q
    WHERE 1=1 {code_filter}
    ORDER BY e.embedding <=> q.qemb
    LIMIT $3
),
-- Sparse search (top N)
sparse AS (
    SELECT f.chunk_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC) as rank_sparse,
           ts_rank_cd(f.tsv_it, q.qtsv) as score_sparse
    FROM kb.normativa_chunk_fts f
    JOIN kb.normativa_chunk c ON c.id = f.chunk_id
    JOIN kb.work w ON w.id = c.work_id
    CROSS JOIN query_params q
    WHERE f.tsv_it @@ q.qtsv {code_filter}
    ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC
    LIMIT $4
),
-- RRF Fusion
rrf AS (
    SELECT
        COALESCE(d.chunk_id, s.chunk_id) as chunk_id,
        COALESCE(1.0 / ($5 + d.rank_dense), 0) + COALESCE(1.0 / ($5 + s.rank_sparse), 0) as rrf_score,
        d.score_dense,
        s.score_sparse
    FROM dense d
    FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
)
SELECT
    w.code,
    n.articolo as article,
    c.chunk_no,
    ROUND(r.rrf_score::numeric, 4) as rrf_score,
    ROUND(r.score_dense::numeric, 4) as dense_score,
    ROUND(COALESCE(r.score_sparse, 0)::numeric, 4) as sparse_score,
    c.text,
    w.id as work_id,
    n.id as normativa_id,
    c.id as chunk_id
FROM rrf r
JOIN kb.normativa_chunk c ON c.id = r.chunk_id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
ORDER BY r.rrf_score DESC
LIMIT $6;
    """

    # Build parameters
    params = [
        query_embedding,
        query,
        cfg.dense_limit,
        cfg.sparse_limit,
        cfg.rrf_k,
        top_k,
        *code_values,
    ]

    rows = await conn.fetch(sql, *params)
    return _rows_to_results(rows)


async def _dense_only_search(
    conn: asyncpg.Connection,
    query_embedding: list[float],
    top_k: int,
    code_filter: str,
    code_values: list[Any],
    cfg: HybridSearchConfig,
) -> list[NormativaSearchResult]:
    """Execute dense-only search."""

    sql = f"""
SELECT
    w.code,
    n.articolo as article,
    c.chunk_no,
    ROUND((1 - (e.embedding <=> $1::vector(1536)))::numeric, 4) as rrf_score,
    ROUND((1 - (e.embedding <=> $1::vector(1536)))::numeric, 4) as dense_score,
    NULL::numeric as sparse_score,
    c.text,
    w.id as work_id,
    n.id as normativa_id,
    c.id as chunk_id
FROM kb.normativa_chunk c
JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
WHERE 1=1 {code_filter}
ORDER BY e.embedding <=> $1::vector(1536)
LIMIT $2;
    """

    params = [query_embedding, top_k, *code_values]
    rows = await conn.fetch(sql, *params)
    return _rows_to_results(rows)


async def _sparse_only_search(
    conn: asyncpg.Connection,
    query: str,
    top_k: int,
    code_filter: str,
    code_values: list[Any],
) -> list[NormativaSearchResult]:
    """Execute sparse-only (BM25) search."""

    sql = f"""
WITH query_tsv AS (
    SELECT plainto_tsquery('italian', $1) as qtsv
)
SELECT
    w.code,
    n.articolo as article,
    c.chunk_no,
    ROUND(ts_rank_cd(f.tsv_it, q.qtsv)::numeric, 4) as rrf_score,
    NULL::numeric as dense_score,
    ROUND(ts_rank_cd(f.tsv_it, q.qtsv)::numeric, 4) as sparse_score,
    c.text,
    w.id as work_id,
    n.id as normativa_id,
    c.id as chunk_id
FROM kb.normativa_chunk_fts f
JOIN kb.normativa_chunk c ON c.id = f.chunk_id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
CROSS JOIN query_tsv q
WHERE f.tsv_it @@ q.qtsv {code_filter}
ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC
LIMIT $2;
    """

    params = [query, top_k, *code_values]
    rows = await conn.fetch(sql, *params)
    return _rows_to_results(rows)


def _rows_to_results(rows: list[asyncpg.Record]) -> list[NormativaSearchResult]:
    """Convert database rows to NormativaSearchResult objects."""
    results = []
    for row in rows:
        text = row["text"] or ""
        results.append(
            NormativaSearchResult(
                code=row["code"],
                article=row["article"],
                chunk_no=row["chunk_no"],
                rrf_score=float(row["rrf_score"]) if row["rrf_score"] else 0.0,
                dense_score=float(row["dense_score"]) if row["dense_score"] else None,
                sparse_score=float(row["sparse_score"]) if row["sparse_score"] else None,
                text=text,
                text_preview=text[:200] if len(text) > 200 else None,
                work_id=row["work_id"],
                normativa_id=row["normativa_id"],
                chunk_id=row["chunk_id"],
            )
        )
    return results


# =============================================================================
# Corpus Statistics
# =============================================================================


async def get_normativa_stats(pool: asyncpg.Pool) -> dict[str, Any]:
    """
    Get KB Normativa corpus statistics.

    Returns stats about works, articles, chunks, embeddings, and search readiness.
    """
    async with pool.acquire() as conn:
        # Total counts
        stats_row = await conn.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM kb.work) as total_works,
                (SELECT COUNT(*) FROM kb.normativa) as total_articles,
                (SELECT COUNT(*) FROM kb.normativa_chunk) as total_chunks,
                (SELECT COUNT(*) FROM kb.normativa_chunk_embeddings) as total_embeddings,
                (SELECT COUNT(*) FROM kb.normativa_chunk_fts) as total_fts
        """)

        total_chunks = stats_row["total_chunks"]
        total_embeddings = stats_row["total_embeddings"]
        total_fts = stats_row["total_fts"]

        # Per-work stats
        work_rows = await conn.fetch("""
            SELECT
                w.code,
                w.title,
                COUNT(DISTINCT n.id) as articles,
                COUNT(DISTINCT c.id) as chunks,
                COUNT(DISTINCT e.chunk_id) as embeddings
            FROM kb.work w
            LEFT JOIN kb.normativa n ON n.work_id = w.id
            LEFT JOIN kb.normativa_chunk c ON c.work_id = w.id
            LEFT JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
            GROUP BY w.id, w.code, w.title
            ORDER BY chunks DESC
        """)

        works = [
            {
                "code": row["code"],
                "title": row["title"],
                "articles": row["articles"],
                "chunks": row["chunks"],
                "embeddings": row["embeddings"],
            }
            for row in work_rows
        ]

        # Calculate coverage
        coverage_pct = (total_embeddings / total_chunks * 100) if total_chunks > 0 else 0.0

        return {
            "success": True,
            "total_works": stats_row["total_works"],
            "total_articles": stats_row["total_articles"],
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "embedding_coverage_pct": round(coverage_pct, 1),
            "works": works,
            "search_ready": total_embeddings > 0,
            "fts_ready": total_fts > 0,
        }
