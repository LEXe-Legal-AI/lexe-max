"""
LEXE Knowledge Base - Normativa Retrieval API Router

FastAPI endpoints for KB Normativa hybrid search.

Endpoints:
- POST /api/v1/kb/normativa/search - Hybrid search (dense + sparse + RRF)
- GET /api/v1/kb/normativa/stats - Get corpus statistics
"""

import time

import structlog
from fastapi import APIRouter, HTTPException, status

from lexe_api.kb.retrieval.normativa_hybrid import (
    get_embedding,
    get_normativa_pool,
    get_normativa_stats,
    hybrid_search_normativa,
)
from lexe_api.kb.retrieval.schemas import (
    ErrorResponse,
    NormativaSearchRequest,
    NormativaSearchResponse,
    NormativaStatsResponse,
    SearchMode,
)

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/kb/normativa",
    tags=["kb-normativa"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


@router.post(
    "/search",
    response_model=NormativaSearchResponse,
    summary="Hybrid Search KB Normativa",
    description="""
Search the KB Normativa corpus using hybrid search (Dense + Sparse + RRF fusion).

**Search Modes:**
- `hybrid` (default): Combines dense (vector) and sparse (BM25) search with RRF fusion
- `dense`: Vector similarity search only
- `sparse`: BM25 full-text search only

**Filters:**
- `codes`: Filter by work codes (e.g., `["CC", "CPC"]` for Codice Civile and Codice di Procedura Civile)

**Example query:**
```json
{
    "query": "risarcimento danno responsabilita civile",
    "top_k": 10,
    "codes": ["CC"],
    "mode": "hybrid"
}
```

**Requirements:**
- `OPENROUTER_API_KEY` environment variable must be set for embedding generation
    """,
    responses={
        200: {
            "description": "Search results",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "query": "risarcimento danno",
                        "mode": "hybrid",
                        "total_results": 10,
                        "results": [
                            {
                                "code": "CC",
                                "article": "2043",
                                "chunk_no": 0,
                                "rrf_score": 0.0312,
                                "dense_score": 0.782,
                                "sparse_score": 0.156,
                                "text": "Qualunque fatto doloso o colposo...",
                                "text_preview": "Qualunque fatto doloso o colposo...",
                            }
                        ],
                        "query_time_ms": 145.2,
                        "embedding_time_ms": 89.3,
                        "total_chunks": 10246,
                    }
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def search_normativa(request: NormativaSearchRequest) -> NormativaSearchResponse:
    """
    Hybrid search on KB Normativa.

    Combines dense (vector) and sparse (BM25) search using RRF fusion
    to find relevant legal text chunks.
    """
    start_time = time.perf_counter()

    try:
        # Get database pool
        pool = await get_normativa_pool()

        # Generate query embedding (only for dense/hybrid modes)
        embedding_time_ms = None
        query_embedding = []

        if request.mode in (SearchMode.HYBRID, SearchMode.DENSE):
            try:
                query_embedding, embedding_time_ms = await get_embedding(request.query)
            except ValueError as e:
                # Missing API key
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(e),
                )
            except Exception as e:
                logger.error("Embedding generation failed", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Embedding service error: {str(e)}",
                )

        # Execute search
        results, query_time_ms = await hybrid_search_normativa(
            query=request.query,
            query_embedding=query_embedding,
            pool=pool,
            top_k=request.top_k,
            codes=request.codes,
            mode=request.mode,
        )

        # Filter by min_score if specified
        if request.min_score > 0:
            results = [r for r in results if r.rrf_score >= request.min_score]

        # Get total chunks for context (optional, cached)
        total_chunks = None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT COUNT(*) FROM kb.normativa_chunk")
                total_chunks = row[0]
        except Exception:
            pass  # Non-critical

        total_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Search completed",
            query=request.query[:50],
            mode=request.mode.value,
            results=len(results),
            query_time_ms=round(query_time_ms, 1),
            total_time_ms=round(total_time_ms, 1),
        )

        return NormativaSearchResponse(
            success=True,
            query=request.query,
            mode=request.mode,
            total_results=len(results),
            results=results,
            query_time_ms=query_time_ms,
            embedding_time_ms=embedding_time_ms,
            total_chunks=total_chunks,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/stats",
    response_model=NormativaStatsResponse,
    summary="Get KB Normativa Statistics",
    description="""
Get statistics about the KB Normativa corpus.

Returns:
- Total works, articles, chunks, and embeddings
- Embedding coverage percentage
- Per-work breakdown
- Search readiness status
    """,
    responses={
        200: {
            "description": "Corpus statistics",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "total_works": 69,
                        "total_articles": 6335,
                        "total_chunks": 10246,
                        "total_embeddings": 3300,
                        "embedding_coverage_pct": 32.2,
                        "works": [
                            {
                                "code": "CC",
                                "title": "Codice Civile",
                                "articles": 2969,
                                "chunks": 5000,
                                "embeddings": 1500,
                            }
                        ],
                        "search_ready": True,
                        "fts_ready": True,
                    }
                }
            },
        },
    },
)
async def get_stats() -> NormativaStatsResponse:
    """
    Get KB Normativa corpus statistics.

    Returns information about the corpus size, embedding coverage,
    and search readiness.
    """
    try:
        pool = await get_normativa_pool()
        stats = await get_normativa_stats(pool)

        return NormativaStatsResponse(**stats)

    except Exception as e:
        logger.exception("Stats fetch failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@router.get(
    "/health",
    summary="KB Normativa Health Check",
    description="Check if KB Normativa search is operational.",
    responses={
        200: {"description": "Service healthy"},
        503: {"description": "Service unavailable"},
    },
)
async def health_check() -> dict:
    """
    Health check for KB Normativa search.

    Verifies database connectivity and basic query capability.
    """
    try:
        pool = await get_normativa_pool()
        async with pool.acquire() as conn:
            # Simple query to verify connectivity
            row = await conn.fetchrow("SELECT 1 as ok, NOW() as ts")
            return {
                "status": "healthy",
                "service": "kb-normativa",
                "database": "connected",
                "timestamp": str(row["ts"]),
            }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )
