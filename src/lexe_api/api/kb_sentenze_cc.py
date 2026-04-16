"""
LEXE Knowledge Base — Sentenze Corte Costituzionale API Router

Endpoints:
- POST /api/v1/kb/sentenze_cc/search — Sparse (Phase 1) / Hybrid (Phase 2) search
- GET  /api/v1/kb/sentenze_cc/stats  — Corpus statistics
"""
import time
from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from lexe_api.kb.retrieval.sentenze_cc_search import (
    SentenzeSearchMode,
    get_sentenze_cc_pool,
    get_sentenze_cc_stats,
    search_sentenze_cc,
)

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/kb/sentenze_cc",
    tags=["kb-sentenze-cc"],
)


class SentenzeCCSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    year_min: Optional[int] = Field(default=None, ge=1948, le=2030)
    year_max: Optional[int] = Field(default=None, ge=1948, le=2030)
    ruling_type: str = Field(default="any", pattern=r"^(sentenza|ordinanza|any)$")


class SentenzaCCItem(BaseModel):
    id: str
    tipo: str
    numero: int
    anno: int
    presidente: Optional[str] = None
    relatore: Optional[str] = None
    data_deposito: Optional[str] = None
    dispositivo: Optional[str] = None
    text_preview: str
    sparse_score: float
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    source_url: Optional[str] = None


class SentenzeCCSearchResponse(BaseModel):
    success: bool
    query: str
    total_results: int
    results: list[SentenzaCCItem]
    query_time_ms: float
    total_sentenze: Optional[int] = None


@router.post("/search", response_model=SentenzeCCSearchResponse)
async def search(request: SentenzeCCSearchRequest) -> SentenzeCCSearchResponse:
    """Search Corte Costituzionale decisions (sentenze + ordinanze)."""
    start = time.perf_counter()
    try:
        pool = await get_sentenze_cc_pool()
        results, query_time_ms = await search_sentenze_cc(
            query=request.query,
            pool=pool,
            top_k=request.top_k,
            year_min=request.year_min,
            year_max=request.year_max,
            ruling_type=request.ruling_type if request.ruling_type != "any" else None,
            mode=SentenzeSearchMode.SPARSE,
        )
        total = None
        try:
            async with pool.acquire() as conn:
                total = await conn.fetchval("SELECT COUNT(*) FROM kb.sentenze_cc")
        except Exception:
            pass

        return SentenzeCCSearchResponse(
            success=True,
            query=request.query,
            total_results=len(results),
            results=[SentenzaCCItem(**r.to_dict()) for r in results],
            query_time_ms=round(query_time_ms, 1),
            total_sentenze=total,
        )
    except Exception as e:
        logger.exception("sentenze_cc search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def stats() -> dict[str, Any]:
    """Get sentenze CC corpus statistics."""
    pool = await get_sentenze_cc_pool()
    return await get_sentenze_cc_stats(pool)
