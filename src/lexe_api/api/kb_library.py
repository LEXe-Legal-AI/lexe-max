"""LEXE Knowledge Base - Library Ingestion, Search & Management API.

Internal endpoints for chunking, embedding, searching, and managing private
library documents.  No auth middleware here — HMAC validation happens at the
gateway (lexe-core) level.

Endpoints:
    POST   /api/v1/kb/library/ingest              — Chunk + embed a document
    POST   /api/v1/kb/library/search              — Hybrid search (dense+sparse+RRF)
    DELETE /api/v1/kb/library/document/{id}        — Delete chunks for a doc
    GET    /api/v1/kb/library/stats                — Per-tenant stats
"""

from __future__ import annotations

import time
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from lexe_api.database import get_kb_pool
from lexe_api.kb.ingestion.library_ingest import (
    delete_library_document_chunks,
    get_library_stats,
    ingest_library_document,
)
from lexe_api.kb.retrieval.library_hybrid import (
    LibrarySearchResult,
    library_hybrid_search,
)

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/kb/library",
    tags=["kb-library"],
    responses={
        500: {"description": "Internal server error"},
    },
)


# ── Request / Response schemas ────────────────────────────────────


class IngestRequest(BaseModel):
    """Body for the ingest endpoint."""

    document_id: UUID
    tenant_id: UUID
    text: str = Field(..., min_length=1, description="Plain-text content to chunk and embed")


class IngestResponse(BaseModel):
    """Returned after successful ingestion."""

    chunks_count: int
    embeddings_count: int
    processing_time_ms: float


class DeleteResponse(BaseModel):
    """Returned after deleting chunks for a document."""

    deleted_chunks: int


class StatsResponse(BaseModel):
    """Per-tenant library statistics."""

    tenant_id: str
    documents: int
    chunks: int
    embeddings: int


class SearchRequest(BaseModel):
    """Body for the library search endpoint."""

    query: str = Field(..., min_length=2, max_length=1000, description="Search query in natural language")
    tenant_id: UUID = Field(..., description="Tenant UUID (mandatory for data isolation)")
    top_k: int = Field(default=5, ge=1, le=20, description="Max results to return")
    doc_type: str | None = Field(default=None, description="Optional document type filter")


class SearchResultItem(BaseModel):
    """Single search result from library hybrid search."""

    document_id: str
    chunk_id: str
    chunk_no: int
    text: str
    rrf_score: float
    dense_score: float
    sparse_score: float
    filename: str | None = None


class SearchResponse(BaseModel):
    """Response from library hybrid search."""

    success: bool = True
    results: list[SearchResultItem] = Field(default_factory=list)
    query: str
    query_time_ms: float
    embedding_time_ms: float | None = None
    total_results: int


# ── Endpoints ─────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Chunk and embed a library document",
    description=(
        "Accepts plain text for a private library document, splits it into "
        "overlapping chunks (sliding window, 150-char overlap), generates "
        "Qwen3 1536d embeddings, and stores everything in kb.library_chunk "
        "and kb.library_chunk_embeddings.  FTS is populated automatically "
        "by a database trigger."
    ),
    status_code=status.HTTP_200_OK,
)
async def ingest(body: IngestRequest) -> IngestResponse:
    """Ingest a library document: chunk + embed + store."""
    logger.info(
        "Library ingest request",
        document_id=str(body.document_id),
        tenant_id=str(body.tenant_id),
        text_len=len(body.text),
    )

    try:
        pool = await get_kb_pool()
        result = await ingest_library_document(
            pool=pool,
            document_id=body.document_id,
            tenant_id=body.tenant_id,
            text=body.text,
        )
        return IngestResponse(
            chunks_count=result.chunks_count,
            embeddings_count=result.embeddings_count,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as exc:
        logger.exception(
            "Library ingest failed",
            document_id=str(body.document_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid search over private library",
    description=(
        "Search a tenant's private library documents using hybrid search "
        "(dense vector similarity + BM25 full-text + RRF fusion).  "
        "All results are strictly scoped to the given tenant_id."
    ),
    status_code=status.HTTP_200_OK,
)
async def search(body: SearchRequest) -> SearchResponse:
    """Hybrid search on private library chunks."""
    logger.info(
        "Library search request",
        tenant_id=str(body.tenant_id),
        query=body.query[:80],
        top_k=body.top_k,
    )

    try:
        pool = await get_kb_pool()
        results, query_time_ms, embedding_time_ms = await library_hybrid_search(
            pool=pool,
            query=body.query,
            tenant_id=body.tenant_id,
            top_k=body.top_k,
            doc_type=body.doc_type,
        )

        items = [
            SearchResultItem(
                document_id=str(r.document_id),
                chunk_id=str(r.chunk_id),
                chunk_no=r.chunk_no,
                text=r.text,
                rrf_score=r.rrf_score,
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                filename=r.filename,
            )
            for r in results
        ]

        return SearchResponse(
            success=True,
            results=items,
            query=body.query,
            query_time_ms=round(query_time_ms, 1),
            embedding_time_ms=round(embedding_time_ms, 1) if embedding_time_ms else None,
            total_results=len(items),
        )

    except Exception as exc:
        logger.exception(
            "Library search failed",
            tenant_id=str(body.tenant_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        )


@router.delete(
    "/document/{document_id}",
    response_model=DeleteResponse,
    summary="Delete all chunks for a library document",
    description=(
        "Removes all chunks, FTS entries, and embeddings for the given "
        "document_id.  CASCADE in the schema handles FTS and embeddings."
    ),
)
async def delete_document(document_id: UUID) -> DeleteResponse:
    """Delete chunks + embeddings for a library document."""
    logger.info("Library delete request", document_id=str(document_id))

    try:
        pool = await get_kb_pool()
        deleted = await delete_library_document_chunks(pool, document_id)
        return DeleteResponse(deleted_chunks=deleted)

    except Exception as exc:
        logger.exception(
            "Library delete failed",
            document_id=str(document_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {exc}",
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get per-tenant library statistics",
    description="Returns document, chunk, and embedding counts for a tenant.",
)
async def stats(
    tenant_id: UUID = Query(..., description="Tenant UUID"),
) -> StatsResponse:
    """Return per-tenant library stats."""
    try:
        pool = await get_kb_pool()
        data = await get_library_stats(pool, tenant_id)
        return StatsResponse(**data)

    except Exception as exc:
        logger.exception(
            "Library stats failed",
            tenant_id=str(tenant_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats failed: {exc}",
        )
