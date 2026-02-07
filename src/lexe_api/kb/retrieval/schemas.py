"""
LEXE Knowledge Base - Normativa Retrieval Schemas

Pydantic models for KB Normativa hybrid search API.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search mode for KB Normativa."""

    HYBRID = "hybrid"  # Dense + Sparse + RRF (default)
    DENSE = "dense"  # Vector search only
    SPARSE = "sparse"  # BM25 only


class NormativaSearchRequest(BaseModel):
    """Request for KB Normativa search."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Search query in natural language",
        examples=["risarcimento danno responsabilita civile"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return",
    )
    codes: list[str] | None = Field(
        default=None,
        description="Filter by work codes (e.g., ['CC', 'CPC']). If None, search all.",
        examples=[["CC", "CP"], ["CPC"]],
    )
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode: hybrid, dense, or sparse",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold (filters results below this)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "risarcimento danno responsabilita civile",
                    "top_k": 10,
                    "codes": ["CC"],
                    "mode": "hybrid",
                }
            ]
        }
    }


class NormativaSearchResult(BaseModel):
    """Single search result from KB Normativa."""

    # Document identification
    code: str = Field(..., description="Work code (e.g., CC, CPC, CP)")
    article: str = Field(..., description="Article number/identifier")
    chunk_no: int = Field(..., description="Chunk number within article")
    source: str = Field(
        default="altalex",
        description="Content source (altalex, brocardi, etc.)",
    )

    # Scores
    rrf_score: float = Field(..., description="RRF fusion score (0-1)")
    dense_score: float | None = Field(None, description="Dense (vector) similarity score")
    sparse_score: float | None = Field(None, description="Sparse (BM25) score")

    # Content
    text: str = Field(..., description="Chunk text content")
    text_preview: str | None = Field(
        None,
        description="Truncated text preview (first 200 chars)",
    )

    # Metadata
    work_id: int | None = Field(None, description="Work table ID")
    normativa_id: int | None = Field(None, description="Normativa table ID")
    chunk_id: int | None = Field(None, description="Chunk table ID")


class NormativaSearchResponse(BaseModel):
    """Response from KB Normativa search."""

    success: bool = Field(default=True)
    query: str = Field(..., description="Original query")
    mode: SearchMode = Field(..., description="Search mode used")
    total_results: int = Field(..., description="Number of results returned")
    results: list[NormativaSearchResult] = Field(default_factory=list)

    # Search metadata
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    embedding_time_ms: float | None = Field(
        None,
        description="Embedding generation time in milliseconds",
    )
    total_chunks: int | None = Field(
        None,
        description="Total chunks in corpus (for context)",
    )

    # Error info (if any)
    error: str | None = Field(None, description="Error message if search failed")


class NormativaStatsResponse(BaseModel):
    """Response with KB Normativa corpus statistics."""

    success: bool = Field(default=True)

    # Corpus stats
    total_works: int = Field(..., description="Total works/codes in KB")
    total_articles: int = Field(..., description="Total articles (normativa rows)")
    total_chunks: int = Field(..., description="Total chunks")
    total_embeddings: int = Field(..., description="Total embeddings generated")

    # Coverage
    embedding_coverage_pct: float = Field(
        ...,
        description="Percentage of chunks with embeddings",
    )

    # By work breakdown
    works: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Stats per work code",
    )

    # Search readiness
    search_ready: bool = Field(
        ...,
        description="True if hybrid search is available",
    )
    fts_ready: bool = Field(
        ...,
        description="True if full-text search is available",
    )


class ErrorResponse(BaseModel):
    """API error response."""

    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_type: str | None = Field(None, description="Error classification")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
