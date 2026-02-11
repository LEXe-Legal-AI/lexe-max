#!/usr/bin/env python3
"""
Hybrid Retriever per KB Normativa + Annotations.

Combina:
- Dense search (vector embeddings)
- Sparse search (tsvector FTS)
- RRF fusion per ranking unificato

Usage:
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/hybrid_retriever.py "query"
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Optional

import httpx
import asyncpg


# =============================================================================
# CONFIG
# =============================================================================

DB_URL = os.environ.get(
    "LEXE_KB_DSN",
    "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# RRF parameters
RRF_K = 60  # Standard RRF constant
DENSE_LIMIT = 50  # Top-K for dense search
SPARSE_LIMIT = 50  # Top-K for sparse search

# Weights for source types (optional boosting)
WEIGHT_NORMATIVA = 1.0
WEIGHT_ANNOTATION = 0.9  # Slight penalty to avoid note spam


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SearchResult:
    """Single search result."""
    id: str
    source_type: str  # 'NORM' or 'NOTE'
    code: Optional[str]
    article: Optional[str]
    title: Optional[str]
    preview: Optional[str]
    rrf_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]
    sources: list[str]  # e.g. ['ND', 'NS', 'AD', 'AS']


# =============================================================================
# EMBEDDING CLIENT
# =============================================================================

async def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": EMBEDDING_MODEL, "input": [text]},
            timeout=30.0
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

async def search_normativa_dense(
    conn: asyncpg.Connection,
    query_embedding: list[float],
    limit: int = DENSE_LIMIT
) -> list[dict]:
    """Dense search on normativa embeddings."""
    return await conn.fetch(
        """
        SELECT
            n.id,
            'NORM' as source_type,
            w.code,
            n.articolo as article,
            n.rubrica as title,
            LEFT(n.testo, 200) as preview,
            1 - (e.embedding <=> $1::vector) as score
        FROM kb.normativa_embeddings e
        JOIN kb.normativa n ON n.id = e.normativa_id
        JOIN kb.work w ON w.id = n.work_id
        WHERE e.channel = 'testo'
        ORDER BY e.embedding <=> $1::vector
        LIMIT $2
        """,
        str(query_embedding),
        limit
    )


async def search_normativa_sparse(
    conn: asyncpg.Connection,
    query: str,
    limit: int = SPARSE_LIMIT
) -> list[dict]:
    """Sparse FTS search on normativa."""
    # Use plainto_tsquery for safe query parsing (no special chars)
    return await conn.fetch(
        """
        SELECT
            n.id,
            'NORM' as source_type,
            w.code,
            n.articolo as article,
            n.rubrica as title,
            LEFT(n.testo, 200) as preview,
            ts_rank_cd(f.tsv_it, plainto_tsquery('italian', $1)) as score
        FROM kb.normativa_fts f
        JOIN kb.normativa n ON n.id = f.normativa_id
        JOIN kb.work w ON w.id = n.work_id
        WHERE f.tsv_it @@ plainto_tsquery('italian', $1)
        ORDER BY score DESC
        LIMIT $2
        """,
        query,
        limit
    )


async def search_annotation_dense(
    conn: asyncpg.Connection,
    query_embedding: list[float],
    limit: int = DENSE_LIMIT
) -> list[dict]:
    """Dense search on annotation embeddings."""
    return await conn.fetch(
        """
        SELECT
            a.id,
            'NOTE' as source_type,
            w.code,
            n.articolo as article,
            a.title,
            LEFT(a.content, 200) as preview,
            1 - (e.embedding <=> $1::vector) as score
        FROM kb.annotation_embeddings e
        JOIN kb.annotation a ON a.id = e.annotation_id
        LEFT JOIN kb.annotation_link al ON al.annotation_id = a.id
        LEFT JOIN kb.normativa n ON n.id = al.normativa_id
        LEFT JOIN kb.work w ON w.id = n.work_id
        ORDER BY e.embedding <=> $1::vector
        LIMIT $2
        """,
        str(query_embedding),
        limit
    )


async def search_annotation_sparse(
    conn: asyncpg.Connection,
    query: str,
    limit: int = SPARSE_LIMIT
) -> list[dict]:
    """Sparse FTS search on annotations."""
    return await conn.fetch(
        """
        SELECT
            a.id,
            'NOTE' as source_type,
            w.code,
            n.articolo as article,
            a.title,
            LEFT(a.content, 200) as preview,
            ts_rank_cd(f.tsv_it, plainto_tsquery('italian', $1)) as score
        FROM kb.annotation_fts f
        JOIN kb.annotation a ON a.id = f.annotation_id
        LEFT JOIN kb.annotation_link al ON al.annotation_id = a.id
        LEFT JOIN kb.normativa n ON n.id = al.normativa_id
        LEFT JOIN kb.work w ON w.id = n.work_id
        WHERE f.tsv_it @@ plainto_tsquery('italian', $1)
        ORDER BY score DESC
        LIMIT $2
        """,
        query,
        limit
    )


# =============================================================================
# RRF FUSION
# =============================================================================

def rrf_fusion(
    results_lists: list[tuple[list[dict], str, float]],
    k: int = RRF_K
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Args:
        results_lists: List of (results, source_tag, weight)
        k: RRF constant (default 60)

    Returns:
        Fused and sorted results
    """
    scores: dict[str, dict] = {}

    for results, source_tag, weight in results_lists:
        for rank, row in enumerate(results):
            doc_id = f"{row['source_type']}:{row['id']}"

            if doc_id not in scores:
                scores[doc_id] = {
                    'id': str(row['id']),
                    'source_type': row['source_type'],
                    'code': row['code'],
                    'article': row['article'],
                    'title': row['title'],
                    'preview': row['preview'],
                    'rrf': 0.0,
                    'dense_rank': None,
                    'sparse_rank': None,
                    'sources': []
                }

            # RRF score with optional weight
            rrf_score = weight / (k + rank + 1)
            scores[doc_id]['rrf'] += rrf_score
            scores[doc_id]['sources'].append(source_tag)

            # Track ranks
            if source_tag.endswith('D'):  # Dense
                scores[doc_id]['dense_rank'] = rank + 1
            else:  # Sparse
                scores[doc_id]['sparse_rank'] = rank + 1

    # Sort by RRF score descending
    ranked = sorted(scores.values(), key=lambda x: x['rrf'], reverse=True)

    return [
        SearchResult(
            id=r['id'],
            source_type=r['source_type'],
            code=r['code'],
            article=r['article'],
            title=r['title'],
            preview=r['preview'],
            rrf_score=r['rrf'],
            dense_rank=r['dense_rank'],
            sparse_rank=r['sparse_rank'],
            sources=r['sources']
        )
        for r in ranked
    ]


# =============================================================================
# MAIN HYBRID SEARCH
# =============================================================================

async def hybrid_search(
    query: str,
    top_k: int = 10,
    include_normativa: bool = True,
    include_annotations: bool = True
) -> list[SearchResult]:
    """
    Perform hybrid search across normativa and annotations.

    Args:
        query: Search query
        top_k: Number of results to return
        include_normativa: Include normativa in search
        include_annotations: Include annotations in search

    Returns:
        List of SearchResult ordered by RRF score
    """
    # Get query embedding
    query_embedding = await get_embedding(query)

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)

    try:
        results_lists = []

        if include_normativa:
            # Normativa dense
            norm_dense = await search_normativa_dense(conn, query_embedding)
            results_lists.append((norm_dense, 'ND', WEIGHT_NORMATIVA))

            # Normativa sparse
            norm_sparse = await search_normativa_sparse(conn, query)
            results_lists.append((norm_sparse, 'NS', WEIGHT_NORMATIVA))

        if include_annotations:
            # Annotation dense
            ann_dense = await search_annotation_dense(conn, query_embedding)
            results_lists.append((ann_dense, 'AD', WEIGHT_ANNOTATION))

            # Annotation sparse
            ann_sparse = await search_annotation_sparse(conn, query)
            results_lists.append((ann_sparse, 'AS', WEIGHT_ANNOTATION))

        # RRF fusion
        fused = rrf_fusion(results_lists)

        return fused[:top_k]

    finally:
        await conn.close()


# =============================================================================
# CLI
# =============================================================================

async def main():
    if len(sys.argv) < 2:
        print("Usage: python hybrid_retriever.py 'query'")
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f'Query: "{query}"')
    print("=" * 80)

    results = await hybrid_search(query, top_k=top_k)

    print(f"\n{'#':<3} {'Type':<5} {'Code':<6} {'Article':<10} {'RRF':<8} {'Sources':<12} Title/Preview")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        code = r.code or '-'
        article = r.article or '-'
        sources = '+'.join(r.sources)
        title = (r.title[:40] + '..') if r.title and len(r.title) > 40 else (r.title or '')
        print(f"{i:<3} {r.source_type:<5} {code:<6} {article:<10} {r.rrf_score:<8.4f} {sources:<12} {title}")


if __name__ == "__main__":
    asyncio.run(main())
