"""Library Document Ingestion Service.

Chunks and embeds user-uploaded documents (contracts, legal opinions, memos)
into ``kb.library_chunk`` and ``kb.library_chunk_embeddings``.

The FTS column in ``kb.library_chunk_fts`` is populated automatically by
the database trigger created in migration 084.

Usage:
    result = await ingest_library_document(pool, document_id, tenant_id, text)
    print(result.chunks_count, result.embeddings_count)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

import asyncpg
import structlog

from ..config import EMBEDDING_DIMS, EmbeddingModel, KBSettings
from .embedder import EmbeddingClient
from .generic_chunker import ChunkResult, GenericChunker

logger = structlog.get_logger(__name__)

# Library documents use Qwen3 1536d via LiteLLM (same as normativa).
_LIBRARY_MODEL = EmbeddingModel.QWEN3
_LIBRARY_CHANNEL = "testo"


@dataclass
class IngestResult:
    """Summary returned after ingesting a document."""

    document_id: UUID
    tenant_id: UUID
    chunks_count: int
    embeddings_count: int
    processing_time_ms: float


async def ingest_library_document(
    pool: asyncpg.Pool,
    document_id: UUID,
    tenant_id: UUID,
    text: str,
    *,
    embedding_client: EmbeddingClient | None = None,
    target_size: int = 1000,
    overlap_chars: int = 150,
) -> IngestResult:
    """Chunk and embed a library document.

    Steps:
        1. Chunk text with :class:`GenericChunker` (sliding window, overlap).
        2. INSERT chunks into ``kb.library_chunk`` (FTS trigger fires).
        3. Generate embeddings via :class:`EmbeddingClient`.
        4. INSERT embeddings into ``kb.library_chunk_embeddings``.

    Args:
        pool:             asyncpg connection pool to the KB database.
        document_id:      UUID of the library document (from core.library_documents).
        tenant_id:        Owning tenant UUID.
        text:             Plain-text content to chunk.
        embedding_client: Optional pre-initialised client; created from env if None.
        target_size:      Target chunk size in characters.
        overlap_chars:    Overlap between consecutive chunks.

    Returns:
        IngestResult with counts and timing.
    """
    t0 = time.perf_counter()

    # ── 1. Chunk ─────────────────────────────────────────────────
    chunker = GenericChunker(
        target_size=target_size,
        overlap_chars=overlap_chars,
    )
    chunks: list[ChunkResult] = chunker.chunk(text)

    if not chunks:
        logger.warning(
            "No chunks produced — text too short?",
            document_id=str(document_id),
            text_len=len(text),
        )
        return IngestResult(
            document_id=document_id,
            tenant_id=tenant_id,
            chunks_count=0,
            embeddings_count=0,
            processing_time_ms=(time.perf_counter() - t0) * 1000,
        )

    logger.info(
        "Chunks produced",
        document_id=str(document_id),
        count=len(chunks),
    )

    # ── 2. Store chunks ──────────────────────────────────────────
    chunk_ids: list[UUID] = []
    async with pool.acquire() as conn:
        # Delete existing chunks for this document (idempotent re-ingest)
        await conn.execute(
            "DELETE FROM kb.library_chunk WHERE document_id = $1",
            document_id,
        )

        for c in chunks:
            row = await conn.fetchrow(
                """
                INSERT INTO kb.library_chunk
                    (document_id, tenant_id, chunk_no, char_start, char_end, text, token_est)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                document_id,
                tenant_id,
                c.chunk_no,
                c.char_start,
                c.char_end,
                c.text,
                c.token_est,
            )
            chunk_ids.append(row["id"])

    logger.info(
        "Chunks stored",
        document_id=str(document_id),
        stored=len(chunk_ids),
    )

    # ── 3. Generate embeddings ───────────────────────────────────
    own_client = False
    if embedding_client is None:
        settings = KBSettings()
        embedding_client = EmbeddingClient(litellm_url=settings.kb_litellm_url)
        own_client = True

    embeddings_count = 0
    try:
        texts = [c.text for c in chunks]
        embeddings = await embedding_client.embed_batch(
            texts,
            _LIBRARY_MODEL,
            batch_size=32,
        )

        # ── 4. Store embeddings ──────────────────────────────────
        dims = EMBEDDING_DIMS[_LIBRARY_MODEL]
        async with pool.acquire() as conn:
            for chunk_id, emb in zip(chunk_ids, embeddings, strict=True):
                await conn.execute(
                    """
                    INSERT INTO kb.library_chunk_embeddings
                        (chunk_id, model, channel, dims, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (chunk_id, model, channel, dims) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            created_at = now()
                    """,
                    chunk_id,
                    _LIBRARY_MODEL.value,
                    _LIBRARY_CHANNEL,
                    dims,
                    emb,
                )
                embeddings_count += 1

        logger.info(
            "Embeddings stored",
            document_id=str(document_id),
            count=embeddings_count,
            model=_LIBRARY_MODEL.value,
            dims=dims,
        )
    except Exception:
        logger.exception(
            "Embedding generation/storage failed — chunks are still available for FTS",
            document_id=str(document_id),
        )
    finally:
        if own_client:
            await embedding_client.close()

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Library ingestion complete",
        document_id=str(document_id),
        chunks=len(chunk_ids),
        embeddings=embeddings_count,
        elapsed_ms=round(elapsed, 1),
    )

    return IngestResult(
        document_id=document_id,
        tenant_id=tenant_id,
        chunks_count=len(chunk_ids),
        embeddings_count=embeddings_count,
        processing_time_ms=round(elapsed, 1),
    )


async def delete_library_document_chunks(
    pool: asyncpg.Pool,
    document_id: UUID,
) -> int:
    """Delete all chunks (and cascade embeddings + FTS) for a document.

    Returns the number of deleted chunks.
    """
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM kb.library_chunk WHERE document_id = $1",
            document_id,
        )
        # result is e.g. "DELETE 42"
        count = int(result.split()[-1])
        logger.info(
            "Library chunks deleted",
            document_id=str(document_id),
            deleted=count,
        )
        return count


async def get_library_stats(
    pool: asyncpg.Pool,
    tenant_id: UUID,
) -> dict:
    """Return per-tenant library stats.

    Returns dict with keys: documents, chunks, embeddings.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT document_id) AS documents,
                COUNT(*) AS chunks,
                COALESCE(
                    (SELECT COUNT(*) FROM kb.library_chunk_embeddings e
                     WHERE e.chunk_id IN (
                         SELECT id FROM kb.library_chunk WHERE tenant_id = $1
                     )),
                    0
                ) AS embeddings
            FROM kb.library_chunk
            WHERE tenant_id = $1
            """,
            tenant_id,
        )
        return {
            "tenant_id": str(tenant_id),
            "documents": row["documents"],
            "chunks": row["chunks"],
            "embeddings": row["embeddings"],
        }
