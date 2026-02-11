"""
LEXE Knowledge Base - Altalex Normativa Storage

UPSERT operations for kb.normativa_altalex table.
Handles articles and embeddings with idempotent inserts.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg
import structlog

from .marker_chunker import ExtractedArticle

logger = structlog.get_logger(__name__)


@dataclass
class StoredArticle:
    """Result of storing an article."""
    id: UUID
    global_key: str
    codice: str
    articolo: str
    is_new: bool  # True if INSERT, False if UPDATE


@dataclass
class StoreResult:
    """Result of batch store operation."""
    inserted: int
    updated: int
    failed: int
    articles: list[StoredArticle]
    errors: list[dict]


class AltalexStore:
    """
    Storage layer for Altalex normativa articles.

    Uses UPSERT (INSERT ... ON CONFLICT DO UPDATE) for idempotent operations.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize store with database pool.

        Args:
            pool: asyncpg connection pool (KB database)
        """
        self._pool = pool

    async def store_article(
        self,
        article: ExtractedArticle,
        codice: str,
        embedding: list[float] | None = None,
        embedding_model: str = "openai/text-embedding-3-small",
        source_file: str | None = None,
        conn: asyncpg.Connection | None = None,
    ) -> StoredArticle:
        """
        Store single article with UPSERT.

        Args:
            article: Extracted article from marker_chunker
            codice: Document code (CC, CP, GDPR, etc.)
            embedding: Embedding vector (optional)
            embedding_model: Model used for embedding
            source_file: Source JSON file path
            conn: Optional connection (for batch operations)

        Returns:
            StoredArticle with ID and status
        """
        should_release = conn is None
        if conn is None:
            conn = await self._pool.acquire()

        try:
            # UPSERT article
            row = await conn.fetchrow(
                """
                INSERT INTO kb.normativa_altalex (
                    codice, articolo, articolo_num_norm, articolo_suffix,
                    rubrica, testo, testo_context,
                    libro, titolo, capo, sezione,
                    page_start, page_end, content_hash,
                    marker_json_path
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                )
                ON CONFLICT (global_key) DO UPDATE SET
                    rubrica = EXCLUDED.rubrica,
                    testo = EXCLUDED.testo,
                    testo_context = EXCLUDED.testo_context,
                    libro = EXCLUDED.libro,
                    titolo = EXCLUDED.titolo,
                    capo = EXCLUDED.capo,
                    sezione = EXCLUDED.sezione,
                    page_start = EXCLUDED.page_start,
                    page_end = EXCLUDED.page_end,
                    content_hash = EXCLUDED.content_hash,
                    marker_json_path = EXCLUDED.marker_json_path,
                    updated_at = NOW()
                RETURNING id, global_key, (xmax = 0) AS is_new
                """,
                codice,
                article.articolo_num,
                article.articolo_num_norm,
                article.articolo_suffix,
                article.rubrica,
                article.testo,
                article.testo_context,
                article.libro,
                article.titolo,
                article.capo,
                article.sezione,
                article.page_start,
                article.page_end,
                article.content_hash,
                source_file,
            )

            article_id = row["id"]
            global_key = row["global_key"]
            is_new = row["is_new"]

            # Store embedding if provided
            if embedding:
                await self._store_embedding(
                    conn=conn,
                    altalex_id=article_id,
                    embedding=embedding,
                    model=embedding_model,
                    text_hash=article.content_hash,
                )

            return StoredArticle(
                id=article_id,
                global_key=global_key,
                codice=codice,
                articolo=article.articolo_num,
                is_new=is_new,
            )
        finally:
            if should_release:
                await self._pool.release(conn)

    async def _store_embedding(
        self,
        conn: asyncpg.Connection,
        altalex_id: UUID,
        embedding: list[float],
        model: str,
        text_hash: str,
        channel: str = "testo",
    ) -> None:
        """Store embedding with UPSERT."""
        dims = len(embedding)

        # Convert list to pgvector string format: "[0.1, 0.2, ...]"
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        await conn.execute(
            """
            INSERT INTO kb.altalex_embeddings (
                altalex_id, model, channel, dims, embedding, text_hash
            ) VALUES ($1, $2, $3, $4, $5::vector, $6)
            ON CONFLICT (altalex_id, model, channel) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                text_hash = EXCLUDED.text_hash,
                dims = EXCLUDED.dims,
                created_at = NOW()
            """,
            altalex_id,
            model,
            channel,
            dims,
            embedding_str,
            text_hash,
        )

    async def store_batch(
        self,
        articles: list[tuple[ExtractedArticle, list[float] | None]],
        codice: str,
        embedding_model: str = "openai/text-embedding-3-small",
        source_file: str | None = None,
        batch_size: int = 100,
    ) -> StoreResult:
        """
        Store batch of articles with embeddings.

        Args:
            articles: List of (ExtractedArticle, embedding) tuples
            codice: Document code
            embedding_model: Model used for embeddings
            source_file: Source file path
            batch_size: Transaction batch size

        Returns:
            StoreResult with counts and stored articles
        """
        inserted = 0
        updated = 0
        failed = 0
        stored_articles: list[StoredArticle] = []
        errors: list[dict] = []

        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]

            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for article, embedding in batch:
                        try:
                            result = await self.store_article(
                                article=article,
                                codice=codice,
                                embedding=embedding,
                                embedding_model=embedding_model,
                                source_file=source_file,
                                conn=conn,  # Pass connection for batch
                            )

                            if result.is_new:
                                inserted += 1
                            else:
                                updated += 1

                            stored_articles.append(result)

                        except Exception as e:
                            failed += 1
                            errors.append({
                                "articolo": article.articolo_num,
                                "error": str(e),
                            })
                            logger.warning(
                                "Failed to store article",
                                articolo=article.articolo_num,
                                error=str(e),
                            )

        logger.info(
            "Batch store complete",
            codice=codice,
            inserted=inserted,
            updated=updated,
            failed=failed,
        )

        return StoreResult(
            inserted=inserted,
            updated=updated,
            failed=failed,
            articles=stored_articles,
            errors=errors,
        )

    async def log_ingestion(
        self,
        source_file: str,
        codice: str,
        articolo: str | None,
        stage: str,
        status: str,
        message: str | None = None,
        details: dict | None = None,
    ) -> None:
        """
        Log ingestion event (success, warning, error, quarantine).

        Args:
            source_file: Source file path
            codice: Document code
            articolo: Article number (if applicable)
            stage: Pipeline stage (convert, chunk, validate, embed, store)
            status: Status (success, warning, error, quarantine)
            message: Human-readable message
            details: Additional JSON details
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO kb.altalex_ingestion_logs (
                    source_file, codice, articolo, stage, status, message, details
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                source_file,
                codice,
                articolo,
                stage,
                status,
                message,
                json.dumps(details or {}),
            )

    async def get_document_stats(self, codice: str) -> dict:
        """Get statistics for a document."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_articles,
                    COUNT(DISTINCT articolo_num_norm) as unique_articles,
                    MIN(articolo_num_norm) as min_article,
                    MAX(articolo_num_norm) as max_article,
                    COUNT(*) FILTER (WHERE testo IS NOT NULL AND testo != '') as with_text,
                    COUNT(*) FILTER (WHERE rubrica IS NOT NULL) as with_rubrica
                FROM kb.normativa_altalex
                WHERE codice = $1
                """,
                codice,
            )
            return dict(row) if row else {}

    async def count_embeddings(self, codice: str) -> int:
        """Count embeddings for a document."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as count
                FROM kb.altalex_embeddings e
                JOIN kb.normativa_altalex a ON e.altalex_id = a.id
                WHERE a.codice = $1
                """,
                codice,
            )
            return row["count"] if row else 0


async def create_altalex_store() -> AltalexStore:
    """
    Create AltalexStore with KB database pool.

    Uses the KB database connection string from environment.
    """
    from ...database import get_kb_pool

    pool = await get_kb_pool()
    return AltalexStore(pool)
