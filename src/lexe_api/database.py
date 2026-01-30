"""LEXe Database Client.

Async PostgreSQL client for legal document storage.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import structlog

from lexe_api.config import settings

logger = structlog.get_logger(__name__)


class DatabaseClient:
    """Async PostgreSQL client for LEXe."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            logger.info("Database pool created", dsn=self.dsn[:30] + "...")

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.connect()
        async with self._pool.acquire() as conn:
            yield conn

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def find_document_by_urn(self, urn: str) -> dict | None:
        """Find document by URN."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, source, urn, act_type, act_number, act_date,
                       article, version, title, content, is_vigente,
                       abrogato_da, modificato_da, metadata, scraped_at
                FROM documents
                WHERE urn = $1
                """,
                urn,
            )
            if row:
                return dict(row)
            return None

    async def store_document(
        self,
        source: str,
        urn: str,
        act_type: str | None = None,
        act_number: str | None = None,
        act_date: datetime | None = None,
        article: str | None = None,
        version: str = "vigente",
        title: str | None = None,
        content: str | None = None,
        html_raw: str | None = None,
        is_vigente: bool = True,
        abrogato_da: str | None = None,
        modificato_da: list[str] | None = None,
        metadata: dict | None = None,
    ) -> UUID:
        """Store or update a document."""
        expires_at = datetime.utcnow() + timedelta(days=settings.document_expire_days)

        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    source, urn, act_type, act_number, act_date, article,
                    version, title, content, html_raw, is_vigente,
                    abrogato_da, modificato_da, metadata, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (urn) DO UPDATE SET
                    content = EXCLUDED.content,
                    html_raw = EXCLUDED.html_raw,
                    is_vigente = EXCLUDED.is_vigente,
                    abrogato_da = EXCLUDED.abrogato_da,
                    modificato_da = EXCLUDED.modificato_da,
                    metadata = EXCLUDED.metadata,
                    scraped_at = NOW(),
                    expires_at = EXCLUDED.expires_at
                RETURNING id
                """,
                source,
                urn,
                act_type,
                act_number,
                act_date,
                article,
                version,
                title,
                content,
                html_raw,
                is_vigente,
                abrogato_da,
                modificato_da,
                json.dumps(metadata or {}),
                expires_at,
            )
            return row["id"]

    # =========================================================================
    # Massime Operations
    # =========================================================================

    async def store_massima(
        self,
        document_id: UUID,
        autorita: str,
        numero: str | None = None,
        data: datetime | None = None,
        testo: str = "",
        principio: str | None = None,
        sezione: str | None = None,
        keywords: list[str] | None = None,
        materia: str | None = None,
        brocardi_url: str | None = None,
        metadata: dict | None = None,
    ) -> UUID:
        """Store a case law summary (massima)."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO massime (
                    document_id, autorita, sezione, numero, data,
                    testo, principio, keywords, materia, brocardi_url, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                document_id,
                autorita,
                sezione,
                numero,
                data,
                testo,
                principio,
                keywords or [],
                materia,
                brocardi_url,
                json.dumps(metadata or {}),
            )
            return row["id"]

    async def get_massime_for_document(self, document_id: UUID) -> list[dict]:
        """Get all massime for a document."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, autorita, sezione, numero, data, testo,
                       principio, keywords, materia, brocardi_url
                FROM massime
                WHERE document_id = $1
                ORDER BY data DESC NULLS LAST
                """,
                document_id,
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # Tool Health Operations
    # =========================================================================

    async def get_tool_health(self, tool_name: str) -> dict | None:
        """Get health status for a tool."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT tool_name, state, failure_count, success_count,
                       circuit_state, circuit_opened_at, circuit_retry_at,
                       last_success_at, last_failure_at, last_error_message
                FROM tool_health
                WHERE tool_name = $1
                """,
                tool_name,
            )
            if row:
                return dict(row)
            return None

    async def update_tool_health(
        self,
        tool_name: str,
        *,
        state: str | None = None,
        failure_count: int | None = None,
        success_count: int | None = None,
        circuit_state: str | None = None,
        circuit_opened_at: datetime | None = None,
        circuit_retry_at: datetime | None = None,
        last_success_at: datetime | None = None,
        last_failure_at: datetime | None = None,
        last_error_message: str | None = None,
        last_error_type: str | None = None,
    ) -> None:
        """Update tool health status."""
        updates: list[str] = []
        values: list[Any] = [tool_name]
        idx = 2

        if state is not None:
            updates.append(f"state = ${idx}")
            values.append(state)
            idx += 1
        if failure_count is not None:
            updates.append(f"failure_count = ${idx}")
            values.append(failure_count)
            idx += 1
        if success_count is not None:
            updates.append(f"success_count = ${idx}")
            values.append(success_count)
            idx += 1
        if circuit_state is not None:
            updates.append(f"circuit_state = ${idx}")
            values.append(circuit_state)
            idx += 1
        if circuit_opened_at is not None:
            updates.append(f"circuit_opened_at = ${idx}")
            values.append(circuit_opened_at)
            idx += 1
        if circuit_retry_at is not None:
            updates.append(f"circuit_retry_at = ${idx}")
            values.append(circuit_retry_at)
            idx += 1
        if last_success_at is not None:
            updates.append(f"last_success_at = ${idx}")
            values.append(last_success_at)
            idx += 1
        if last_failure_at is not None:
            updates.append(f"last_failure_at = ${idx}")
            values.append(last_failure_at)
            idx += 1
        if last_error_message is not None:
            updates.append(f"last_error_message = ${idx}")
            values.append(last_error_message)
            idx += 1
        if last_error_type is not None:
            updates.append(f"last_error_type = ${idx}")
            values.append(last_error_type)
            idx += 1

        if updates:
            async with self.acquire() as conn:
                await conn.execute(
                    f"UPDATE tool_health SET {', '.join(updates)} WHERE tool_name = $1",
                    *values,
                )

    async def increment_tool_failure(self, tool_name: str) -> int:
        """Increment failure count and return new value."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE tool_health
                SET failure_count = failure_count + 1,
                    last_failure_at = NOW()
                WHERE tool_name = $1
                RETURNING failure_count
                """,
                tool_name,
            )
            return row["failure_count"] if row else 0

    async def increment_tool_success(self, tool_name: str) -> None:
        """Increment success count and reset failures."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                UPDATE tool_health
                SET success_count = success_count + 1,
                    failure_count = 0,
                    last_success_at = NOW(),
                    state = 'healthy',
                    circuit_state = 'closed'
                WHERE tool_name = $1
                """,
                tool_name,
            )


# Global instance
db = DatabaseClient()
