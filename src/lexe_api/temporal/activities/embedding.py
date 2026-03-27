"""Embedding activity for KB nightly sync.

Generates vector embeddings for normativa chunks via LiteLLM gateway
and upserts them into kb.normativa_chunk_embeddings.

Schema reference: kb.normativa_chunk_embeddings (055_chunking_schema.sql)
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from uuid import UUID

import asyncpg
import httpx
from temporalio import activity

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
)
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

# Model alias configured in LiteLLM for embeddings
EMBEDDING_MODEL = "lexe-embedding"
EMBEDDING_DIMS = 1536
EMBEDDING_CHANNEL = "testo"

# Batch size: LiteLLM / OpenAI accept up to ~2048 inputs, but we keep it
# conservative to stay within token limits and provide regular heartbeats.
BATCH_SIZE = 50

_TIMEOUT_S = 60.0  # Embedding calls can be slower than search


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EmbedResult:
    """Result of an embedding generation run."""

    chunks_embedded: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn
async def reembed_chunks(chunk_ids: list[str]) -> dict:
    """Generate embeddings for normativa chunks and upsert into the KB.

    For each chunk:
    1. Fetch text from kb.normativa_chunk
    2. Call LiteLLM embeddings endpoint in batches of 50
    3. UPSERT into kb.normativa_chunk_embeddings

    Args:
        chunk_ids: List of normativa_chunk UUID strings to embed.

    Returns:
        Dict serialisation of EmbedResult.
    """
    if not chunk_ids:
        return asdict(EmbedResult())

    result = EmbedResult()

    conn: asyncpg.Connection = await asyncpg.connect(DATABASE_URL)
    try:
        # Fetch all chunk texts in one query for efficiency
        uuid_ids = [UUID(cid) for cid in chunk_ids]
        rows = await conn.fetch(
            """
            SELECT id, text
            FROM kb.normativa_chunk
            WHERE id = ANY($1::uuid[])
            """,
            uuid_ids,
        )

        # Build lookup: id -> text
        chunk_texts: dict[UUID, str] = {row["id"]: row["text"] for row in rows}

        if not chunk_texts:
            logger.warning("reembed_chunks: none of %d chunk_ids found in DB", len(chunk_ids))
            return asdict(result)

        # Process in batches
        items = list(chunk_texts.items())

        async with httpx.AsyncClient(
            base_url=LITELLM_API_BASE,
            timeout=_TIMEOUT_S,
            headers={
                "Authorization": f"Bearer {LITELLM_API_KEY}" if LITELLM_API_KEY else "",
                "Content-Type": "application/json",
            },
        ) as http_client:

            for batch_start in range(0, len(items), BATCH_SIZE):
                activity.heartbeat()

                batch = items[batch_start : batch_start + BATCH_SIZE]
                batch_ids = [bid for bid, _ in batch]
                batch_texts = [text for _, text in batch]

                try:
                    embeddings = await _call_embedding_api(http_client, batch_texts)
                except Exception as exc:
                    logger.error(
                        "reembed_chunks: embedding API error for batch %d-%d: %s",
                        batch_start,
                        batch_start + len(batch),
                        exc,
                    )
                    result.failed += len(batch)
                    continue

                if len(embeddings) != len(batch):
                    logger.error(
                        "reembed_chunks: expected %d embeddings, got %d",
                        len(batch),
                        len(embeddings),
                    )
                    result.failed += len(batch)
                    continue

                # Upsert embeddings into DB
                for chunk_id, embedding in zip(batch_ids, embeddings):
                    try:
                        await conn.execute(
                            """
                            INSERT INTO kb.normativa_chunk_embeddings (
                                chunk_id, model, channel, dims, embedding
                            ) VALUES ($1, $2, $3, $4, $5::vector)
                            ON CONFLICT (chunk_id, model, channel, dims)
                            DO UPDATE SET embedding = EXCLUDED.embedding,
                                         created_at = NOW()
                            """,
                            chunk_id,
                            EMBEDDING_MODEL,
                            EMBEDDING_CHANNEL,
                            EMBEDDING_DIMS,
                            _format_vector(embedding),
                        )
                        result.chunks_embedded += 1
                    except Exception as exc:
                        logger.error(
                            "reembed_chunks: DB upsert error for chunk %s: %s",
                            chunk_id,
                            exc,
                        )
                        result.failed += 1

                logger.info(
                    "reembed_chunks: batch %d-%d done (%d embedded, %d failed so far)",
                    batch_start,
                    batch_start + len(batch),
                    result.chunks_embedded,
                    result.failed,
                )

    finally:
        await conn.close()

    logger.info(
        "reembed_chunks: total embedded=%d failed=%d",
        result.chunks_embedded,
        result.failed,
    )
    return asdict(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_embedding_api(
    client: httpx.AsyncClient,
    texts: list[str],
) -> list[list[float]]:
    """Call LiteLLM /v1/embeddings endpoint and return embedding vectors."""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    response = await client.post("/v1/embeddings", json=payload)
    response.raise_for_status()

    data = response.json()
    # OpenAI-compatible response: {"data": [{"embedding": [...], "index": 0}, ...]}
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def _format_vector(embedding: list[float]) -> str:
    """Format embedding as pgvector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
