#!/usr/bin/env python3
"""
Generate Embeddings for Principle Anchors (NOVA RATIO)

Generates text-embedding-3-small (1536d) embeddings for all principle_anchors
that have NULL embedding, and updates them in-place.

Usage:
    uv run python scripts/graph/seed_principle_embeddings.py
    uv run python scripts/graph/seed_principle_embeddings.py --dry-run
    uv run python scripts/graph/seed_principle_embeddings.py --litellm-url http://localhost:4001

Requires:
    - lexe-max PostgreSQL running with kb.principle_anchors populated (migration 082)
    - LiteLLM proxy accessible for embedding generation
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import asyncpg
import httpx
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = structlog.get_logger(__name__)

# Default connection
DEFAULT_DSN = os.environ.get(
    "KB_DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
)
DEFAULT_LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:4001")
DEFAULT_LITELLM_KEY = os.environ.get("LITELLM_API_KEY", "sk-lexe-tools")
EMBEDDING_MODEL = "lexe-embedding"  # alias for text-embedding-3-small


async def get_principles_without_embedding(conn: asyncpg.Connection) -> list[dict]:
    """Fetch all principle_anchors with NULL embedding."""
    rows = await conn.fetch(
        """
        SELECT id, principle_name, principle_text, keywords
        FROM kb.principle_anchors
        WHERE embedding IS NULL AND is_active = TRUE
        ORDER BY id
        """
    )
    return [
        {
            "id": r["id"],
            "name": r["principle_name"],
            "text": r["principle_text"],
            "keywords": r["keywords"] or [],
        }
        for r in rows
    ]


def build_embedding_text(principle: dict) -> str:
    """Build the text to embed for a principle.

    Combines name + text + keywords for maximum semantic coverage.
    """
    parts = [
        f"Principio giuridico: {principle['name']}",
        principle["text"],
    ]
    if principle["keywords"]:
        parts.append(f"Keywords: {', '.join(principle['keywords'])}")
    return " ".join(parts)


async def generate_embeddings(
    texts: list[str],
    litellm_url: str,
    litellm_key: str,
    model: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Generate embeddings via LiteLLM proxy."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{litellm_url}/v1/embeddings",
            headers={"Authorization": f"Bearer {litellm_key}"},
            json={
                "model": model,
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


async def update_embeddings(
    conn: asyncpg.Connection,
    principle_id: str,
    embedding: list[float],
) -> None:
    """Update a single principle_anchor with its embedding."""
    await conn.execute(
        """
        UPDATE kb.principle_anchors
        SET embedding = $1::vector, updated_at = NOW()
        WHERE id = $2
        """,
        str(embedding),  # pgvector accepts string representation
        principle_id,
    )


async def main(args: argparse.Namespace) -> None:
    logger.info("Connecting to database", dsn=args.dsn.split("@")[-1])
    conn = await asyncpg.connect(args.dsn)

    try:
        principles = await get_principles_without_embedding(conn)
        logger.info("Found principles without embedding", count=len(principles))

        if not principles:
            logger.info("All principles already have embeddings. Nothing to do.")
            return

        # Build texts for embedding
        texts = [build_embedding_text(p) for p in principles]

        if args.dry_run:
            for p, t in zip(principles, texts):
                logger.info("Would embed", id=p["id"], text_preview=t[:80])
            logger.info("Dry run complete", total=len(principles))
            return

        # Generate embeddings in batches of 10
        batch_size = 10
        updated = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_principles = principles[i : i + batch_size]

            logger.info(
                "Generating embeddings",
                batch=i // batch_size + 1,
                count=len(batch_texts),
            )

            embeddings = await generate_embeddings(
                batch_texts, args.litellm_url, args.litellm_key,
            )

            for principle, embedding in zip(batch_principles, embeddings):
                await update_embeddings(conn, principle["id"], embedding)
                updated += 1
                logger.info(
                    "Updated embedding",
                    id=principle["id"],
                    name=principle["name"],
                    dims=len(embedding),
                )

        logger.info("Embedding generation complete", total_updated=updated)

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for principle_anchors")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument("--litellm-url", default=DEFAULT_LITELLM_URL, help="LiteLLM proxy URL")
    parser.add_argument("--litellm-key", default=DEFAULT_LITELLM_KEY, help="LiteLLM API key")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")

    args = parser.parse_args()
    asyncio.run(main(args))
