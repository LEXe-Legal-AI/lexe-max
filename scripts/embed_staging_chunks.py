#!/usr/bin/env python3
"""
Generate embeddings for normativa chunks on STAGING.

Uses SSH tunnel on port 5437 -> staging:5436.
Includes Windows asyncio fix.

Usage:
    uv run python scripts/embed_staging_chunks.py --estimate-only
    uv run python scripts/embed_staging_chunks.py
"""

import argparse
import asyncio
import os
import platform
import sys
from datetime import datetime

# Windows event loop fix - MUST be before asyncpg import
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncpg
import httpx

# =============================================================================
# CONFIG
# =============================================================================

# Staging via SSH tunnel (5437 -> staging:5436)
DB_HOST = "localhost"
DB_PORT = 5437
DB_USER = "lexe_max"
DB_PASS = "lexe_max_dev_password"
DB_NAME = "lexe_max"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Embedding config
MODEL = "openai/text-embedding-3-small"
DIMS = 1536
CHANNEL = "testo"

# Cost estimation
EMBEDDING_PRICE_PER_1K = 0.00002  # USD per 1k tokens

# Processing
DEFAULT_BATCH_SIZE = 100


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

async def get_connection():
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        ssl=False
    )


# =============================================================================
# EMBEDDING CLIENT
# =============================================================================

async def get_embeddings(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenRouter API."""
    resp = await client.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": MODEL, "input": texts},
        timeout=120.0
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def get_cost_estimate(conn) -> dict:
    """Get cost estimate for all works."""
    rows = await conn.fetch("""
        SELECT
            w.code as work_code,
            count(*) as total_chunks,
            round(avg(c.token_est)) as avg_token_est,
            sum(c.token_est) as total_tokens
        FROM kb.normativa_chunk c
        JOIN kb.work w ON w.id = c.work_id
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.normativa_chunk_embeddings e
            WHERE e.chunk_id = c.id
            AND e.model = $1
            AND e.channel = $2
            AND e.dims = $3
        )
        GROUP BY w.code
        ORDER BY w.code
    """, MODEL, CHANNEL, DIMS)

    estimates = []
    for row in rows:
        cost_usd = (row['total_tokens'] / 1000) * EMBEDDING_PRICE_PER_1K
        estimates.append({
            "work_code": row['work_code'],
            "total_chunks": row['total_chunks'],
            "avg_token_est": row['avg_token_est'],
            "total_tokens": row['total_tokens'],
            "cost_usd": round(cost_usd, 4)
        })
    return {
        "works": estimates,
        "total_cost_usd": round(sum(e['cost_usd'] for e in estimates), 4)
    }


async def count_missing_chunks(conn) -> int:
    """Count chunks missing embeddings."""
    return await conn.fetchval("""
        SELECT count(*)
        FROM kb.normativa_chunk c
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.normativa_chunk_embeddings e
            WHERE e.chunk_id = c.id
            AND e.model = $1
            AND e.channel = $2
            AND e.dims = $3
        )
    """, MODEL, CHANNEL, DIMS)


async def get_missing_chunks(conn, limit: int = 100, offset: int = 0) -> list:
    """Get chunks missing embeddings."""
    return await conn.fetch("""
        SELECT c.id, c.text, c.token_est
        FROM kb.normativa_chunk c
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.normativa_chunk_embeddings e
            WHERE e.chunk_id = c.id
            AND e.model = $1
            AND e.channel = $2
            AND e.dims = $3
        )
        ORDER BY c.created_at, c.id
        LIMIT $4 OFFSET $5
    """, MODEL, CHANNEL, DIMS, limit, offset)


async def insert_embedding(conn, chunk_id, embedding: list[float]) -> None:
    """Insert embedding for a chunk."""
    await conn.execute("""
        INSERT INTO kb.normativa_chunk_embeddings (chunk_id, model, channel, dims, embedding)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (chunk_id, model, channel, dims) DO NOTHING
    """, chunk_id, MODEL, CHANNEL, DIMS, str(embedding))


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for staging chunks")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY and not args.estimate_only:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to STAGING via tunnel...")
    conn = await get_connection()
    print("Connected!")

    try:
        # Cost estimate
        estimate = await get_cost_estimate(conn)
        print(f"\n=== COST ESTIMATE ===")
        print(f"{'Work':<8} {'Chunks':>10} {'Avg Tok':>10} {'Cost USD':>12}")
        print("-" * 44)
        for e in estimate["works"]:
            print(f"{e['work_code']:<8} {e['total_chunks']:>10} {e['avg_token_est']:>10} ${e['cost_usd']:>10.4f}")
        print("-" * 44)
        total_chunks = sum(e['total_chunks'] for e in estimate['works'])
        print(f"{'TOTAL':<8} {total_chunks:>10} {'':>10} ${estimate['total_cost_usd']:>10.4f}")

        if args.estimate_only:
            await conn.close()
            return

        # Count missing
        missing = await count_missing_chunks(conn)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Chunks missing embeddings: {missing}")

        if missing == 0:
            print("Nothing to do!")
            await conn.close()
            return

        # Process batches
        async with httpx.AsyncClient() as client:
            processed = 0
            batch_no = 0
            offset = 0

            while processed < missing:
                chunks = await get_missing_chunks(conn, args.batch_size, 0)  # Always offset 0 since we insert
                if not chunks:
                    break

                batch_no += 1
                texts = [row['text'][:8000] for row in chunks]
                ids = [row['id'] for row in chunks]

                try:
                    embeddings = await get_embeddings(client, texts)

                    for chunk_id, emb in zip(ids, embeddings):
                        await insert_embedding(conn, chunk_id, emb)

                    processed += len(chunks)
                    ts = datetime.now().strftime('%H:%M:%S')
                    pct = 100 * processed / missing
                    print(f"  [{ts}] Batch {batch_no}: {processed}/{missing} ({pct:.1f}%)")

                except Exception as e:
                    print(f"  ERROR at batch {batch_no}: {e}")
                    await asyncio.sleep(5)
                    continue

                await asyncio.sleep(0.5)

        # Final stats
        total_emb = await conn.fetchval(
            "SELECT COUNT(*) FROM kb.normativa_chunk_embeddings WHERE model = $1",
            MODEL
        )
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Total chunk embeddings: {total_emb}")

    finally:
        await conn.close()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
