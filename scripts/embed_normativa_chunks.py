#!/usr/bin/env python3
"""
Generate embeddings for normativa chunks.

Resumable with cursor-based pagination (created_at, id).
Includes cost estimation and threshold warning.

Usage:
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/embed_normativa_chunks.py
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/embed_normativa_chunks.py --batch-size 50
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/embed_normativa_chunks.py --work COST
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/embed_normativa_chunks.py --estimate-only
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import asyncpg
import httpx

# =============================================================================
# CONFIG
# =============================================================================

DB_URL = os.environ.get(
    "LEXE_KB_DSN",
    "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Embedding config
MODEL = "openai/text-embedding-3-small"
DIMS = 1536
CHANNEL = "testo"

# Cost estimation
EMBEDDING_PRICE_PER_1K = 0.00002  # USD per 1k tokens
CURRENCY = "USD"
COST_THRESHOLD_USD = 5.0  # Warn if cost > this per work

# Processing
DEFAULT_BATCH_SIZE = 100
STATE_FILE = Path("embed_chunks_state.json")


# =============================================================================
# STATE MANAGEMENT (for resume)
# =============================================================================

def load_state() -> dict:
    """Load resume state from file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_created_at": None, "last_id": None}


def save_state(created_at: str, chunk_id: str) -> None:
    """Save resume state to file."""
    state = {"last_created_at": created_at, "last_id": chunk_id}
    STATE_FILE.write_text(json.dumps(state))


def clear_state() -> None:
    """Clear resume state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


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

async def get_missing_chunks(
    conn: asyncpg.Connection,
    work_id: Optional[UUID] = None,
    last_created_at: Optional[str] = None,
    last_id: Optional[str] = None,
    limit: int = 100
) -> list[dict]:
    """
    Get chunks missing embeddings with cursor-based pagination.

    Uses (created_at, id) cursor for deterministic resume.
    """
    if work_id:
        if last_created_at and last_id:
            return await conn.fetch("""
                SELECT c.id, c.created_at, c.work_id, c.normativa_id, c.chunk_no, c.token_est, c.text
                FROM kb.normativa_chunk c
                WHERE c.work_id = $1
                AND NOT EXISTS (
                    SELECT 1 FROM kb.normativa_chunk_embeddings e
                    WHERE e.chunk_id = c.id
                    AND e.model = $2
                    AND e.channel = $3
                    AND e.dims = $4
                )
                AND (c.created_at, c.id) > ($5::timestamptz, $6::uuid)
                ORDER BY c.created_at, c.id
                LIMIT $7
            """, work_id, MODEL, CHANNEL, DIMS, last_created_at, UUID(last_id), limit)
        else:
            return await conn.fetch("""
                SELECT c.id, c.created_at, c.work_id, c.normativa_id, c.chunk_no, c.token_est, c.text
                FROM kb.normativa_chunk c
                WHERE c.work_id = $1
                AND NOT EXISTS (
                    SELECT 1 FROM kb.normativa_chunk_embeddings e
                    WHERE e.chunk_id = c.id
                    AND e.model = $2
                    AND e.channel = $3
                    AND e.dims = $4
                )
                ORDER BY c.created_at, c.id
                LIMIT $5
            """, work_id, MODEL, CHANNEL, DIMS, limit)
    else:
        if last_created_at and last_id:
            return await conn.fetch("""
                SELECT c.id, c.created_at, c.work_id, c.normativa_id, c.chunk_no, c.token_est, c.text
                FROM kb.normativa_chunk c
                WHERE NOT EXISTS (
                    SELECT 1 FROM kb.normativa_chunk_embeddings e
                    WHERE e.chunk_id = c.id
                    AND e.model = $1
                    AND e.channel = $2
                    AND e.dims = $3
                )
                AND (c.created_at, c.id) > ($4::timestamptz, $5::uuid)
                ORDER BY c.created_at, c.id
                LIMIT $6
            """, MODEL, CHANNEL, DIMS, last_created_at, UUID(last_id), limit)
        else:
            return await conn.fetch("""
                SELECT c.id, c.created_at, c.work_id, c.normativa_id, c.chunk_no, c.token_est, c.text
                FROM kb.normativa_chunk c
                WHERE NOT EXISTS (
                    SELECT 1 FROM kb.normativa_chunk_embeddings e
                    WHERE e.chunk_id = c.id
                    AND e.model = $1
                    AND e.channel = $2
                    AND e.dims = $3
                )
                ORDER BY c.created_at, c.id
                LIMIT $4
            """, MODEL, CHANNEL, DIMS, limit)


async def count_missing_chunks(
    conn: asyncpg.Connection,
    work_id: Optional[UUID] = None
) -> int:
    """Count chunks missing embeddings."""
    if work_id:
        return await conn.fetchval("""
            SELECT count(*)
            FROM kb.normativa_chunk c
            WHERE c.work_id = $1
            AND NOT EXISTS (
                SELECT 1 FROM kb.normativa_chunk_embeddings e
                WHERE e.chunk_id = c.id
                AND e.model = $2
                AND e.channel = $3
                AND e.dims = $4
            )
        """, work_id, MODEL, CHANNEL, DIMS)
    else:
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


async def get_cost_estimate(
    conn: asyncpg.Connection,
    work_id: Optional[UUID] = None
) -> dict:
    """Get cost estimate per work."""
    if work_id:
        row = await conn.fetchrow("""
            SELECT
                w.code as work_code,
                count(*) as total_chunks,
                round(avg(c.token_est)) as avg_token_est,
                sum(c.token_est) as total_tokens
            FROM kb.normativa_chunk c
            JOIN kb.work w ON w.id = c.work_id
            WHERE c.work_id = $1
            AND NOT EXISTS (
                SELECT 1 FROM kb.normativa_chunk_embeddings e
                WHERE e.chunk_id = c.id
                AND e.model = $2
                AND e.channel = $3
                AND e.dims = $4
            )
            GROUP BY w.code
        """, work_id, MODEL, CHANNEL, DIMS)
        if row:
            cost_usd = (row['total_tokens'] / 1000) * EMBEDDING_PRICE_PER_1K
            return {
                "work_code": row['work_code'],
                "total_chunks": row['total_chunks'],
                "avg_token_est": row['avg_token_est'],
                "total_tokens": row['total_tokens'],
                "cost_usd": round(cost_usd, 4)
            }
        return {}
    else:
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
        return {"works": estimates, "total_cost_usd": round(sum(e['cost_usd'] for e in estimates), 4)}


async def insert_embedding(
    conn: asyncpg.Connection,
    chunk_id: UUID,
    embedding: list[float]
) -> None:
    """Insert embedding for a chunk."""
    await conn.execute("""
        INSERT INTO kb.normativa_chunk_embeddings (chunk_id, model, channel, dims, embedding)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (chunk_id, model, channel, dims) DO NOTHING
    """, chunk_id, MODEL, CHANNEL, DIMS, str(embedding))


async def get_work_id(conn: asyncpg.Connection, work_code: str) -> Optional[UUID]:
    """Get work UUID by code."""
    return await conn.fetchval("SELECT id FROM kb.work WHERE code = $1", work_code)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for normativa chunks")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--work", type=str, help="Work code (e.g., COST, CC) or omit for ALL")
    parser.add_argument("--estimate-only", action="store_true", help="Only show cost estimate")
    parser.add_argument("--force", action="store_true", help="Skip cost threshold warning")
    parser.add_argument("--reset-state", action="store_true", help="Clear resume state and start fresh")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY and not args.estimate_only:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to DB...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Get work ID if specified
        work_id = None
        if args.work:
            work_id = await get_work_id(conn, args.work.upper())
            if not work_id:
                print(f"ERROR: Work '{args.work}' not found")
                sys.exit(1)
            print(f"  Processing work: {args.work.upper()}")
        else:
            print("  Processing ALL works")

        # Cost estimate
        estimate = await get_cost_estimate(conn, work_id)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === COST ESTIMATE ===")

        if args.work:
            if estimate:
                print(f"  Work: {estimate['work_code']}")
                print(f"  Chunks: {estimate['total_chunks']}")
                print(f"  Avg tokens: {estimate['avg_token_est']}")
                print(f"  Estimated cost: ${estimate['cost_usd']} {CURRENCY}")
                if estimate['cost_usd'] > COST_THRESHOLD_USD and not args.force:
                    print(f"\n  WARNING: Cost > ${COST_THRESHOLD_USD}. Use --force to proceed.")
                    if not args.estimate_only:
                        sys.exit(1)
            else:
                print("  No chunks missing embeddings!")
        else:
            if "works" in estimate:
                print(f"{'Work':<8} {'Chunks':>10} {'Avg Tok':>10} {'Cost USD':>12}")
                print("-" * 44)
                for e in estimate["works"]:
                    print(f"{e['work_code']:<8} {e['total_chunks']:>10} {e['avg_token_est']:>10} ${e['cost_usd']:>10.4f}")
                print("-" * 44)
                print(f"{'TOTAL':<8} {sum(e['total_chunks'] for e in estimate['works']):>10} {'':>10} ${estimate['total_cost_usd']:>10.4f}")

                if estimate['total_cost_usd'] > COST_THRESHOLD_USD and not args.force:
                    print(f"\n  WARNING: Total cost > ${COST_THRESHOLD_USD}. Use --force to proceed.")
                    if not args.estimate_only:
                        sys.exit(1)

        if args.estimate_only:
            await conn.close()
            return

        # Reset state if requested
        if args.reset_state:
            clear_state()
            print("\n  Resume state cleared.")

        # Load resume state
        state = load_state()
        if state["last_created_at"]:
            print(f"\n  Resuming from: {state['last_created_at']}")

        # Count missing
        missing = await count_missing_chunks(conn, work_id)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Chunks missing embeddings: {missing}")

        if missing == 0:
            print("  Nothing to do!")
            clear_state()
            await conn.close()
            return

        # Process batches
        async with httpx.AsyncClient() as client:
            processed = 0
            batch_no = 0
            last_created_at = state["last_created_at"]
            last_id = state["last_id"]

            while True:
                # Get batch
                chunks = await get_missing_chunks(
                    conn, work_id, last_created_at, last_id, args.batch_size
                )

                if not chunks:
                    break

                batch_no += 1

                # Extract texts
                texts = [row['text'][:8000] for row in chunks]  # Truncate if needed
                ids = [row['id'] for row in chunks]

                try:
                    # Get embeddings
                    embeddings = await get_embeddings(client, texts)

                    # Insert
                    for chunk_id, emb in zip(ids, embeddings):
                        await insert_embedding(conn, chunk_id, emb)

                    # Update cursor
                    last_row = chunks[-1]
                    last_created_at = str(last_row['created_at'])
                    last_id = str(last_row['id'])
                    save_state(last_created_at, last_id)

                    processed += len(chunks)
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"  [{ts}] Batch {batch_no}: {processed}/{missing} ({100*processed/missing:.1f}%)")

                except Exception as e:
                    print(f"  ERROR at batch {batch_no}: {e}")
                    await asyncio.sleep(5)
                    continue

                # Rate limiting
                await asyncio.sleep(0.5)

        # Final stats
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === FINAL STATS ===")
        total_emb = await conn.fetchval(
            "SELECT COUNT(*) FROM kb.normativa_chunk_embeddings WHERE model = $1",
            MODEL
        )
        print(f"  Total chunk embeddings: {total_emb}")

        # Clear state on success
        clear_state()

        # Save cost log
        log_file = Path(f"embed_cost_log_{datetime.now().strftime('%Y%m%d')}.csv")
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['date', 'work_code', 'chunks', 'cost_usd', 'model'])
            if args.work and estimate:
                writer.writerow([
                    datetime.now().isoformat(),
                    estimate['work_code'],
                    estimate['total_chunks'],
                    estimate['cost_usd'],
                    MODEL
                ])

    finally:
        await conn.close()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
