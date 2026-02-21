#!/usr/bin/env python3
"""
Generate embeddings for normativa chunks on STAGING.
Runs directly on the staging server (not via SSH tunnel).

Connects to:
  - lexe-max DB: localhost:5436 (lexe_kb/lexe_kb_secret)
  - lexe-litellm: localhost:4001 (model: lexe-embedding)

Usage:
    python3 /tmp/embed_staging.py --estimate-only
    python3 /tmp/embed_staging.py
    python3 /tmp/embed_staging.py --batch-size 50
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime

import asyncpg
import httpx

# ── Config ──────────────────────────────────────────────────────
DB_HOST = "localhost"
DB_PORT = 5436
DB_USER = "lexe_kb"
DB_PASS = "lexe_kb_secret"
DB_NAME = "lexe_kb"

LITELLM_URL = "http://localhost:4001/v1/embeddings"
LITELLM_KEY = "sk-lexe-litellm-stage-2026-secure"
MODEL_NAME = "lexe-embedding"       # LiteLLM alias → text-embedding-3-small
MODEL_DB = "openai/text-embedding-3-small"  # stored in DB for consistency
DIMS = 1536
CHANNEL = "testo"

DEFAULT_BATCH = 100
MAX_TEXT_LEN = 8000  # truncate to avoid token limit


# ── Database ────────────────────────────────────────────────────
async def get_pool():
    return await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        database=DB_NAME,
        min_size=2, max_size=5
    )


async def count_missing(pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval("""
            SELECT count(*)
            FROM kb.normativa_chunk c
            WHERE NOT EXISTS (
                SELECT 1 FROM kb.normativa_chunk_embeddings e
                WHERE e.chunk_id = c.id AND e.model = $1
            )
        """, MODEL_DB)


async def get_estimate(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT w.code, count(*) as chunks,
                   sum(c.token_est) as total_tokens
            FROM kb.normativa_chunk c
            JOIN kb.work w ON w.id = c.work_id
            WHERE NOT EXISTS (
                SELECT 1 FROM kb.normativa_chunk_embeddings e
                WHERE e.chunk_id = c.id AND e.model = $1
            )
            GROUP BY w.code ORDER BY w.code
        """, MODEL_DB)
    return rows


async def get_batch(pool, limit: int):
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT c.id, c.text
            FROM kb.normativa_chunk c
            WHERE NOT EXISTS (
                SELECT 1 FROM kb.normativa_chunk_embeddings e
                WHERE e.chunk_id = c.id AND e.model = $1
            )
            ORDER BY c.created_at, c.id
            LIMIT $2
        """, MODEL_DB, limit)


async def store_embeddings(pool, pairs: list[tuple]):
    """Store list of (chunk_id, embedding_vector_str) pairs."""
    async with pool.acquire() as conn:
        await conn.executemany("""
            INSERT INTO kb.normativa_chunk_embeddings
                (chunk_id, model, channel, dims, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            ON CONFLICT (chunk_id, model, channel, dims) DO NOTHING
        """, [(cid, MODEL_DB, CHANNEL, DIMS, emb) for cid, emb in pairs])


# ── Embedding API ───────────────────────────────────────────────
async def get_embeddings(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    resp = await client.post(
        LITELLM_URL,
        headers={
            "Authorization": f"Bearer {LITELLM_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL_NAME, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


# ── Main ────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    ts = lambda: datetime.now().strftime("%H:%M:%S")
    print(f"[{ts()}] Connecting to DB...")
    pool = await get_pool()

    # Estimate
    rows = await get_estimate(pool)
    total_chunks = sum(r["chunks"] for r in rows)
    total_tokens = sum(r["total_tokens"] for r in rows)
    cost = (total_tokens / 1000) * 0.00002

    print(f"\n{'Code':<8} {'Chunks':>10} {'Tokens':>12} {'Cost $':>10}")
    print("-" * 42)
    for r in rows:
        c = (r["total_tokens"] / 1000) * 0.00002
        print(f"{r['code']:<8} {r['chunks']:>10} {r['total_tokens']:>12} {c:>10.4f}")
    print("-" * 42)
    print(f"{'TOTAL':<8} {total_chunks:>10} {total_tokens:>12} {cost:>10.4f}")

    if args.estimate_only or total_chunks == 0:
        await pool.close()
        return

    print(f"\n[{ts()}] Starting embedding generation ({total_chunks} chunks)...")

    processed = 0
    errors = 0
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        while True:
            batch = await get_batch(pool, args.batch_size)
            if not batch:
                break

            texts = [row["text"][:MAX_TEXT_LEN] for row in batch]
            ids = [row["id"] for row in batch]

            try:
                embeddings = await get_embeddings(client, texts)
                pairs = [
                    (cid, str(emb))
                    for cid, emb in zip(ids, embeddings)
                ]
                await store_embeddings(pool, pairs)
                processed += len(batch)

                elapsed = time.monotonic() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_chunks - processed) / rate if rate > 0 else 0
                print(f"  [{ts()}] {processed}/{total_chunks} "
                      f"({100*processed/total_chunks:.1f}%) "
                      f"rate={rate:.0f}/s ETA={eta:.0f}s")

            except Exception as e:
                errors += 1
                print(f"  [{ts()}] ERROR batch: {e}")
                if errors > 10:
                    print("Too many errors, stopping.")
                    break
                await asyncio.sleep(5)
                continue

            await asyncio.sleep(0.3)  # rate limit courtesy

    # Final stats
    async with pool.acquire() as conn:
        total_emb = await conn.fetchval(
            "SELECT count(*) FROM kb.normativa_chunk_embeddings WHERE model = $1",
            MODEL_DB
        )
        stats = await conn.fetch("SELECT * FROM kb.v_chunk_stats ORDER BY work_code")

    elapsed = time.monotonic() - t0
    print(f"\n[{ts()}] Done in {elapsed:.0f}s")
    print(f"Processed: {processed}, Errors: {errors}")
    print(f"Total embeddings in DB: {total_emb}")
    print(f"\n{'Code':<8} {'Articles':>10} {'Chunks':>10} {'ChunkCov%':>10} {'EmbCov%':>10}")
    for s in stats:
        print(f"{s['work_code']:<8} {s['articoli_tot']:>10} "
              f"{s['articoli_chunkizzati']:>10} "
              f"{s['chunk_coverage_pct']:>9.1f}% "
              f"{s['emb_coverage_pct']:>9.1f}%")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
