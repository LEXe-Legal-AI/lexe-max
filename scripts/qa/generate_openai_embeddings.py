"""
Generate OpenAI Embeddings for KB Massime

Model: text-embedding-3-small
Dimension: 1536 (native)
Normalization: Already unit-length (OpenAI guarantees this)
Metric: cosine (equivalent to IP for unit vectors)

Key points:
- OpenAI returns unit-length vectors, NO renormalization needed
- cosine = dot product = same ranking for unit vectors
- Embeddings are NOT node IDs - use (doc_id, citation_key, content_hash) for graph

Usage:
    uv run python scripts/qa/generate_openai_embeddings.py --dry-run
    uv run python scripts/qa/generate_openai_embeddings.py --commit --batch-size 100
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass

import asyncpg
import httpx

# ============================================================
# Configuration
# ============================================================

DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
# Use OpenRouter as proxy to OpenAI
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model config - LOCKED for this batch
MODEL_ID = "openai/text-embedding-3-small"  # OpenRouter format
MODEL_VERSION = "v1"
EXPECTED_DIM = 1536
EMBEDDING_DISTANCE = "cosine"
EMBEDDING_BATCH_ID = 1  # Increment if changing model/dim/distance

# Processing config
API_BATCH_SIZE = 100  # OpenAI supports up to 2048 texts per call
MAX_TEXT_LENGTH = 8000  # text-embedding-3-small has 8191 token limit
RATE_LIMIT_DELAY = 0.2  # Seconds between API calls (stay under 3000 RPM)


@dataclass
class EmbeddingStats:
    total_processed: int = 0
    total_inserted: int = 0
    api_calls: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_time_s: float = 0


# ============================================================
# OpenAI API
# ============================================================

async def get_embeddings_batch(
    client: httpx.AsyncClient,
    texts: list[str],
    stats: EmbeddingStats,
) -> tuple[list[list[float]] | None, str | None]:
    """
    Get embeddings from OpenAI API.

    Note: OpenAI returns already normalized (unit-length) vectors.
    No need to renormalize - just use them directly.
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY not set"

    # Truncate texts (OpenAI has 8191 token limit for this model)
    truncated = [t[:MAX_TEXT_LENGTH] for t in texts]

    try:
        response = await client.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_ID,
                "input": truncated,
                "encoding_format": "float",  # Default, explicit for clarity
            },
            timeout=120.0,
        )

        stats.api_calls += 1

        if response.status_code == 429:
            return None, "Rate limited - waiting..."

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        # Track token usage
        if "usage" in data:
            stats.total_tokens += data["usage"].get("total_tokens", 0)

        # Extract embeddings (already sorted by index)
        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return embeddings, None

    except httpx.TimeoutException:
        return None, "Request timeout (120s)"
    except Exception as e:
        return None, str(e)


# ============================================================
# Database Operations
# ============================================================

async def get_massime_without_embeddings(
    conn: asyncpg.Connection,
    limit: int,
) -> list[dict]:
    """
    Fetch active massime that don't have embeddings yet.
    Prioritizes newer batches (citation-anchored extraction).
    """
    rows = await conn.fetch("""
        SELECT m.id, m.testo, m.ingest_batch_id
        FROM kb.massime m
        LEFT JOIN kb.embeddings e ON e.massima_id = m.id
            AND e.model_name = $1
        WHERE m.is_active = TRUE
        AND e.id IS NULL
        AND m.testo IS NOT NULL
        AND LENGTH(m.testo) > 50
        ORDER BY m.ingest_batch_id DESC NULLS LAST, m.id
        LIMIT $2
    """, MODEL_ID, limit)

    return [dict(row) for row in rows]


async def insert_embeddings_batch(
    conn: asyncpg.Connection,
    records: list[tuple],
) -> int:
    """
    Insert embeddings batch into database.

    Args:
        records: List of (massima_id, embedding)
    """
    insert_data = []
    for massima_id, embedding in records:
        # Build vector string format for pgvector
        vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
        insert_data.append((
            massima_id,
            MODEL_ID,
            MODEL_VERSION,
            EXPECTED_DIM,
            EMBEDDING_DISTANCE,
            EMBEDDING_BATCH_ID,
            vec_str,
            True,  # is_normalized (OpenAI guarantees this)
        ))

    await conn.executemany("""
        INSERT INTO kb.embeddings
        (massima_id, model_name, model_version, dimension,
         embedding_distance, embedding_batch_id, embedding, is_normalized)
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
        ON CONFLICT (massima_id, model_name) DO NOTHING
    """, insert_data)

    return len(insert_data)


async def get_embedding_progress(conn: asyncpg.Connection) -> dict:
    """Get current embedding progress."""
    row = await conn.fetchrow("""
        SELECT
            (SELECT COUNT(*) FROM kb.massime WHERE is_active = TRUE) as total_active,
            (SELECT COUNT(*) FROM kb.embeddings WHERE model_name = $1) as total_embedded
    """, MODEL_ID)

    return {
        "total_active": row["total_active"],
        "total_embedded": row["total_embedded"],
        "remaining": row["total_active"] - row["total_embedded"],
    }


# ============================================================
# Main Processing
# ============================================================

async def process_embeddings(
    batch_size: int,
    dry_run: bool,
    max_batches: int | None = None,
):
    """Main embedding generation loop."""
    print("=" * 70)
    print("OPENAI EMBEDDING GENERATION")
    print("=" * 70)
    print(f"Model:           {MODEL_ID}")
    print(f"Dimension:       {EXPECTED_DIM}")
    print(f"Distance:        {EMBEDDING_DISTANCE}")
    print(f"Batch ID:        {EMBEDDING_BATCH_ID}")
    print(f"API batch size:  {batch_size}")
    print(f"Mode:            {'DRY RUN' if dry_run else 'COMMIT'}")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("\n[ERROR] OPENROUTER_API_KEY not set!")
        print("  export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Check progress
    progress = await get_embedding_progress(conn)
    print(f"\nProgress:")
    print(f"  Total active massime: {progress['total_active']:,}")
    print(f"  Already embedded:     {progress['total_embedded']:,}")
    print(f"  Remaining:            {progress['remaining']:,}")

    if progress["remaining"] == 0:
        print("\n[OK] All active massime already have embeddings!")
        await conn.close()
        return

    # Estimate cost
    # text-embedding-3-small: $0.02 per 1M tokens
    # Avg massima ~500 tokens
    est_tokens = progress["remaining"] * 500
    est_cost = est_tokens / 1_000_000 * 0.02
    print(f"\nEstimated cost: ~${est_cost:.2f} ({est_tokens:,} tokens @ $0.02/1M)")

    # Initialize stats
    stats = EmbeddingStats()
    start_time = time.time()
    batch_num = 0

    async with httpx.AsyncClient() as client:
        while True:
            batch_num += 1

            if max_batches and batch_num > max_batches:
                print(f"\n[STOP] Reached max batches ({max_batches})")
                break

            # Fetch next batch
            massime = await get_massime_without_embeddings(conn, batch_size)

            if not massime:
                print("\n[OK] No more massime to process!")
                break

            print(f"\n--- Batch {batch_num} ({len(massime)} massime) ---")

            # Get embeddings from API
            texts = [m["testo"] for m in massime]
            embeddings, error = await get_embeddings_batch(client, texts, stats)

            if error:
                print(f"  [ERROR] API call failed: {error}")
                stats.errors += 1
                if "Rate limited" in error:
                    await asyncio.sleep(10)  # Back off on rate limit
                else:
                    await asyncio.sleep(2)
                continue

            # Validate response
            if len(embeddings) != len(massime):
                print(f"  [ERROR] Response count mismatch: {len(embeddings)} vs {len(massime)}")
                stats.errors += 1
                continue

            # Validate dimension
            actual_dim = len(embeddings[0])
            if actual_dim != EXPECTED_DIM:
                print(f"  [CRITICAL] Dimension mismatch: expected {EXPECTED_DIM}, got {actual_dim}")
                await conn.close()
                sys.exit(1)

            # Prepare records (no renormalization - OpenAI already unit-length)
            records = [(m["id"], emb) for m, emb in zip(massime, embeddings)]
            stats.total_processed += len(massime)

            # Insert to database
            if not dry_run:
                inserted = await insert_embeddings_batch(conn, records)
                stats.total_inserted += inserted
                print(f"  [OK] Inserted {inserted} embeddings")
            else:
                print(f"  [DRY RUN] Would insert {len(records)} embeddings")

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

    # Final stats
    elapsed = time.time() - start_time
    stats.total_time_s = elapsed

    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    print(f"Total processed:      {stats.total_processed:,}")
    print(f"Total inserted:       {stats.total_inserted:,}")
    print(f"API calls:            {stats.api_calls}")
    print(f"Total tokens:         {stats.total_tokens:,}")
    print(f"Errors:               {stats.errors}")
    print(f"Total time:           {elapsed:.1f}s")
    if elapsed > 0:
        print(f"Rate:                 {stats.total_processed / elapsed:.1f} massime/sec")

    # Cost calculation
    actual_cost = stats.total_tokens / 1_000_000 * 0.02
    print(f"Actual cost:          ${actual_cost:.4f}")

    # Updated progress
    progress = await get_embedding_progress(conn)
    pct = 100 * progress["total_embedded"] / max(1, progress["total_active"])
    print(f"\nProgress: {progress['total_embedded']:,}/{progress['total_active']:,} ({pct:.1f}%)")

    await conn.close()
    print("\n[DONE]")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenAI embeddings for KB massime"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of massime per API call (default: 100, max: 2048)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't insert to database, just test API"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually insert embeddings to database"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.commit:
        print("ERROR: Must specify --dry-run or --commit")
        sys.exit(1)

    if args.batch_size > 2048:
        print("ERROR: batch-size cannot exceed 2048 (OpenAI limit)")
        sys.exit(1)

    asyncio.run(process_embeddings(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        max_batches=args.max_batches,
    ))


if __name__ == "__main__":
    main()
