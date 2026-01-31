"""
Generate Gemini Embeddings for KB Massime

Model: google/gemini-embedding-001
Dimension: 1536 (enforced)
Normalization: ON (unit length)
Metric: inner product (treat as cosine)

Mini-checklist operativa:
1. Enforce dimension: assert len(vec) == 1536
2. Enforce norm: if outside 0.99-1.01, renormalize and log
3. Don't mix batches with different metrics
4. Run retrieval eval before promoting

Usage:
    uv run python scripts/qa/generate_gemini_embeddings.py --batch-size 50 --dry-run
    uv run python scripts/qa/generate_gemini_embeddings.py --batch-size 50 --commit
"""

import argparse
import asyncio
import math
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
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model config - LOCKED for this batch
MODEL_ID = "google/gemini-embedding-001"
MODEL_VERSION = "v1"
EXPECTED_DIM = 1536
EMBEDDING_BATCH_ID = 1  # Increment if changing model/dim/normalization

# Processing config
API_BATCH_SIZE = 50  # Texts per API call (OpenRouter limit)
MAX_TEXT_LENGTH = 8000  # Truncate longer texts
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls

# Normalization thresholds
NORM_MIN = 0.99
NORM_MAX = 1.01


@dataclass
class EmbeddingStats:
    total_processed: int = 0
    total_inserted: int = 0
    norm_renormalized: int = 0
    errors: int = 0
    total_time_ms: float = 0


# ============================================================
# Vector Operations
# ============================================================

def compute_norm(vec: list[float]) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def normalize_vector(vec: list[float]) -> list[float]:
    """Normalize vector to unit length."""
    norm = compute_norm(vec)
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def validate_and_normalize(
    vec: list[float],
    expected_dim: int,
    stats: EmbeddingStats,
) -> tuple[list[float], float, bool]:
    """
    Validate dimension and normalize vector.

    Returns:
        (normalized_vec, original_norm, was_renormalized)
    """
    # Enforce dimension
    if len(vec) != expected_dim:
        raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {len(vec)}")

    # Compute original norm
    original_norm = compute_norm(vec)

    # Check if renormalization needed
    was_renormalized = False
    if original_norm < NORM_MIN or original_norm > NORM_MAX:
        vec = normalize_vector(vec)
        was_renormalized = True
        stats.norm_renormalized += 1

    return vec, original_norm, was_renormalized


# ============================================================
# OpenRouter API
# ============================================================

async def get_embeddings_batch(
    client: httpx.AsyncClient,
    texts: list[str],
) -> tuple[list[list[float]] | None, str | None]:
    """
    Get embeddings from OpenRouter API.

    Returns:
        (embeddings, error_message)
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY not set"

    # Truncate texts
    truncated = [t[:MAX_TEXT_LENGTH] for t in texts]

    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_ID,
                "input": truncated,
            },
            timeout=120.0,
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
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
        ORDER BY m.ingest_batch_id NULLS LAST, m.id
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
        records: List of (massima_id, embedding, norm_original)

    Returns:
        Number of inserted rows
    """
    # Build vector string format for pgvector
    insert_data = []
    for massima_id, embedding, norm_original in records:
        vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
        insert_data.append((
            massima_id,
            MODEL_ID,
            MODEL_VERSION,
            EXPECTED_DIM,
            vec_str,
            norm_original,
            True,  # is_normalized
            EMBEDDING_BATCH_ID,
        ))

    result = await conn.executemany("""
        INSERT INTO kb.embeddings
        (massima_id, model_name, model_version, dimension, embedding,
         norm_original, is_normalized, ingest_batch_id)
        VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8)
        ON CONFLICT (massima_id, model_name) DO NOTHING
    """, insert_data)

    # Count inserted (executemany doesn't return count directly)
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
    """
    Main embedding generation loop.
    """
    print("=" * 70)
    print("GEMINI EMBEDDING GENERATION")
    print("=" * 70)
    print(f"Model:       {MODEL_ID}")
    print(f"Dimension:   {EXPECTED_DIM}")
    print(f"Batch ID:    {EMBEDDING_BATCH_ID}")
    print(f"Batch size:  {batch_size}")
    print(f"Mode:        {'DRY RUN' if dry_run else 'COMMIT'}")
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

            embeddings, error = await get_embeddings_batch(client, texts)

            if error:
                print(f"  [ERROR] API call failed: {error}")
                stats.errors += 1
                await asyncio.sleep(5)  # Back off on error
                continue

            # Validate dimensions
            if len(embeddings) != len(massime):
                print(f"  [ERROR] Response count mismatch: {len(embeddings)} vs {len(massime)}")
                stats.errors += 1
                continue

            # Validate dimension of first embedding
            actual_dim = len(embeddings[0])
            if actual_dim != EXPECTED_DIM:
                print(f"  [ERROR] Dimension mismatch: expected {EXPECTED_DIM}, got {actual_dim}")
                print(f"  [CRITICAL] Update EXPECTED_DIM or use different model!")
                await conn.close()
                sys.exit(1)

            # Validate and normalize all embeddings
            records = []
            for i, (massima, embedding) in enumerate(zip(massime, embeddings)):
                try:
                    norm_vec, orig_norm, was_renorm = validate_and_normalize(
                        embedding, EXPECTED_DIM, stats
                    )
                    records.append((massima["id"], norm_vec, orig_norm))
                except ValueError as e:
                    print(f"  [ERROR] Massima {massima['id']}: {e}")
                    stats.errors += 1

            stats.total_processed += len(massime)

            # Insert to database
            if not dry_run and records:
                inserted = await insert_embeddings_batch(conn, records)
                stats.total_inserted += inserted
                print(f"  [OK] Inserted {inserted} embeddings (renorm: {stats.norm_renormalized})")
            else:
                print(f"  [DRY RUN] Would insert {len(records)} embeddings")

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

    # Final stats
    elapsed = time.time() - start_time
    stats.total_time_ms = elapsed * 1000

    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    print(f"Total processed:      {stats.total_processed:,}")
    print(f"Total inserted:       {stats.total_inserted:,}")
    print(f"Renormalized:         {stats.norm_renormalized:,}")
    print(f"Errors:               {stats.errors}")
    print(f"Total time:           {elapsed:.1f}s")
    print(f"Rate:                 {stats.total_processed / elapsed:.1f} massime/sec")

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
        description="Generate Gemini embeddings for KB massime"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of massime per API call (default: 50)"
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
        help="Don't insert to database, just test"
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

    asyncio.run(process_embeddings(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        max_batches=args.max_batches,
    ))


if __name__ == "__main__":
    main()
