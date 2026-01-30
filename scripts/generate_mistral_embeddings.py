"""
Generate Mistral Embeddings for all KB Massime via OpenRouter.
Stores embeddings in kb.emb_mistral table.

Usage:
    export OPENROUTER_API_KEY='sk-or-...'
    python scripts/generate_mistral_embeddings.py

    # Resume from partial run
    python scripts/generate_mistral_embeddings.py --skip-existing

    # Verify only
    python scripts/generate_mistral_embeddings.py --verify-only
"""
import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from uuid import UUID

import asyncpg
import httpx

# Config
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Mistral Embed config - model ID from OpenRouter benchmark
# Winner: mistralai/mistral-embed (older 2312 version had 0.793 avg score)
# Current available: mistralai/mistral-embed or codestral-embed-2505
MISTRAL_MODEL = "mistralai/mistral-embed-2312"
MISTRAL_DIM = 1024
BATCH_SIZE = 20  # OpenRouter rate limit friendly
MAX_TEXT_LENGTH = 8000  # Mistral supports 8k tokens, ~32k chars


@dataclass
class Massima:
    id: UUID
    testo: str
    anno: int
    tipo: str


@dataclass
class EmbeddingStats:
    total: int
    processed: int
    skipped: int
    errors: int
    avg_latency_ms: float


async def get_embedding_openrouter(
    client: httpx.AsyncClient,
    texts: list[str],
) -> tuple[list[list[float]] | None, float, str | None]:
    """Get embeddings from OpenRouter API."""
    start = time.time()

    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "input": texts,
            },
            timeout=120.0,  # Generous timeout for batch
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            return None, latency, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings, latency, None

    except Exception as e:
        latency = (time.time() - start) * 1000
        return None, latency, str(e)


async def load_all_massime(conn: asyncpg.Connection) -> list[Massima]:
    """Load all massime from database."""
    rows = await conn.fetch("""
        SELECT m.id, m.testo, d.anno, d.tipo
        FROM kb.massime m
        JOIN kb.documents d ON d.id = m.document_id
        ORDER BY d.anno, m.id
    """)

    return [
        Massima(
            id=row["id"],
            testo=row["testo"],
            anno=row["anno"],
            tipo=row["tipo"],
        )
        for row in rows
    ]


async def load_existing_ids(conn: asyncpg.Connection) -> set[UUID]:
    """Load IDs of massime that already have embeddings."""
    rows = await conn.fetch("""
        SELECT DISTINCT massima_id FROM kb.emb_mistral
    """)
    return {row["massima_id"] for row in rows}


def embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def insert_embeddings(
    conn: asyncpg.Connection,
    massima_ids: list[UUID],
    embeddings: list[list[float]],
):
    """Insert embeddings into database."""
    values = [
        (mid, 0, embedding_to_pgvector(emb))  # chunk_idx = 0 for full massima
        for mid, emb in zip(massima_ids, embeddings)
    ]

    await conn.executemany(
        """
        INSERT INTO kb.emb_mistral (massima_id, chunk_idx, embedding)
        VALUES ($1, $2, $3::vector)
        ON CONFLICT (massima_id, chunk_idx) DO UPDATE
        SET embedding = EXCLUDED.embedding
        """,
        values,
    )


async def generate_embeddings(skip_existing: bool = False):
    """Generate Mistral embeddings for all massime."""
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  export OPENROUTER_API_KEY='sk-or-...'")
        return

    print("=" * 70)
    print("GENERATE MISTRAL EMBEDDINGS - KB MASSIMARI")
    print("=" * 70)
    print(f"Model: {MISTRAL_MODEL}")
    print(f"Dimension: {MISTRAL_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Load all massime
    massime = await load_all_massime(conn)
    print(f"[OK] Loaded {len(massime)} massime")

    # Check existing if skip mode
    existing_ids = set()
    if skip_existing:
        existing_ids = await load_existing_ids(conn)
        print(f"[OK] Found {len(existing_ids)} existing embeddings")

    # Filter massime to process
    to_process = [m for m in massime if m.id not in existing_ids]
    print(f"[OK] Will process {len(to_process)} massime")

    if not to_process:
        print("\n[DONE] All massime already have embeddings!")
        await conn.close()
        return

    # Process in batches
    stats = EmbeddingStats(
        total=len(to_process),
        processed=0,
        skipped=0,
        errors=0,
        avg_latency_ms=0,
    )
    total_latency = 0

    async with httpx.AsyncClient() as client:
        for batch_idx in range(0, len(to_process), BATCH_SIZE):
            batch = to_process[batch_idx:batch_idx + BATCH_SIZE]

            # Prepare texts (truncate if needed)
            texts = [m.testo[:MAX_TEXT_LENGTH] for m in batch]
            massima_ids = [m.id for m in batch]

            # Get embeddings
            embeddings, latency, error = await get_embedding_openrouter(client, texts)
            total_latency += latency

            if error:
                print(f"  [ERROR] Batch {batch_idx // BATCH_SIZE + 1}: {error}")
                stats.errors += len(batch)
                continue

            # Verify dimensions
            if embeddings and len(embeddings[0]) != MISTRAL_DIM:
                print(f"  [WARN] Unexpected dimension: {len(embeddings[0])} (expected {MISTRAL_DIM})")

            # Insert into DB
            await insert_embeddings(conn, massima_ids, embeddings)
            stats.processed += len(batch)

            # Progress
            progress = (batch_idx + len(batch)) / len(to_process) * 100
            print(f"  Batch {batch_idx // BATCH_SIZE + 1}: "
                  f"{len(batch)} embeddings, {latency:.0f}ms, "
                  f"{progress:.1f}% complete")

            # Rate limit: 0.5s between batches
            await asyncio.sleep(0.5)

    # Calculate stats
    if stats.processed > 0:
        stats.avg_latency_ms = total_latency / (stats.processed / BATCH_SIZE)

    # Final stats
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total massime:     {stats.total}")
    print(f"Processed:         {stats.processed}")
    print(f"Errors:            {stats.errors}")
    print(f"Avg latency/batch: {stats.avg_latency_ms:.0f}ms")

    # Verify in DB
    count = await conn.fetchval("SELECT COUNT(*) FROM kb.emb_mistral")
    print(f"\n[DB] Total embeddings in kb.emb_mistral: {count}")

    await conn.close()
    print("\n[DONE]")


async def verify_embeddings():
    """Verify embeddings in database."""
    print("=" * 70)
    print("VERIFY MISTRAL EMBEDDINGS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)

    # Count massime
    total_massime = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")
    print(f"Total massime: {total_massime}")

    # Count embeddings
    total_emb = await conn.fetchval("SELECT COUNT(*) FROM kb.emb_mistral")
    print(f"Total embeddings: {total_emb}")

    # Count unique massime with embeddings
    unique_massime = await conn.fetchval(
        "SELECT COUNT(DISTINCT massima_id) FROM kb.emb_mistral"
    )
    print(f"Unique massime with embeddings: {unique_massime}")

    # Coverage
    coverage = unique_massime / total_massime * 100 if total_massime > 0 else 0
    print(f"Coverage: {coverage:.1f}%")

    # Dimension check
    sample = await conn.fetchrow("""
        SELECT massima_id, array_length(embedding::real[], 1) as dim
        FROM kb.emb_mistral
        LIMIT 1
    """)
    if sample:
        print(f"Embedding dimension: {sample['dim']}")

    # Check for nulls
    nulls = await conn.fetchval(
        "SELECT COUNT(*) FROM kb.emb_mistral WHERE embedding IS NULL"
    )
    print(f"Null embeddings: {nulls}")

    # Distribution by document type
    print("\nBy document type:")
    rows = await conn.fetch("""
        SELECT d.tipo, d.anno, COUNT(*) as count
        FROM kb.emb_mistral e
        JOIN kb.massime m ON m.id = e.massima_id
        JOIN kb.documents d ON d.id = m.document_id
        GROUP BY d.tipo, d.anno
        ORDER BY d.tipo, d.anno
    """)
    for row in rows:
        print(f"  {row['tipo']} {row['anno']}: {row['count']} embeddings")

    # Quick similarity test
    print("\nSimilarity test (top 3 for 'responsabilita medica'):")
    test_embedding = await get_test_embedding("responsabilita medica per colpa grave")
    if test_embedding:
        results = await conn.fetch("""
            SELECT m.id, LEFT(m.testo, 100) as preview,
                   1 - (e.embedding <=> $1::vector) as similarity
            FROM kb.emb_mistral e
            JOIN kb.massime m ON m.id = e.massima_id
            ORDER BY e.embedding <=> $1::vector
            LIMIT 3
        """, test_embedding)

        for i, row in enumerate(results, 1):
            print(f"  {i}. [{row['similarity']:.4f}] {row['preview']}...")

    await conn.close()
    print("\n[DONE]")


async def get_test_embedding(text: str) -> list[float] | None:
    """Get embedding for a test query."""
    if not OPENROUTER_API_KEY:
        return None

    async with httpx.AsyncClient() as client:
        embeddings, _, error = await get_embedding_openrouter(client, [text])
        if error:
            print(f"  [WARN] Test embedding failed: {error}")
            return None
        return embeddings[0] if embeddings else None


async def main():
    parser = argparse.ArgumentParser(description="Generate Mistral embeddings for KB")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip massime that already have embeddings",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing embeddings",
    )

    args = parser.parse_args()

    if args.verify_only:
        await verify_embeddings()
    else:
        await generate_embeddings(skip_existing=args.skip_existing)


if __name__ == "__main__":
    asyncio.run(main())
