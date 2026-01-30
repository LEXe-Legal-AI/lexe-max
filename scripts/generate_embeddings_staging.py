"""
KB Massimari - Generate Mistral Embeddings on Staging
Uses OpenRouter API to generate embeddings for all massime.

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export PATH=$HOME/.local/bin:$PATH
    export OPENROUTER_API_KEY='sk-or-v1-...'
    uv run python scripts/generate_embeddings_staging.py
"""
import asyncio
import os
import time
from uuid import UUID

import asyncpg
import httpx

# Config
DB_URL = "postgresql://leo:stage_postgres_2026_secure@localhost:5432/leo"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_MODEL = "mistralai/mistral-embed-2312"
BATCH_SIZE = 20
MAX_TEXT_LENGTH = 8000


def embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def get_embeddings(client: httpx.AsyncClient, texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
    """Get embeddings from OpenRouter API."""
    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MISTRAL_MODEL, "input": texts},
            timeout=120.0,
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings, None

    except Exception as e:
        return None, str(e)


async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return

    print("=" * 70)
    print("KB MASSIMARI - GENERATE MISTRAL EMBEDDINGS (STAGING)")
    print("=" * 70)
    print(f"Model: {MISTRAL_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Get massime without embeddings
    rows = await conn.fetch("""
        SELECT m.id, m.testo
        FROM kb.massime m
        LEFT JOIN kb.emb_mistral e ON e.massima_id = m.id
        WHERE e.id IS NULL
        ORDER BY m.id
    """)

    print(f"[OK] Found {len(rows)} massime without embeddings")

    if not rows:
        print("\n[DONE] All massime already have embeddings!")
        await conn.close()
        return

    # Process in batches
    processed = 0
    errors = 0
    total_latency = 0

    async with httpx.AsyncClient() as client:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            texts = [row["testo"][:MAX_TEXT_LENGTH] for row in batch]
            ids = [row["id"] for row in batch]

            start = time.time()
            embeddings, error = await get_embeddings(client, texts)
            latency = (time.time() - start) * 1000
            total_latency += latency

            if error:
                print(f"  [ERROR] Batch {i // BATCH_SIZE + 1}: {error}")
                errors += len(batch)
                continue

            # Insert embeddings
            for mid, emb in zip(ids, embeddings):
                emb_str = embedding_to_pgvector(emb)
                await conn.execute("""
                    INSERT INTO kb.emb_mistral (massima_id, chunk_idx, embedding)
                    VALUES ($1, 0, $2::vector)
                    ON CONFLICT (massima_id, chunk_idx) DO UPDATE SET embedding = EXCLUDED.embedding
                """, mid, emb_str)

            processed += len(batch)
            progress = (i + len(batch)) / len(rows) * 100
            print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} embeddings, {latency:.0f}ms, {progress:.1f}%")

            # Rate limit
            await asyncio.sleep(0.5)

    # Final stats
    emb_count = await conn.fetchval("SELECT COUNT(*) FROM kb.emb_mistral")
    await conn.close()

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total embeddings in DB: {emb_count}")
    print(f"Avg latency/batch: {total_latency / max(1, processed // BATCH_SIZE):.0f}ms")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
