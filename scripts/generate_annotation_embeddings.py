#!/usr/bin/env python3
"""Generate embeddings for kb.annotation (Brocardi notes)."""

import asyncio
import os
import sys
import re
import httpx
import asyncpg
from datetime import datetime

# Config
DB_URL = "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "openai/text-embedding-3-small"
DIMS = 1536
CHANNEL = "content"
BATCH_SIZE = 50
MIN_LEN = 50


def normalize_text(title: str | None, content: str) -> str:
    """Normalize annotation text for embedding."""
    t = (title or "").strip()
    c = (content or "").strip()

    # Collapse multiple spaces/newlines
    c = re.sub(r'\s+', ' ', c)

    if t:
        return f"{t}\n\n{c}"
    return c


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


async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to DB...")
    conn = await asyncpg.connect(DB_URL)

    # Get annotations missing embeddings (with quality filter)
    query = """
        SELECT a.id, a.type::text, a.title, a.content
        FROM kb.annotation a
        WHERE a.content IS NOT NULL
        AND TRIM(a.content) != ''
        AND LENGTH(TRIM(a.content)) >= $1
        AND NOT EXISTS (
            SELECT 1 FROM kb.annotation_embeddings e
            WHERE e.annotation_id = a.id
            AND e.model = $2
            AND e.channel = $3
        )
        ORDER BY a.created_at, a.id
    """
    rows = await conn.fetch(query, MIN_LEN, MODEL, CHANNEL)
    total = len(rows)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {total} annotations to embed (min_len={MIN_LEN})")

    if total == 0:
        print("  Nothing to do!")
        await conn.close()
        return

    async with httpx.AsyncClient() as client:
        processed = 0
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i:i+BATCH_SIZE]

            # Prepare texts
            texts = []
            ids = []
            for row in batch:
                text = normalize_text(row["title"], row["content"])
                # Truncate very long texts
                texts.append(text[:8000] if len(text) > 8000 else text)
                ids.append(row["id"])

            try:
                embeddings = await get_embeddings(client, texts)

                # Insert into DB
                for ann_id, emb in zip(ids, embeddings):
                    await conn.execute("""
                        INSERT INTO kb.annotation_embeddings
                        (annotation_id, model, channel, dims, embedding)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (annotation_id, model, channel, dims) DO NOTHING
                    """, ann_id, MODEL, CHANNEL, DIMS, str(emb))

                processed += len(batch)
                ann_type = batch[0]["type"] if batch else "?"
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {processed}/{total} ({ann_type})")

            except Exception as e:
                print(f"  ERROR at batch {i}: {e}")
                await asyncio.sleep(5)
                continue

            # Rate limiting
            await asyncio.sleep(0.5)

    # Final stats
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === FINAL STATS ===")
    total_ann_emb = await conn.fetchval("SELECT COUNT(*) FROM kb.annotation_embeddings")
    total_norm_emb = await conn.fetchval("SELECT COUNT(*) FROM kb.normativa_embeddings")

    print(f"  Annotation embeddings: {total_ann_emb}")
    print(f"  Normativa embeddings: {total_norm_emb}")
    print(f"  TOTAL embeddings: {total_ann_emb + total_norm_emb}")

    await conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
