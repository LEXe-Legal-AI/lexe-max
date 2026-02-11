#!/usr/bin/env python3
"""Generate rubrica embeddings ONLY for articles with valid rubrica."""

import asyncio
import os
import sys
import httpx
import asyncpg
from datetime import datetime

# Config
DB_URL = "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "openai/text-embedding-3-small"
DIMS = 1536
BATCH_SIZE = 50


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
    
    # Get articles with valid rubrica that don't have embeddings yet
    query = """
        SELECT n.id, n.rubrica, w.code
        FROM kb.normativa n
        JOIN kb.work w ON w.id = n.work_id
        WHERE n.rubrica IS NOT NULL 
        AND TRIM(n.rubrica) != ''
        AND NOT EXISTS (
            SELECT 1 FROM kb.normativa_embeddings e 
            WHERE e.normativa_id = n.id 
            AND e.channel = 'rubrica' 
            AND e.model = $1
        )
        ORDER BY w.code, n.articolo_sort_key
    """
    rows = await conn.fetch(query, MODEL)
    total = len(rows)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {total} articles with valid rubrica missing embedding")
    
    if total == 0:
        print("  Nothing to do!")
        await conn.close()
        return
    
    async with httpx.AsyncClient() as client:
        processed = 0
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i:i+BATCH_SIZE]
            texts = [row["rubrica"][:8000] for row in batch]
            
            try:
                embeddings = await get_embeddings(client, texts)
                
                for row, emb in zip(batch, embeddings):
                    await conn.execute("""
                        INSERT INTO kb.normativa_embeddings (normativa_id, model, channel, dims, embedding)
                        VALUES ($1, $2, 'rubrica', $3, $4)
                        ON CONFLICT (normativa_id, model, channel, dims) DO NOTHING
                    """, row["id"], MODEL, DIMS, str(emb))
                
                processed += len(batch)
                current_code = batch[0]["code"] if batch else "?"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] rubrica: {processed}/{total} ({current_code})")
                
            except Exception as e:
                print(f"ERROR at batch {i}: {e}")
                await asyncio.sleep(5)
                continue
            
            await asyncio.sleep(0.5)
    
    # Final stats
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === FINAL STATS ===")
    stats = await conn.fetch("""
        SELECT channel, COUNT(*) as cnt 
        FROM kb.normativa_embeddings 
        GROUP BY channel
    """)
    for row in stats:
        print(f"  {row['channel']}: {row['cnt']}")
    
    total_emb = await conn.fetchval("SELECT COUNT(*) FROM kb.normativa_embeddings")
    print(f"  Total embeddings: {total_emb}")
    
    await conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
