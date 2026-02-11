#!/usr/bin/env python3
"""Continue generating embeddings for kb.normativa (resumable)."""

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
CHANNELS = ["testo", "rubrica"]


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
    
    # Find articles without embeddings for each channel
    for channel in CHANNELS:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing channel: {channel}")
        
        # Get articles missing this channel's embedding
        # For rubrica, exclude NULL/empty values
        if channel == "rubrica":
            query = """
                SELECT n.id, n.testo, n.rubrica, w.code
                FROM kb.normativa n
                JOIN kb.work w ON w.id = n.work_id
                WHERE n.rubrica IS NOT NULL
                AND TRIM(n.rubrica) != ''
                AND NOT EXISTS (
                    SELECT 1 FROM kb.normativa_embeddings e
                    WHERE e.normativa_id = n.id
                    AND e.channel = $1
                    AND e.model = $2
                )
                ORDER BY w.code, n.articolo_sort_key
            """
        else:
            query = """
                SELECT n.id, n.testo, n.rubrica, w.code
                FROM kb.normativa n
                JOIN kb.work w ON w.id = n.work_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM kb.normativa_embeddings e
                    WHERE e.normativa_id = n.id
                    AND e.channel = $1
                    AND e.model = $2
                )
                ORDER BY w.code, n.articolo_sort_key
            """
        rows = await conn.fetch(query, channel, MODEL)
        total = len(rows)
        
        if total == 0:
            print(f"  All {channel} embeddings already exist!")
            continue
        
        print(f"  Found {total} articles missing {channel} embeddings")
        
        async with httpx.AsyncClient() as client:
            processed = 0
            for i in range(0, total, BATCH_SIZE):
                batch = rows[i:i+BATCH_SIZE]
                
                # Prepare texts
                texts = []
                for row in batch:
                    if channel == "testo":
                        text = row["testo"] or ""
                    else:  # rubrica
                        text = row["rubrica"] or ""
                    # Truncate very long texts
                    texts.append(text[:8000] if len(text) > 8000 else text)
                
                # Skip empty batch
                if not any(texts):
                    processed += len(batch)
                    continue
                
                try:
                    embeddings = await get_embeddings(client, texts)
                    
                    # Insert into DB
                    for row, emb in zip(batch, embeddings):
                        await conn.execute("""
                            INSERT INTO kb.normativa_embeddings (normativa_id, model, channel, dims, embedding)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (normativa_id, model, channel, dims) DO NOTHING
                        """, row["id"], MODEL, channel, DIMS, str(emb))
                    
                    processed += len(batch)
                    current_code = batch[0]["code"] if batch else "?"
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {channel}: {processed}/{total} ({current_code})")
                    
                except Exception as e:
                    print(f"  ERROR at batch {i}: {e}")
                    # Wait and retry
                    await asyncio.sleep(5)
                    continue
                
                # Rate limiting
                await asyncio.sleep(0.5)
    
    # Final stats
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === FINAL STATS ===")
    stats = await conn.fetch("""
        SELECT model, channel, COUNT(*) as cnt 
        FROM kb.normativa_embeddings 
        GROUP BY model, channel
    """)
    for row in stats:
        print(f"  {row['model']}/{row['channel']}: {row['cnt']}")
    
    total_emb = await conn.fetchval("SELECT COUNT(*) FROM kb.normativa_embeddings")
    total_art = await conn.fetchval("SELECT COUNT(*) FROM kb.normativa")
    print(f"\n  Total embeddings: {total_emb}")
    print(f"  Total articles: {total_art}")
    print(f"  Coverage: {total_emb / (total_art * 2) * 100:.1f}%")
    
    await conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
