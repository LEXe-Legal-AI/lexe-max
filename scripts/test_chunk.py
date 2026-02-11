#!/usr/bin/env python3
"""Minimal test for chunking."""

import asyncio
import platform
import sys

# Windows event loop fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncpg

async def main():
    print("Connecting...", flush=True)
    conn = await asyncpg.connect(
        host="localhost",
        port=5434,
        user="lexe_kb",
        password="lexe_kb_dev_password",
        database="lexe_kb",
        ssl=False
    )
    print("Connected!", flush=True)

    # Get COST work
    work = await conn.fetchrow("SELECT id FROM kb.work WHERE code = 'COST'")
    print(f"Work ID: {work['id']}", flush=True)

    # Get articles
    articles = await conn.fetch("""
        SELECT id, articolo, testo, articolo_sort_key, articolo_num, articolo_suffix
        FROM kb.normativa
        WHERE work_id = $1
        ORDER BY articolo_sort_key
    """, work['id'])
    print(f"Found {len(articles)} articles", flush=True)

    # Simple chunking
    total_chunks = 0
    for art in articles:
        text = art['testo']
        if len(text) < 30:
            continue
        # Simple split by 1000 chars
        chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
        total_chunks += len(chunks)

    print(f"Total chunks: {total_chunks}", flush=True)

    await conn.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
