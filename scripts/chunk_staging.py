#!/usr/bin/env python3
"""
Chunk normativa articles on STAGING (normativa_altalex).
Uses SSH tunnel on port 5437 -> staging:5436.

Usage:
    uv run python scripts/chunk_staging.py --dry-run
    uv run python scripts/chunk_staging.py
"""

import argparse
import asyncio
import platform
import re
from dataclasses import dataclass
from datetime import datetime

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncpg

# Staging via SSH tunnel (port 5438 -> staging:5435)
DB_HOST = "localhost"
DB_PORT = 5438
DB_USER = "lexe"
DB_PASS = "lexe_stage_cc07b664a88cb8e6"
DB_NAME = "lexe_kb"

TARGET_CHARS = 1000
MIN_CHARS = 900
OVERLAP_CHARS = 150
MIN_CHUNK_LEN = 30


@dataclass
class Chunk:
    chunk_no: int
    char_start: int
    char_end: int
    text: str
    token_est: int


def normalize_for_chunking(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def find_split_point(text: str, target_pos: int, window: int = 100) -> int:
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    search_text = text[start:end]
    idx = search_text.rfind('\n\n')
    if idx != -1:
        return start + idx + 2
    for pattern in ['. ', '.\n']:
        idx = search_text.rfind(pattern)
        if idx != -1:
            return start + idx + len(pattern)
    idx = search_text.rfind(', ')
    if idx != -1:
        return start + idx + 2
    idx = search_text.rfind(' ')
    if idx != -1:
        return start + idx + 1
    return target_pos


def create_chunks(text: str) -> list[Chunk]:
    text = normalize_for_chunking(text)
    if len(text) < MIN_CHUNK_LEN:
        return []
    chunks = []
    pos = 0
    chunk_no = 0
    while pos < len(text):
        end_pos = min(pos + TARGET_CHARS, len(text))
        if end_pos < len(text):
            end_pos = find_split_point(text, end_pos)
        chunk_text = text[pos:end_pos].strip()
        if len(chunk_text) >= MIN_CHUNK_LEN:
            chunks.append(Chunk(
                chunk_no=chunk_no,
                char_start=pos,
                char_end=end_pos,
                text=chunk_text,
                token_est=len(chunk_text) // 4
            ))
            chunk_no += 1
        pos = end_pos - OVERLAP_CHARS if end_pos < len(text) else end_pos
    return chunks


async def get_connection():
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        ssl=False
    )


async def create_chunk_table(conn):
    """Create the chunk table if it doesn't exist."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.normativa_chunk (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            normativa_id UUID NOT NULL REFERENCES kb.normativa_altalex(id),
            codice VARCHAR(50) NOT NULL,
            articolo VARCHAR(20) NOT NULL,
            chunk_no INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_est INTEGER NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_normativa_chunk_codice
        ON kb.normativa_chunk(codice)
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_normativa_chunk_normativa_id
        ON kb.normativa_chunk(normativa_id)
    """)
    print("  Table kb.normativa_chunk ready")


async def main():
    parser = argparse.ArgumentParser(description="Chunk normativa on staging")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert")
    parser.add_argument("--codice", help="Single codice (e.g., CC, CP)")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to STAGING via tunnel...")
    conn = await get_connection()
    print("Connected!")

    try:
        if not args.dry_run:
            await create_chunk_table(conn)

        # Get articles from normativa_altalex
        if args.codice:
            articles = await conn.fetch("""
                SELECT id, codice, articolo, testo
                FROM kb.normativa_altalex
                WHERE codice = $1
                ORDER BY articolo
            """, args.codice.upper())
        else:
            articles = await conn.fetch("""
                SELECT id, codice, articolo, testo
                FROM kb.normativa_altalex
                ORDER BY codice, articolo
            """)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Articles: {len(articles)}")

        articoli_chunkizzati = 0
        chunks_creati = 0
        articoli_troppo_corti = 0
        stats_per_codice = {}

        for art in articles:
            text = art['testo']
            codice = art['codice']

            if codice not in stats_per_codice:
                stats_per_codice[codice] = {'articles': 0, 'chunks': 0}

            stats_per_codice[codice]['articles'] += 1

            if len(text.strip()) < MIN_CHUNK_LEN:
                articoli_troppo_corti += 1
                continue

            chunks = create_chunks(text)
            if not chunks:
                articoli_troppo_corti += 1
                continue

            articoli_chunkizzati += 1
            chunks_creati += len(chunks)
            stats_per_codice[codice]['chunks'] += len(chunks)

            if not args.dry_run:
                await conn.execute(
                    "DELETE FROM kb.normativa_chunk WHERE normativa_id = $1",
                    art['id']
                )

                for chunk in chunks:
                    await conn.execute("""
                        INSERT INTO kb.normativa_chunk
                        (normativa_id, codice, articolo, chunk_no, char_start, char_end, text, token_est)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, art['id'], codice, art['articolo'],
                        chunk.chunk_no, chunk.char_start, chunk.char_end,
                        chunk.text, chunk.token_est)

        # Summary
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SUMMARY")
        print(f"{'='*60}")

        for codice, stats in sorted(stats_per_codice.items()):
            avg = stats['chunks'] / stats['articles'] if stats['articles'] > 0 else 0
            print(f"  {codice}: {stats['articles']} articles -> {stats['chunks']} chunks (avg {avg:.1f})")

        print(f"\nTotal articles: {len(articles)}")
        print(f"Chunked: {articoli_chunkizzati}")
        print(f"Chunks: {chunks_creati}")
        print(f"Too short: {articoli_troppo_corti}")

        if args.dry_run:
            print("\n[DRY-RUN] No changes made")

    finally:
        await conn.close()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
