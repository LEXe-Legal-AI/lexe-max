#!/usr/bin/env python3
"""
Chunk normativa articles for fine-grained retrieval.

Splits articles into chunks of ~900-1200 chars with smart split points.

Usage:
    uv run python scripts/chunk_normativa_v2.py --work COST --dry-run
    uv run python scripts/chunk_normativa_v2.py --work ALL
"""

import argparse
import asyncio
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

# Windows event loop fix - MUST be before asyncpg import
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncpg


# =============================================================================
# CONFIG
# =============================================================================

TARGET_CHARS = 1000
MIN_CHARS = 900
MAX_CHARS = 1200
OVERLAP_CHARS = 150
MIN_CHUNK_LEN = 30


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Chunk:
    chunk_no: int
    char_start: int
    char_end: int
    text: str
    token_est: int


@dataclass
class ChunkStats:
    work_code: str
    articoli_tot: int
    articoli_chunkizzati: int
    chunks_creati: int
    media_chunk_per_articolo: float
    max_chunk_per_articolo: int
    articoli_troppo_corti: int


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def normalize_for_chunking(text: str) -> str:
    if not text:
        return ""
    # Typographic chars -> standard
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u00a0', ' ')
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def find_split_point(text: str, target_pos: int, window: int = 100) -> int:
    """Find best split point near target position."""
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    search_text = text[start:end]

    # Priority 1: Double newline
    idx = search_text.rfind('\n\n')
    if idx != -1:
        return start + idx + 2

    # Priority 2: Period + space/newline
    for pattern in ['. ', '.\n']:
        idx = search_text.rfind(pattern)
        if idx != -1:
            return start + idx + len(pattern)

    # Priority 3: Comma + space
    idx = search_text.rfind(', ')
    if idx != -1:
        return start + idx + 2

    # Fallback: space
    idx = search_text.rfind(' ')
    if idx != -1:
        return start + idx + 1

    return target_pos


def create_chunks(text: str) -> list[Chunk]:
    """Create chunks from text."""
    text = normalize_for_chunking(text)
    if len(text) < MIN_CHUNK_LEN:
        return []

    chunks = []
    pos = 0
    chunk_no = 0

    while pos < len(text):
        # Calculate end position
        end_pos = min(pos + TARGET_CHARS, len(text))

        # If not at end, find better split point
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

        # Move position with overlap
        pos = end_pos - OVERLAP_CHARS if end_pos < len(text) else end_pos

    return chunks


# =============================================================================
# DATABASE
# =============================================================================

async def get_connection():
    return await asyncpg.connect(
        host="localhost",
        port=5434,
        user="lexe_kb",
        password="lexe_kb_dev_password",
        database="lexe_kb",
        ssl=False
    )


async def get_work_codes(conn, work_arg: str) -> list[str]:
    if work_arg.upper() == "ALL":
        rows = await conn.fetch("""
            SELECT DISTINCT w.code
            FROM kb.work w
            JOIN kb.normativa n ON n.work_id = w.id
            ORDER BY w.code
        """)
        return [r['code'] for r in rows]
    return [work_arg.upper()]


async def chunk_work(conn, work_code: str, dry_run: bool) -> Optional[ChunkStats]:
    """Chunk all articles for a work."""
    # Get work_id
    work = await conn.fetchrow("SELECT id FROM kb.work WHERE code = $1", work_code)
    if not work:
        print(f"  ERROR: Work '{work_code}' not found")
        return None

    work_id = work['id']

    # Get articles
    articles = await conn.fetch("""
        SELECT id, articolo, testo, articolo_sort_key, articolo_num, articolo_suffix
        FROM kb.normativa
        WHERE work_id = $1
        ORDER BY articolo_sort_key
    """, work_id)

    articoli_tot = len(articles)
    articoli_chunkizzati = 0
    chunks_creati = 0
    max_chunk_per_articolo = 0
    articoli_troppo_corti = 0

    print(f"\n=== {work_code} ===")
    print(f"  Articles: {articoli_tot}")

    for art in articles:
        text = art['testo']
        if len(text.strip()) < MIN_CHUNK_LEN:
            articoli_troppo_corti += 1
            continue

        chunks = create_chunks(text)
        if not chunks:
            articoli_troppo_corti += 1
            continue

        articoli_chunkizzati += 1
        chunks_creati += len(chunks)
        max_chunk_per_articolo = max(max_chunk_per_articolo, len(chunks))

        if not dry_run:
            # Delete existing chunks
            await conn.execute(
                "DELETE FROM kb.normativa_chunk WHERE normativa_id = $1",
                art['id']
            )

            # Insert new chunks
            for chunk in chunks:
                await conn.execute("""
                    INSERT INTO kb.normativa_chunk
                    (normativa_id, work_id, articolo_sort_key, articolo_num, articolo_suffix,
                     chunk_no, char_start, char_end, text, token_est)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, art['id'], work_id, art['articolo_sort_key'],
                    art['articolo_num'], art['articolo_suffix'],
                    chunk.chunk_no, chunk.char_start, chunk.char_end,
                    chunk.text, chunk.token_est)

    media = chunks_creati / articoli_chunkizzati if articoli_chunkizzati > 0 else 0

    print(f"  Chunked: {articoli_chunkizzati}/{articoli_tot}")
    print(f"  Chunks: {chunks_creati} (avg {media:.1f}/article, max {max_chunk_per_articolo})")
    print(f"  Too short: {articoli_troppo_corti}")

    return ChunkStats(
        work_code=work_code,
        articoli_tot=articoli_tot,
        articoli_chunkizzati=articoli_chunkizzati,
        chunks_creati=chunks_creati,
        media_chunk_per_articolo=media,
        max_chunk_per_articolo=max_chunk_per_articolo,
        articoli_troppo_corti=articoli_troppo_corti
    )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Chunk normativa articles")
    parser.add_argument("--work", required=True, help="Work code (CC, COST) or ALL")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting...")
    conn = await get_connection()

    try:
        work_codes = await get_work_codes(conn, args.work)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing: {', '.join(work_codes)}")

        all_stats = []
        for code in work_codes:
            stats = await chunk_work(conn, code, args.dry_run)
            if stats:
                all_stats.append(stats)

        # Summary
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SUMMARY")
        print(f"{'='*60}")

        total_chunks = sum(s.chunks_creati for s in all_stats)
        total_articles = sum(s.articoli_tot for s in all_stats)
        total_chunked = sum(s.articoli_chunkizzati for s in all_stats)

        print(f"Works: {len(all_stats)}")
        print(f"Articles: {total_articles}")
        print(f"Chunked: {total_chunked}")
        print(f"Chunks: {total_chunks}")

        if args.dry_run:
            print("\n[DRY-RUN] No changes made")

    finally:
        await conn.close()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
