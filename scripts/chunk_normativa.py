#!/usr/bin/env python3
"""
Chunk normativa articles for fine-grained retrieval.

Splits articles into chunks of ~900-1200 chars with smart split points
at legal structure markers (Art., Comma, etc.).

Usage:
    uv run python scripts/chunk_normativa.py --work COST
    uv run python scripts/chunk_normativa.py --work ALL
    uv run python scripts/chunk_normativa.py --work ALL --dry-run
"""

print("DEBUG: Starting imports...", flush=True)
import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID
print("DEBUG: Standard imports done...", flush=True)

import asyncpg
print("DEBUG: asyncpg imported...", flush=True)

# =============================================================================
# CONFIG
# =============================================================================

DB_URL = os.environ.get(
    "LEXE_KB_DSN",
    "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
)

# Chunking parameters
TARGET_CHARS = 1000      # Target chunk size
MIN_CHARS = 900          # Minimum chunk size
MAX_CHARS = 1200         # Maximum chunk size
OVERLAP_CHARS = 150      # Overlap between chunks
MIN_CHUNK_LEN = 30       # Minimum final chunk length (DB CHECK)

# Legal split patterns (Italian)
LEGAL_SPLIT_PATTERNS = [
    r'^\s*Art\.\s',           # "Art. "
    r'^\s*Articolo\s',        # "Articolo "
    r'^\s*Comma\s',           # "Comma "
    r'^\s*Lettera\s',         # "Lettera "
    r'^\s*Capo\s',            # "Capo "
    r'^\s*Sezione\s',         # "Sezione "
    r'^\s*\d+\.\s',           # "1. ", "2. "
    r'^\s*[a-z]\)\s',         # "a) ", "b) "
    r'^\s*\d+-[a-z]+\.\s',    # "1-bis. "
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Chunk:
    """A text chunk from an article."""
    chunk_no: int
    char_start: int
    char_end: int
    text: str
    token_est: int


@dataclass
class ChunkStats:
    """Statistics for chunking operation."""
    work_code: str
    articoli_tot: int
    articoli_chunkizzati: int
    chunks_creati: int
    media_chunk_per_articolo: float
    max_chunk_per_articolo: int
    articoli_troppo_corti: int


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_for_chunking(text: str) -> str:
    """
    Normalize text for chunking and embedding.

    - Converts typographic quotes/apostrophes to standard
    - Converts NBSP to regular space
    - Collapses multiple spaces
    - Limits consecutive newlines to 2
    - Does NOT remove legal references
    """
    if not text:
        return ""

    # Typographic apostrophes -> standard (Unicode codepoints)
    text = text.replace('\u2019', "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace('\u2018', "'")  # LEFT SINGLE QUOTATION MARK

    # Typographic quotes -> standard
    text = text.replace('\u201c', '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace('\u201d', '"')  # RIGHT DOUBLE QUOTATION MARK

    # Non-breaking space -> regular space
    text = text.replace('\u00a0', ' ')

    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Max 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# =============================================================================
# CHUNKING LOGIC
# =============================================================================

def find_split_point(text: str, target_pos: int, window: int = 100) -> int:
    """
    Find best split point near target position.

    Priority:
    1. Legal structure marker (Art., Comma, etc.) within window
    2. Double newline (paragraph break)
    3. Sentence end (. followed by space/newline)
    4. Target position as fallback
    """
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    search_text = text[start:end]

    # 1. Look for legal markers
    for pattern in LEGAL_SPLIT_PATTERNS:
        match = re.search(pattern, search_text, re.MULTILINE)
        if match:
            return start + match.start()

    # 2. Look for paragraph break (double newline)
    para_break = search_text.find('\n\n')
    if para_break != -1:
        return start + para_break

    # 3. Look for sentence end
    sentence_end = re.search(r'\.\s', search_text)
    if sentence_end:
        return start + sentence_end.end()

    # 4. Fallback to target position
    return target_pos


def chunk_text(text: str) -> list[Chunk]:
    """
    Split text into chunks with smart split points.

    Returns list of Chunk objects with char positions and token estimates.
    """
    if not text or len(text.strip()) < MIN_CHUNK_LEN:
        return []

    # Normalize first
    normalized = normalize_for_chunking(text)
    if len(normalized) < MIN_CHUNK_LEN:
        return []

    chunks = []
    pos = 0
    chunk_no = 0

    while pos < len(normalized):
        # Calculate end position
        if pos + MAX_CHARS >= len(normalized):
            # Last chunk - take all remaining
            end = len(normalized)
        else:
            # Find smart split point
            target = pos + TARGET_CHARS
            end = find_split_point(normalized, target)

            # Ensure we're within bounds
            if end <= pos:
                end = min(pos + MAX_CHARS, len(normalized))
            if end > pos + MAX_CHARS:
                end = pos + MAX_CHARS

        # Extract chunk text
        chunk_text = normalized[pos:end].strip()

        # Skip if too short (will be appended to previous)
        if len(chunk_text) < MIN_CHUNK_LEN:
            if chunks and chunk_no > 0:
                # Append to previous chunk
                prev = chunks[-1]
                new_text = prev.text + " " + chunk_text
                chunks[-1] = Chunk(
                    chunk_no=prev.chunk_no,
                    char_start=prev.char_start,
                    char_end=end,
                    text=new_text,
                    token_est=len(new_text) // 4
                )
            break

        # Create chunk
        chunks.append(Chunk(
            chunk_no=chunk_no,
            char_start=pos,
            char_end=end,
            text=chunk_text,
            token_est=len(chunk_text) // 4
        ))

        # Move to next position (with overlap)
        chunk_no += 1
        pos = end - OVERLAP_CHARS
        if pos < end - MIN_CHARS:
            pos = end  # Avoid too much overlap

    return chunks


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def get_work_id(conn: asyncpg.Connection, work_code: str) -> Optional[UUID]:
    """Get work UUID by code."""
    return await conn.fetchval(
        "SELECT id FROM kb.work WHERE code = $1",
        work_code
    )


async def get_all_work_codes(conn: asyncpg.Connection) -> list[str]:
    """Get all work codes."""
    rows = await conn.fetch("SELECT code FROM kb.work ORDER BY code")
    return [row['code'] for row in rows]


async def get_articles_for_work(
    conn: asyncpg.Connection,
    work_id: UUID
) -> list[dict]:
    """Get all articles for a work."""
    return await conn.fetch("""
        SELECT
            id,
            articolo,
            articolo_num,
            articolo_suffix,
            articolo_sort_key,
            testo
        FROM kb.normativa
        WHERE work_id = $1
        ORDER BY articolo_sort_key
    """, work_id)


async def delete_chunks_for_work(
    conn: asyncpg.Connection,
    work_id: UUID
) -> int:
    """Delete all chunks for a work (CASCADE cleans FTS and embeddings)."""
    result = await conn.execute(
        "DELETE FROM kb.normativa_chunk WHERE work_id = $1",
        work_id
    )
    # Parse "DELETE X" to get count
    return int(result.split()[-1]) if result else 0


async def insert_chunk(
    conn: asyncpg.Connection,
    normativa_id: UUID,
    work_id: UUID,
    articolo_sort_key: str,
    articolo_num: Optional[int],
    articolo_suffix: Optional[str],
    chunk: Chunk
) -> None:
    """Insert a single chunk."""
    await conn.execute("""
        INSERT INTO kb.normativa_chunk (
            normativa_id, work_id, articolo_sort_key,
            articolo_num, articolo_suffix,
            chunk_no, char_start, char_end, text, token_est
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """,
        normativa_id, work_id, articolo_sort_key,
        articolo_num, articolo_suffix,
        chunk.chunk_no, chunk.char_start, chunk.char_end,
        chunk.text, chunk.token_est
    )


async def check_short_articles(
    conn: asyncpg.Connection,
    work_id: UUID
) -> int:
    """Count articles with text too short for chunking."""
    return await conn.fetchval("""
        SELECT count(*)
        FROM kb.normativa
        WHERE work_id = $1
          AND (testo IS NULL OR length(trim(testo)) < $2)
    """, work_id, MIN_CHUNK_LEN)


# =============================================================================
# MAIN CHUNKING FUNCTION
# =============================================================================

async def chunk_work(
    conn: asyncpg.Connection,
    work_code: str,
    dry_run: bool = False
) -> Optional[ChunkStats]:
    """
    Chunk all articles for a work.

    Idempotent: deletes existing chunks before inserting new ones.
    """
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] Processing work: {work_code}")

    # Get work ID
    work_id = await get_work_id(conn, work_code)
    if not work_id:
        print(f"  ERROR: Work '{work_code}' not found")
        return None

    # Pre-check: count short articles
    short_count = await check_short_articles(conn, work_id)
    if short_count > 0:
        print(f"  WARNING: {short_count} articles with text < {MIN_CHUNK_LEN} chars (will be skipped)")

    # Get articles
    articles = await get_articles_for_work(conn, work_id)
    total_articles = len(articles)
    print(f"  Found {total_articles} articles")

    if total_articles == 0:
        return ChunkStats(
            work_code=work_code,
            articoli_tot=0,
            articoli_chunkizzati=0,
            chunks_creati=0,
            media_chunk_per_articolo=0,
            max_chunk_per_articolo=0,
            articoli_troppo_corti=0
        )

    if not dry_run:
        # Delete existing chunks (idempotent)
        deleted = await delete_chunks_for_work(conn, work_id)
        if deleted > 0:
            print(f"  Deleted {deleted} existing chunks")

    # Chunk each article
    chunks_created = 0
    articles_chunked = 0
    articles_skipped = 0
    max_chunks = 0
    chunks_per_article = []

    for art in articles:
        text = art['testo']
        if not text or len(text.strip()) < MIN_CHUNK_LEN:
            articles_skipped += 1
            continue

        chunks = chunk_text(text)
        if not chunks:
            articles_skipped += 1
            continue

        articles_chunked += 1
        chunks_per_article.append(len(chunks))
        max_chunks = max(max_chunks, len(chunks))

        if not dry_run:
            for chunk in chunks:
                await insert_chunk(
                    conn,
                    normativa_id=art['id'],
                    work_id=work_id,
                    articolo_sort_key=art['articolo_sort_key'],
                    articolo_num=art['articolo_num'],
                    articolo_suffix=art['articolo_suffix'],
                    chunk=chunk
                )

        chunks_created += len(chunks)

    # Calculate stats
    avg_chunks = sum(chunks_per_article) / len(chunks_per_article) if chunks_per_article else 0

    stats = ChunkStats(
        work_code=work_code,
        articoli_tot=total_articles,
        articoli_chunkizzati=articles_chunked,
        chunks_creati=chunks_created,
        media_chunk_per_articolo=round(avg_chunks, 2),
        max_chunk_per_articolo=max_chunks,
        articoli_troppo_corti=articles_skipped
    )

    mode = "[DRY-RUN] " if dry_run else ""
    print(f"  {mode}Created {chunks_created} chunks from {articles_chunked} articles")
    print(f"  {mode}Avg chunks/article: {stats.media_chunk_per_articolo}, Max: {max_chunks}")
    if articles_skipped > 0:
        print(f"  {mode}Skipped {articles_skipped} short/empty articles")

    return stats


# =============================================================================
# CLI
# =============================================================================

async def main():
    print("DEBUG: Entered main()", flush=True)
    parser = argparse.ArgumentParser(description="Chunk normativa articles")
    parser.add_argument("--work", required=True, help="Work code (e.g., CC, COST) or ALL")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert, just show stats")
    args = parser.parse_args()
    print("DEBUG: Args parsed", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to DB: {DB_URL.split('@')[1] if '@' in DB_URL else DB_URL}", flush=True)
    conn = await asyncpg.connect(DB_URL)

    try:
        # Get work codes
        if args.work.upper() == "ALL":
            work_codes = await get_all_work_codes(conn)
            print(f"  Processing ALL works: {', '.join(work_codes)}")
        else:
            work_codes = [args.work.upper()]

        # Process each work
        all_stats = []
        for code in work_codes:
            stats = await chunk_work(conn, code, dry_run=args.dry_run)
            if stats:
                all_stats.append(stats)

        # Summary
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === SUMMARY ===")
        print(f"{'Work':<8} {'Articles':>10} {'Chunked':>10} {'Chunks':>10} {'Avg':>8} {'Max':>6} {'Short':>8}")
        print("-" * 70)

        total_chunks = 0
        for s in all_stats:
            print(f"{s.work_code:<8} {s.articoli_tot:>10} {s.articoli_chunkizzati:>10} "
                  f"{s.chunks_creati:>10} {s.media_chunk_per_articolo:>8.2f} "
                  f"{s.max_chunk_per_articolo:>6} {s.articoli_troppo_corti:>8}")
            total_chunks += s.chunks_creati

        print("-" * 70)
        print(f"{'TOTAL':<8} {sum(s.articoli_tot for s in all_stats):>10} "
              f"{sum(s.articoli_chunkizzati for s in all_stats):>10} "
              f"{total_chunks:>10}")

        if not args.dry_run:
            # Verify with view
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Verification from v_chunk_stats:")
            rows = await conn.fetch("SELECT * FROM kb.v_chunk_stats ORDER BY work_code")
            for row in rows:
                print(f"  {row['work_code']}: {row['articoli_chunkizzati_pct']}% chunked, "
                      f"avg {row['media_chunk_per_articolo']} chunks/article")

    finally:
        await conn.close()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done!")


if __name__ == "__main__":
    print("DEBUG: Entering main block...", flush=True)
    # Fix for Windows event loop
    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
