#!/usr/bin/env python3
"""
Chunk normativa articles on STAGING into kb.normativa_chunk.

Uses SSH tunnel to staging lexe-max DB (port 5436).
Migration 060 must be applied first (creates normativa_chunk + FTS trigger).

Setup SSH tunnel:
    ssh -i ~/.ssh/id_stage_new -L 5437:localhost:5436 root@91.99.229.111 -N

Usage:
    uv run python scripts/chunk_staging.py --dry-run
    uv run python scripts/chunk_staging.py
    uv run python scripts/chunk_staging.py --code CPC
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

# Staging via SSH tunnel (5437 -> staging lexe-max:5436)
DB_HOST = "localhost"
DB_PORT = 5437
DB_USER = "lexe_kb"
DB_PASS = "lexe_kb_secret"
DB_NAME = "lexe_kb"

TARGET_CHARS = 1000
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
                token_est=max(1, len(chunk_text) // 4)
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


async def main():
    parser = argparse.ArgumentParser(description="Chunk normativa on staging (migration 060 schema)")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert, just report stats")
    parser.add_argument("--code", help="Single code (e.g., CC, CP, CPC, CPP, COST)")
    args = parser.parse_args()

    ts = lambda: datetime.now().strftime('%H:%M:%S')
    print(f"[{ts()}] Connecting to STAGING via SSH tunnel (localhost:{DB_PORT})...")
    conn = await get_connection()
    print("Connected!")

    try:
        # Fetch articles from kb.normativa with work linkage
        if args.code:
            articles = await conn.fetch("""
                SELECT n.id, n.work_id, n.codice, n.articolo,
                       n.articolo_num, n.articolo_suffix, n.articolo_sort_key,
                       n.testo
                FROM kb.normativa n
                WHERE UPPER(n.codice) = $1
                  AND n.work_id IS NOT NULL
                ORDER BY n.articolo_sort_key
            """, args.code.upper())
        else:
            articles = await conn.fetch("""
                SELECT n.id, n.work_id, n.codice, n.articolo,
                       n.articolo_num, n.articolo_suffix, n.articolo_sort_key,
                       n.testo
                FROM kb.normativa n
                WHERE n.work_id IS NOT NULL
                ORDER BY n.codice, n.articolo_sort_key
            """)

        print(f"[{ts()}] Articles to chunk: {len(articles)}")

        if not args.dry_run:
            # Clear existing chunks for the codes we're processing
            codes = set(art['codice'] for art in articles)
            for code in codes:
                deleted = await conn.fetchval("""
                    DELETE FROM kb.normativa_chunk nc
                    USING kb.work w
                    WHERE nc.work_id = w.id AND w.code = $1
                    RETURNING count(*)
                """, code.upper())
                # fetchval on DELETE RETURNING count(*) won't work, use execute
            for code in codes:
                await conn.execute("""
                    DELETE FROM kb.normativa_chunk nc
                    USING kb.work w
                    WHERE nc.work_id = w.id AND w.code = $1
                """, code.upper())
            print(f"[{ts()}] Cleared existing chunks for: {', '.join(sorted(codes))}")

        articoli_chunkizzati = 0
        chunks_creati = 0
        articoli_troppo_corti = 0
        stats_per_codice = {}
        batch_values = []
        BATCH_SIZE = 500

        for art in articles:
            codice = art['codice']
            if codice not in stats_per_codice:
                stats_per_codice[codice] = {'articles': 0, 'chunks': 0, 'skipped': 0}
            stats_per_codice[codice]['articles'] += 1

            text = art['testo'] or ""
            if len(text.strip()) < MIN_CHUNK_LEN:
                articoli_troppo_corti += 1
                stats_per_codice[codice]['skipped'] += 1
                continue

            chunks = create_chunks(text)
            if not chunks:
                articoli_troppo_corti += 1
                stats_per_codice[codice]['skipped'] += 1
                continue

            articoli_chunkizzati += 1
            chunks_creati += len(chunks)
            stats_per_codice[codice]['chunks'] += len(chunks)

            if not args.dry_run:
                for chunk in chunks:
                    batch_values.append((
                        art['id'],           # normativa_id
                        art['work_id'],      # work_id
                        art['articolo_sort_key'] or '',  # articolo_sort_key
                        art['articolo_num'],  # articolo_num
                        art['articolo_suffix'],  # articolo_suffix
                        chunk.chunk_no,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.text,
                        chunk.token_est,
                    ))

                    if len(batch_values) >= BATCH_SIZE:
                        await conn.executemany("""
                            INSERT INTO kb.normativa_chunk
                            (normativa_id, work_id, articolo_sort_key, articolo_num,
                             articolo_suffix, chunk_no, char_start, char_end, text, token_est)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """, batch_values)
                        print(f"  [{ts()}] Inserted batch of {len(batch_values)} chunks...")
                        batch_values = []

        # Flush remaining batch
        if batch_values and not args.dry_run:
            await conn.executemany("""
                INSERT INTO kb.normativa_chunk
                (normativa_id, work_id, articolo_sort_key, articolo_num,
                 articolo_suffix, chunk_no, char_start, char_end, text, token_est)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, batch_values)
            print(f"  [{ts()}] Inserted final batch of {len(batch_values)} chunks")

        # Summary
        print(f"\n{'='*60}")
        print(f"[{ts()}] SUMMARY")
        print(f"{'='*60}")
        print(f"{'Code':<8} {'Articles':>10} {'Chunks':>10} {'Skipped':>10} {'Avg':>8}")
        print(f"{'-'*46}")

        for codice, stats in sorted(stats_per_codice.items()):
            chunked = stats['articles'] - stats['skipped']
            avg = stats['chunks'] / chunked if chunked > 0 else 0
            print(f"{codice:<8} {stats['articles']:>10} {stats['chunks']:>10} {stats['skipped']:>10} {avg:>8.1f}")

        print(f"{'-'*46}")
        print(f"{'TOTAL':<8} {len(articles):>10} {chunks_creati:>10} {articoli_troppo_corti:>10}")

        if args.dry_run:
            print(f"\n[DRY-RUN] No changes made")
        else:
            # Verify with stats view
            stats = await conn.fetch("SELECT * FROM kb.v_chunk_stats ORDER BY work_code")
            print(f"\n--- DB Coverage (v_chunk_stats) ---")
            print(f"{'Code':<8} {'Articles':>10} {'Chunked':>10} {'Cov%':>8} {'EmbCov%':>8}")
            for s in stats:
                print(f"{s['work_code']:<8} {s['articoli_tot']:>10} {s['articoli_chunkizzati']:>10} {s['chunk_coverage_pct']:>7.1f}% {s['emb_coverage_pct']:>7.1f}%")

    finally:
        await conn.close()

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    asyncio.run(main())
