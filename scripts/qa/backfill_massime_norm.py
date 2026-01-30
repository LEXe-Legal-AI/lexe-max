#!/usr/bin/env python3
"""
QA Protocol - Backfill Massime Normalization

Batch-safe, idempotent backfill of testo_normalizzato, content_hash,
and text_fingerprint for kb.massime table.

Uses norm_v2 from normalization.py for consistency with reference units.

Usage:
    uv run python scripts/qa/backfill_massime_norm.py
    uv run python scripts/qa/backfill_massime_norm.py --batch-size 1000
    uv run python scripts/qa/backfill_massime_norm.py --force  # Re-process all
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import asyncpg

# Add src to path for normalization module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lexe_api.kb.ingestion.normalization import (
    normalize_v2,
    compute_content_hash,
    compute_simhash64,
)

from qa_config import DB_URL

# Configuration
BATCH_SIZE = 5000  # Max rows per UPDATE
LOG_EVERY = 1000   # Log progress every N rows


async def main(batch_size: int = BATCH_SIZE, force: bool = False):
    print("=" * 70)
    print("QA PROTOCOL - BACKFILL MASSIME NORMALIZATION")
    print("=" * 70)
    print(f"Batch size: {batch_size}")
    print(f"Force mode: {force}")
    print()

    conn = await asyncpg.connect(DB_URL)

    # Count total and pending
    total = await conn.fetchval("SELECT count(*) FROM kb.massime")
    if force:
        pending = total
        where_clause = ""
    else:
        pending = await conn.fetchval(
            "SELECT count(*) FROM kb.massime WHERE testo_normalizzato IS NULL OR content_hash IS NULL"
        )
        where_clause = "WHERE testo_normalizzato IS NULL OR content_hash IS NULL"

    print(f"Total massime: {total}")
    print(f"Pending: {pending}")

    if pending == 0:
        print("[OK] Nothing to backfill")
        await conn.close()
        return

    # Process in batches
    processed = 0
    errors = 0
    start_time = time.time()

    while True:
        # Fetch batch
        batch = await conn.fetch(
            f"""
            SELECT id, testo
            FROM kb.massime
            {where_clause}
            ORDER BY id
            LIMIT $1
            """,
            batch_size,
        )

        if not batch:
            break

        # Process batch
        updates = []
        for row in batch:
            massima_id = row["id"]
            testo = row["testo"] or ""

            try:
                # Normalize using norm_v2
                testo_norm, spaced_score = normalize_v2(testo)
                content_hash = compute_content_hash(testo_norm)
                fingerprint = compute_simhash64(testo_norm)

                updates.append((
                    massima_id,
                    testo_norm[:50000] if testo_norm else None,  # Truncate if too long
                    content_hash,
                    fingerprint,
                ))

            except Exception as e:
                print(f"  [ERR] massima {massima_id}: {str(e)[:50]}")
                errors += 1
                continue

        # Batch update
        if updates:
            await conn.executemany(
                """
                UPDATE kb.massime
                SET testo_normalizzato = $2,
                    content_hash = $3,
                    text_fingerprint = $4
                WHERE id = $1
                """,
                updates,
            )

        processed += len(batch)

        # Log progress
        if processed % LOG_EVERY == 0 or processed == pending:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            pct = 100 * processed / pending if pending > 0 else 100
            print(f"  [{pct:5.1f}%] {processed:>6}/{pending} processed, {rate:.0f} rows/sec")

        # If force mode, need to update where_clause to skip already processed
        if force:
            # Use the last ID as cursor
            last_id = batch[-1]["id"]
            where_clause = f"WHERE id > '{last_id}'"

    # Final stats
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Rate: {processed/elapsed:.0f} rows/sec" if elapsed > 0 else "N/A")

    # Verify
    filled = await conn.fetchval(
        "SELECT count(*) FROM kb.massime WHERE testo_normalizzato IS NOT NULL AND content_hash IS NOT NULL"
    )
    print(f"\nMassime with norm data: {filled}/{total} ({100*filled/total:.1f}%)")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill massime normalization")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for updates")
    parser.add_argument("--force", action="store_true", help="Re-process all rows (not just NULL)")
    args = parser.parse_args()

    asyncio.run(main(batch_size=args.batch_size, force=args.force))
