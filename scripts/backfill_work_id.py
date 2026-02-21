#!/usr/bin/env python3
"""
Backfill work_id, articolo_num, articolo_suffix, articolo_sort_key
for normativa rows missing these values.

Run after ingest_normativa_to_db.py to populate the fields
added by migration 060.

Usage:
    LEXE_KB_DSN=postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb \
    uv run python scripts/backfill_work_id.py
"""

import asyncio
import os
import platform

import asyncpg

DB_URL = os.environ.get(
    "LEXE_KB_DSN",
    "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
)


async def main():
    print(f"Connecting to: {DB_URL.split('@')[1] if '@' in DB_URL else DB_URL}")
    conn = await asyncpg.connect(DB_URL)

    try:
        # 1. Backfill work_id
        result = await conn.execute("""
            UPDATE kb.normativa n
            SET work_id = w.id
            FROM kb.work w
            WHERE w.code = n.codice
              AND n.work_id IS NULL
        """)
        print(f"work_id backfill: {result}")

        # 2. Backfill articolo_num + articolo_suffix
        result = await conn.execute("""
            UPDATE kb.normativa
            SET
                articolo_num = (regexp_match(articolo, '^(\d+)'))[1]::INTEGER,
                articolo_suffix = LOWER(NULLIF((regexp_match(articolo, '^\d+[-]?([a-zA-Z]+)'))[1], ''))
            WHERE articolo_num IS NULL
              AND articolo ~ '^\d+'
        """)
        print(f"articolo_num/suffix backfill: {result}")

        # 3. Backfill articolo_sort_key
        result = await conn.execute("""
            UPDATE kb.normativa
            SET articolo_sort_key = LPAD(articolo_num::TEXT, 6, '0') || '.' ||
                LPAD(
                    COALESCE(kb.fn_suffix_ordinal(articolo_suffix), '00'),
                    2, '0'
                )
            WHERE articolo_sort_key IS NULL
              AND articolo_num IS NOT NULL
        """)
        print(f"articolo_sort_key backfill: {result}")

        # 4. Verify
        rows = await conn.fetch("""
            SELECT codice, count(*) as total,
                   count(work_id) as with_work_id,
                   count(articolo_num) as with_num,
                   count(articolo_sort_key) as with_sort_key
            FROM kb.normativa
            GROUP BY codice
            ORDER BY codice
        """)
        print("\nVerification:")
        print(f"{'Code':<8} {'Total':>8} {'work_id':>10} {'art_num':>10} {'sort_key':>10}")
        print("-" * 50)
        for r in rows:
            print(f"{r['codice']:<8} {r['total']:>8} {r['with_work_id']:>10} "
                  f"{r['with_num']:>10} {r['with_sort_key']:>10}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
