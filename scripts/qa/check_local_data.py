#!/usr/bin/env python3
"""Quick check of local reference units for comparison."""

import asyncio
import asyncpg
from qa_config import DB_URL

PDFS = [
    "2014 Mass civile Vol 1 pagg 408.pdf",
    "2015 pricipi di diritto processuale Volume 2 massimario Civile_401_650.pdf",
    "Rassegna Penale 2011.pdf",
    "Rassegna Civile 2012 - II volume.pdf",
    "Rassegna Penale 2012.pdf",
]


async def main():
    conn = await asyncpg.connect(DB_URL)

    print("PDF                                                    RefUnits  AvgChars")
    print("-" * 80)

    for pdf in PDFS:
        row = await conn.fetchrow("""
            SELECT COUNT(*), COALESCE(AVG(LENGTH(testo)), 0)
            FROM kb.qa_reference_units r
            JOIN kb.pdf_manifest m ON r.manifest_id = m.id
            WHERE m.filename = $1
        """, pdf)
        count, avg = row[0], row[1] or 0
        short_name = pdf[:52] if len(pdf) > 52 else pdf
        print(f"{short_name:<55} {count:>5}  {avg:>8.0f}")

    # Total
    total = await conn.fetchval("SELECT COUNT(*) FROM kb.qa_reference_units")
    print("-" * 80)
    print(f"TOTAL reference units in database: {total}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
