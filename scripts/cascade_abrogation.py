#!/usr/bin/env python3
"""Cascade abrogation flag to child articles.

When a parent structural section is abrogated, flags all articles
within that section as parent_abrogated=TRUE.

Usage:
    python cascade_abrogation.py [--db-url DATABASE_URL] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import asyncpg

logger = logging.getLogger(__name__)


async def cascade(db_url: str, dry_run: bool = False) -> int:
    """Find abrogated parents and flag their children."""
    conn = await asyncpg.connect(db_url)

    try:
        # Find articles where vigenza indicates abrogation
        # and propagate to siblings in same concept_path
        count = 0

        if not dry_run:
            result = await conn.execute(
                """
                UPDATE kb.normativa child
                SET parent_abrogated = TRUE
                FROM kb.normativa parent
                WHERE parent.concept_path IS NOT NULL
                  AND child.concept_path IS NOT NULL
                  AND parent.concept_path[1:array_length(parent.concept_path, 1) - 1]
                      = child.concept_path[1:array_length(child.concept_path, 1) - 1]
                  AND parent.id != child.id
                  AND parent.vigenza_stato = 'abrogato'
                  AND child.parent_abrogated IS NOT TRUE
                """
            )
            # Parse count from result like "UPDATE N"
            if result:
                parts = result.split()
                if len(parts) >= 2:
                    count = int(parts[1])
        else:
            rows = await conn.fetch(
                """
                SELECT child.id, child.article, child.act_type
                FROM kb.normativa child
                JOIN kb.normativa parent ON
                    parent.concept_path[1:array_length(parent.concept_path, 1) - 1]
                    = child.concept_path[1:array_length(child.concept_path, 1) - 1]
                    AND parent.id != child.id
                WHERE parent.vigenza_stato = 'abrogato'
                  AND child.parent_abrogated IS NOT TRUE
                LIMIT 100
                """
            )
            count = len(rows)
            for row in rows[:10]:
                logger.info("Would flag: %s art. %s", row["act_type"], row["article"])

        logger.info("Flagged %d articles as parent_abrogated", count)
        return count
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Cascade abrogation to children")
    parser.add_argument("--db-url", default="postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(cascade(args.db_url, args.dry_run))


if __name__ == "__main__":
    main()
