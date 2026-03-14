#!/usr/bin/env python3
"""Populate concept hierarchy for main Italian legal codes.

Builds hierarchical structure for 6 principal codes:
CC (Codice Civile), CP (Codice Penale), CPC (Codice di Procedura Civile),
CPP (Codice di Procedura Penale), COST (Costituzione), CCI (Codice della Crisi).

Usage:
    python populate_concept_hierarchy.py [--db-url DATABASE_URL] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

# Rubrica patterns for structure extraction
STRUCTURE_PATTERNS = [
    (2, re.compile(r"^Libro\s+([IVXLCDM]+)", re.IGNORECASE)),
    (3, re.compile(r"^Titolo\s+([IVXLCDM]+)", re.IGNORECASE)),
    (4, re.compile(r"^Capo\s+([IVXLCDM]+)", re.IGNORECASE)),
    (5, re.compile(r"^Sezione\s+([IVXLCDM]+)", re.IGNORECASE)),
]

# Code abbreviations to work_id mapping (populated at runtime)
CODE_ABBREVIATIONS = {
    "codice civile": "CC",
    "codice penale": "CP",
    "codice di procedura civile": "CPC",
    "codice di procedura penale": "CPP",
    "costituzione": "COST",
    "codice della crisi d'impresa": "CCI",
}


async def populate_for_code(
    conn: asyncpg.Connection,
    work_id: UUID,
    code_abbrev: str,
    dry_run: bool = False,
) -> int:
    """Populate hierarchy for a single code.

    Returns count of articles updated.
    """
    # Get all articles for this code, ordered by article number
    articles = await conn.fetch(
        """
        SELECT id, article, rubrica, urn
        FROM kb.normativa
        WHERE work_id = $1
        ORDER BY
            CASE WHEN article ~ '^[0-9]+'
                 THEN LPAD(regexp_replace(article, '[^0-9].*', ''), 10, '0')
                 ELSE article END
        """,
        work_id,
    )

    if not articles:
        logger.warning("No articles found for work_id=%s", work_id)
        return 0

    logger.info("Processing %d articles for %s (work_id=%s)", len(articles), code_abbrev, work_id)

    # Build concept paths from rubrica patterns
    structure_stack: dict[int, str] = {}  # level -> label
    updated = 0

    for article in articles:
        rubrica = article.get("rubrica") or ""

        # Check if this rubrica indicates a new structural section
        for level, pattern in STRUCTURE_PATTERNS:
            match = pattern.match(rubrica.strip())
            if match:
                label = rubrica.strip()
                structure_stack[level] = label
                # Clear deeper levels
                for deeper in range(level + 1, 7):
                    structure_stack.pop(deeper, None)
                break

        # Build concept_path from current stack
        path = [code_abbrev]
        for lvl in sorted(structure_stack.keys()):
            path.append(structure_stack[lvl])

        if dry_run:
            logger.debug("  art. %s -> %s", article["article"], path)
        else:
            await conn.execute(
                "UPDATE kb.normativa SET concept_path = $1 WHERE id = $2",
                path,
                article["id"],
            )

        updated += 1

    return updated


async def populate_all(db_url: str, dry_run: bool = False) -> None:
    """Populate concept hierarchy for all supported codes."""
    conn = await asyncpg.connect(db_url)

    try:
        # Find work_ids for known codes
        works = await conn.fetch(
            """
            SELECT DISTINCT w.id, w.act_type
            FROM kb.work w
            WHERE LOWER(w.act_type) = ANY($1)
            """,
            list(CODE_ABBREVIATIONS.keys()),
        )

        total = 0
        for work in works:
            abbrev = CODE_ABBREVIATIONS.get(work["act_type"].lower(), work["act_type"][:3].upper())
            count = await populate_for_code(conn, work["id"], abbrev, dry_run)
            total += count
            logger.info("Updated %d articles for %s", count, abbrev)

        logger.info("Total articles updated: %d", total)
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate concept hierarchy")
    parser.add_argument("--db-url", default="postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(populate_all(args.db_url, args.dry_run))


if __name__ == "__main__":
    main()
