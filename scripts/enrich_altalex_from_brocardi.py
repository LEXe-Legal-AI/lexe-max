#!/usr/bin/env python3
"""
Enrich Altalex articles with Brocardi data.

Extracts from Brocardi:
- is_abrogato: TRUE if Brocardi marks article as abrogated
- abrogation_note: The abrogation text from Brocardi
- cross_refs: Cross-references like [art. 2043, art. 2059]

Usage:
    cd lexe-max
    uv run python scripts/enrich_altalex_from_brocardi.py
    uv run python scripts/enrich_altalex_from_brocardi.py --dry-run
"""

import asyncio
import argparse
import re
from datetime import datetime

import asyncpg
import structlog

logger = structlog.get_logger()

# Database connection
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"


# ============================================================
# EXTRACTION PATTERNS
# ============================================================

# Pattern for cross-references in Brocardi: [art. 2043, art. 2059]
CROSS_REF_PATTERN = re.compile(
    r'\[\s*'                           # Opening bracket
    r'(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)'  # Article number
    r'(?:\s*,\s*\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)*'  # Additional numbers
    r'\s*\]',                          # Closing bracket
    re.IGNORECASE
)

# Individual cross-ref numbers
CROSS_REF_NUM_PATTERN = re.compile(
    r'(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)',
    re.IGNORECASE
)

# Abrogation patterns
ABROGATION_PATTERNS = [
    re.compile(r'articolo\s+abrogat[oa]', re.IGNORECASE),
    re.compile(r'abrogat[oa]\s+(?:da|dal|dalla|dall)', re.IGNORECASE),
    re.compile(r'implicitamente\s+abrogat', re.IGNORECASE),
    re.compile(r'soppression[ei]', re.IGNORECASE),
]


def extract_cross_refs(text: str) -> list[str]:
    """Extract cross-references from Brocardi text."""
    refs = []

    # Find all bracketed references
    for match in CROSS_REF_PATTERN.finditer(text):
        bracket_content = match.group(0)
        # Extract individual article numbers
        for num_match in CROSS_REF_NUM_PATTERN.finditer(bracket_content):
            num = num_match.group(1).replace(' ', '').lower()
            if num not in refs:
                refs.append(num)

    return refs


def detect_abrogation(text: str) -> tuple[bool, str | None]:
    """Detect if article is abrogated and extract note."""
    text_lower = text.lower()

    # Quick check
    if 'abrogat' not in text_lower and 'soppression' not in text_lower:
        return False, None

    # Check patterns
    for pattern in ABROGATION_PATTERNS:
        if pattern.search(text):
            # Extract first sentence or line as note
            lines = text.strip().split('\n')
            note = lines[0].strip() if lines else text[:200]
            return True, note

    return False, None


async def enrich_articles(dry_run: bool = False):
    """Main enrichment function."""

    print(f"\n{'='*60}")
    print(f"  ENRICH ALTALEX FROM BROCARDI")
    print(f"  {datetime.now().isoformat()}")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    conn = await asyncpg.connect(DB_URL)
    print("Connected to database")

    # Get all matched articles
    rows = await conn.fetch("""
        SELECT
            a.id AS altalex_id,
            a.codice,
            a.articolo,
            b.testo AS brocardi_testo
        FROM kb.normativa_altalex a
        JOIN kb.normativa b ON b.codice = a.codice AND b.articolo = a.articolo
        WHERE a.brocardi_match_id IS NOT NULL
        AND NOT a.is_preleggi AND NOT a.is_attuazione
    """)

    print(f"Found {len(rows)} matched articles to enrich\n")

    stats = {
        'total': len(rows),
        'abrogated': 0,
        'with_cross_refs': 0,
        'updated': 0,
        'errors': 0,
    }

    for i, row in enumerate(rows, 1):
        try:
            brocardi_text = row['brocardi_testo']

            # Extract enrichment data
            is_abrogato, abrogation_note = detect_abrogation(brocardi_text)
            cross_refs = extract_cross_refs(brocardi_text)

            if is_abrogato:
                stats['abrogated'] += 1
            if cross_refs:
                stats['with_cross_refs'] += 1

            # Update if not dry run
            if not dry_run and (is_abrogato or cross_refs):
                await conn.execute("""
                    UPDATE kb.normativa_altalex SET
                        brocardi_is_abrogato = $1,
                        brocardi_abrogation_note = $2,
                        brocardi_cross_refs = $3,
                        updated_at = NOW()
                    WHERE id = $4
                """,
                    is_abrogato,
                    abrogation_note,
                    cross_refs if cross_refs else None,
                    row['altalex_id']
                )
                stats['updated'] += 1

            # Progress
            if i % 500 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] Abrogated: {stats['abrogated']} | With refs: {stats['with_cross_refs']}")

        except Exception as e:
            stats['errors'] += 1
            logger.error("Error enriching", articolo=row['articolo'], error=str(e))

    await conn.close()

    # Final stats
    print(f"\n{'='*60}")
    print(f"  ENRICHMENT RESULTS")
    print(f"{'='*60}")
    print(f"  Total processed: {stats['total']}")
    print(f"  Abrogated detected: {stats['abrogated']}")
    print(f"  With cross-refs: {stats['with_cross_refs']}")
    print(f"  Updated (if not dry-run): {stats['updated']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="Enrich Altalex from Brocardi")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    await enrich_articles(dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
