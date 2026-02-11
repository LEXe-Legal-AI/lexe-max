#!/usr/bin/env python3
"""Ingest Codice Civile from Altalex MD to kb.normativa_altalex.

Parses Altalex PDF->MD export and loads into new table, then matches
with existing Brocardi records to annotate extras.

Usage:
    cd lexe-max

    # Full ingestion CC
    uv run python scripts/ingest_altalex.py \
        --altalex "C:/Users/Fra/Downloads/Edizione 2025 Altalex.md" \
        --codice CC

    # Dry run (no DB writes)
    uv run python scripts/ingest_altalex.py \
        --altalex "C:/Users/Fra/Downloads/Edizione 2025 Altalex.md" \
        --codice CC --dry-run

    # With Brocardi matching
    uv run python scripts/ingest_altalex.py \
        --altalex "C:/Users/Fra/Downloads/Edizione 2025 Altalex.md" \
        --codice CC --match-brocardi
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import uuid

import asyncpg
import structlog

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexe_api.kb.sources.altalex_adapter import (
    AltalexAdapter,
    AltalexArticle,
    normalize_text,
    compute_hash,
)

logger = structlog.get_logger()

# Database connection
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"


async def ensure_table_exists(conn: asyncpg.Connection) -> None:
    """Create table if not exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.normativa_altalex (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            codice VARCHAR(50) NOT NULL,
            articolo VARCHAR(20) NOT NULL,
            is_preleggi BOOLEAN DEFAULT FALSE,
            is_attuazione BOOLEAN DEFAULT FALSE,
            rubrica TEXT,
            testo TEXT NOT NULL,
            testo_normalizzato TEXT,
            content_hash VARCHAR(64),
            libro VARCHAR(200),
            titolo VARCHAR(200),
            capo VARCHAR(200),
            sezione VARCHAR(200),
            source_file VARCHAR(500),
            source_edition VARCHAR(100),
            line_start INTEGER,
            line_end INTEGER,
            brocardi_match_id UUID,
            brocardi_similarity FLOAT,
            brocardi_match_status VARCHAR(20),
            brocardi_extras TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(codice, articolo, is_preleggi, is_attuazione)
        )
    """)

    # Create indexes
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_normativa_altalex_codice_art
            ON kb.normativa_altalex(codice, articolo)
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_normativa_altalex_hash
            ON kb.normativa_altalex(content_hash)
    """)


async def insert_article(
    conn: asyncpg.Connection,
    article: AltalexArticle,
) -> uuid.UUID:
    """Insert article into database."""
    testo_norm = normalize_text(article.testo)

    result = await conn.fetchrow("""
        INSERT INTO kb.normativa_altalex (
            codice, articolo, is_preleggi, is_attuazione, rubrica, testo,
            testo_normalizzato, content_hash,
            libro, titolo, capo, sezione,
            source_file, source_edition, line_start, line_end
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8,
            $9, $10, $11, $12,
            $13, $14, $15, $16
        )
        ON CONFLICT (codice, articolo, is_preleggi, is_attuazione)
        DO UPDATE SET
            rubrica = EXCLUDED.rubrica,
            testo = EXCLUDED.testo,
            testo_normalizzato = EXCLUDED.testo_normalizzato,
            content_hash = EXCLUDED.content_hash,
            libro = EXCLUDED.libro,
            titolo = EXCLUDED.titolo,
            capo = EXCLUDED.capo,
            sezione = EXCLUDED.sezione,
            source_file = EXCLUDED.source_file,
            line_start = EXCLUDED.line_start,
            line_end = EXCLUDED.line_end,
            updated_at = NOW()
        RETURNING id
    """,
        article.codice,
        article.articolo,
        article.is_preleggi,
        article.is_attuazione,
        article.rubrica,
        article.testo,
        testo_norm,
        article.content_hash,
        article.libro,
        article.titolo,
        article.capo,
        article.sezione,
        article.source_file,
        article.source_edition,
        article.line_start,
        article.line_end,
    )

    return result['id']


async def match_with_brocardi(
    conn: asyncpg.Connection,
    adapter: AltalexAdapter,
    altalex_articles: list[AltalexArticle],
    codice: str,
) -> dict:
    """Match Altalex articles with Brocardi and annotate extras.

    Returns:
        Stats dict with match results
    """
    logger.info("Matching with Brocardi", codice=codice)

    # Get all Brocardi articles for this codice
    brocardi_rows = await conn.fetch("""
        SELECT id, articolo, rubrica, testo
        FROM kb.normativa
        WHERE codice = $1
    """, codice)

    brocardi_by_num = {row['articolo']: dict(row) for row in brocardi_rows}
    logger.info(f"Found {len(brocardi_by_num)} Brocardi articles for {codice}")

    stats = {
        'matched': 0,
        'exact': 0,
        'format_diff': 0,
        'content_diff': 0,
        'no_brocardi': 0,
        'preleggi_skip': 0,
    }

    for article in altalex_articles:
        # Skip preleggi for matching (Brocardi doesn't have them)
        if article.is_preleggi:
            stats['preleggi_skip'] += 1
            continue

        # Skip attuazione/transitorie (Brocardi doesn't have them in kb.normativa)
        if article.is_attuazione:
            stats['attuazione_skip'] = stats.get('attuazione_skip', 0) + 1
            continue

        brocardi = brocardi_by_num.get(article.articolo)

        if brocardi is None:
            stats['no_brocardi'] += 1
            continue

        # Compare texts
        comparison = adapter.compare_with_brocardi(article, brocardi['testo'])

        # Determine extras in Brocardi
        # For now, we note that Brocardi has the article
        # In future, could check for massime, note, etc.
        extras = []
        if brocardi['rubrica'] and 'Articolo' not in brocardi['rubrica']:
            # Brocardi has a real rubrica (not placeholder)
            pass  # Altalex has better rubricas
        # TODO: Check kb.normativa for linked massime, relazioni, etc.

        # Update Altalex record with match info
        await conn.execute("""
            UPDATE kb.normativa_altalex SET
                brocardi_match_id = $1,
                brocardi_similarity = $2,
                brocardi_match_status = $3,
                brocardi_extras = $4,
                updated_at = NOW()
            WHERE codice = $5 AND articolo = $6 AND is_preleggi = FALSE
        """,
            brocardi['id'],
            comparison['similarity'],
            comparison['status'],
            extras if extras else None,
            codice,
            article.articolo,
        )

        stats['matched'] += 1
        if comparison['status'] == 'verified':
            if comparison['diff_type'] == 'exact':
                stats['exact'] += 1
            else:
                stats['format_diff'] += 1
        else:
            stats['content_diff'] += 1

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Ingest Altalex MD to KB")
    parser.add_argument(
        "--altalex",
        required=True,
        help="Path to Altalex MD file"
    )
    parser.add_argument("--codice", default="CC", help="Codice (CC, CP, etc.)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument(
        "--match-brocardi",
        action="store_true",
        help="Match with existing Brocardi records"
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit articles (0=all)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ALTALEX INGESTION TO KB")
    print(f"  {datetime.now().isoformat()}")
    print(f"  Codice: {args.codice}")
    print(f"  Source: {args.altalex}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Parse Altalex file
    adapter = AltalexAdapter()
    altalex_path = Path(args.altalex)

    if not altalex_path.exists():
        print(f"ERROR: File not found: {altalex_path}")
        return

    print("Parsing Altalex MD file...")
    articles = adapter.parse_and_cache(altalex_path, codice=args.codice)

    # Count by type
    preleggi = [a for a in articles if a.is_preleggi]
    regular = [a for a in articles if not a.is_preleggi]

    print(f"  Total articles: {len(articles)}")
    print(f"  - Preleggi (Art. 1-31): {len(preleggi)}")
    print(f"  - Regular: {len(regular)}")

    if args.limit > 0:
        articles = articles[:args.limit]
        print(f"  Limited to: {len(articles)}")

    # Connect to DB
    if not args.dry_run:
        conn = await asyncpg.connect(DB_URL)
        print("Connected to database")

        # Ensure table exists
        await ensure_table_exists(conn)
        print("Table kb.normativa_altalex ready")

    # Stats
    stats = {
        'inserted': 0,
        'updated': 0,
        'errors': 0,
        'preleggi': 0,
        'attuazione': 0,
    }

    print(f"\nIngesting {len(articles)} articles...")

    for i, article in enumerate(articles, 1):
        try:
            if not args.dry_run:
                await insert_article(conn, article)

            if article.is_preleggi:
                stats['preleggi'] += 1
            elif article.is_attuazione:
                stats['attuazione'] += 1

            stats['inserted'] += 1

            # Progress every 100
            if i % 100 == 0 or i == len(articles):
                print(f"  [{i}/{len(articles)}] Processed...")

        except Exception as e:
            stats['errors'] += 1
            logger.error("Insert error", articolo=article.articolo, error=str(e))

    # Match with Brocardi if requested
    match_stats = None
    if args.match_brocardi and not args.dry_run:
        print("\nMatching with Brocardi...")
        match_stats = await match_with_brocardi(conn, adapter, articles, args.codice)

    if not args.dry_run:
        await conn.close()

    # Final stats
    print(f"\n{'='*60}")
    print(f"  INGESTION RESULTS")
    print(f"{'='*60}")
    print(f"  Total processed: {len(articles)}")
    print(f"  Inserted/Updated: {stats['inserted']}")
    print(f"  Preleggi: {stats['preleggi']}")
    print(f"  Attuazione/Transitorie: {stats['attuazione']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Dry run: {args.dry_run}")

    if match_stats:
        print(f"\n  BROCARDI MATCHING:")
        print(f"  Matched: {match_stats['matched']}")
        print(f"  - Exact match: {match_stats['exact']}")
        print(f"  - Format diff: {match_stats['format_diff']}")
        print(f"  - Content diff: {match_stats['content_diff']}")
        print(f"  No Brocardi: {match_stats['no_brocardi']}")
        print(f"  Preleggi skipped: {match_stats['preleggi_skip']}")
        print(f"  Attuazione skipped: {match_stats.get('attuazione_skip', 0)}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
