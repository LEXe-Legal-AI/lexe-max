#!/usr/bin/env python3
"""
Ingest normativa JSON to PostgreSQL kb.normativa

Carica gli articoli scaricati da Brocardi/StudioCataldi nel database.
Crea anche le entry nel Number-Anchored Knowledge Graph.

Usage:
    cd lexe-max
    uv run python scripts/ingest_normativa_to_db.py --input scripts/cc_brocardi_results.json
    uv run python scripts/ingest_normativa_to_db.py --input scripts/cds_ingestion_results.json
"""

import asyncio
import json
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime
from uuid import UUID

import asyncpg
import structlog

logger = structlog.get_logger()


# Database connection (use KB database)
# Credentials from docker env: lexe_max / lexe_max_dev_password on port 5436
DB_URL = "postgresql://lexe_max:lexe_max_dev_password@localhost:5436/lexe_max"


async def normalize_text(text: str) -> str:
    """Normalizza testo per hashing e search."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    # Strip
    text = text.strip()
    return text


def compute_hash(text: str) -> str:
    """SHA256 hash del testo normalizzato."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_citations_from_raw(citations_raw: list[str]) -> list[dict]:
    """Parse citazioni raw e estrai info strutturate."""
    parsed = []
    for raw in citations_raw:
        # Pattern: CC:2043, CASS:12345:2020, etc.
        if ':' in raw:
            parts = raw.split(':')
            if len(parts) >= 2:
                codice = parts[0].upper()
                numero = parts[1]
                anno = int(parts[2]) if len(parts) > 2 else None

                parsed.append({
                    'raw': raw,
                    'codice': codice,
                    'numero': numero,
                    'anno': anno,
                    'canonical_id': raw.upper()
                })
    return parsed


async def upsert_legal_number(conn: asyncpg.Connection, citation: dict) -> UUID | None:
    """Upsert numero legale e ritorna ID."""
    canonical_id = citation['canonical_id']
    codice = citation['codice']
    numero = citation['numero']
    anno = citation.get('anno')

    # Determina tipo
    if codice in ('CC', 'CP', 'CPC', 'CPP', 'COST', 'CDS', 'CCONS', 'CDPR', 'CNAV'):
        number_type = 'article'
    elif codice in ('L', 'LEGGE'):
        number_type = 'law'
    elif codice in ('DLGS', 'DPR', 'DL', 'DM'):
        number_type = 'decree'
    elif codice == 'CASS':
        number_type = 'sentence'
    else:
        number_type = 'article'  # default

    try:
        result = await conn.fetchrow("""
            INSERT INTO kb.legal_numbers (canonical_id, number_type, codice, numero, anno)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (canonical_id) DO UPDATE SET
                last_seen_at = NOW()
            RETURNING id
        """, canonical_id, number_type, codice, numero, anno)
        return result['id'] if result else None
    except Exception as e:
        logger.warning("Failed to upsert legal number", canonical_id=canonical_id, error=str(e))
        return None


async def ingest_article(conn: asyncpg.Connection, article: dict, source: str) -> UUID | None:
    """Inserisce un articolo nel database."""
    codice = article.get('codice', 'CC')
    articolo = article.get('articolo', '')
    rubrica = article.get('rubrica')
    testo = article.get('testo', '')
    urn = article.get('urn')
    libro = article.get('libro')
    titolo = article.get('titolo')
    source_url = article.get('source_url')
    citations_raw = article.get('citations', [])

    # Normalizza e calcola hash
    testo_norm = await normalize_text(testo)
    mirror_hash = compute_hash(testo_norm)

    try:
        # Upsert articolo
        result = await conn.fetchrow("""
            INSERT INTO kb.normativa (
                urn_nir, codice, articolo, rubrica, testo, testo_normalizzato,
                libro, titolo, mirror_source, mirror_url, mirror_hash,
                validation_status, is_current
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending', TRUE)
            ON CONFLICT (codice, articolo, comma, data_vigenza_da) DO UPDATE SET
                testo = EXCLUDED.testo,
                testo_normalizzato = EXCLUDED.testo_normalizzato,
                rubrica = EXCLUDED.rubrica,
                mirror_hash = EXCLUDED.mirror_hash,
                updated_at = NOW()
            RETURNING id
        """, urn, codice, articolo, rubrica, testo, testo_norm,
             libro, titolo, source, source_url, mirror_hash)

        if not result:
            return None

        normativa_id = result['id']

        # Inserisci citazioni
        if citations_raw:
            parsed_citations = extract_citations_from_raw(citations_raw)
            for cit in parsed_citations:
                # Upsert legal number
                number_id = await upsert_legal_number(conn, cit)

                # Crea edge nel graph
                if number_id:
                    await conn.execute("""
                        INSERT INTO kb.number_citations (
                            source_type, source_id, target_number_id, target_canonical, raw_citation
                        ) VALUES ('normativa', $1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                    """, normativa_id, number_id, cit['canonical_id'], cit['raw'])

                # Crea citazione in normativa_citations se è un articolo
                if cit['codice'] in ('CC', 'CP', 'CPC', 'CPP', 'COST', 'CDS'):
                    await conn.execute("""
                        INSERT INTO kb.normativa_citations (
                            source_id, target_codice, target_articolo, raw_citation
                        ) VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                    """, normativa_id, cit['codice'], cit['numero'], cit['raw'])

        return normativa_id

    except Exception as e:
        logger.error("Failed to ingest article", articolo=articolo, error=str(e))
        return None


async def main():
    parser = argparse.ArgumentParser(description="Ingest normativa JSON to PostgreSQL")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--source", default="brocardi", help="Source name (brocardi, studiocataldi)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for commits")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually insert")
    args = parser.parse_args()

    # Load JSON
    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get('articles', [])
    codice = data.get('codice', 'CC')

    print(f"\n{'='*60}")
    print(f"  INGEST NORMATIVA TO DB")
    print(f"  {datetime.now().isoformat()}")
    print(f"{'='*60}")
    print(f"  Input: {args.input}")
    print(f"  Codice: {codice}")
    print(f"  Articles: {len(articles)}")
    print(f"  Source: {args.source}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN - no changes will be made")
        return

    # Connect to DB
    try:
        conn = await asyncpg.connect(DB_URL)
        print(f"Connected to database")
    except Exception as e:
        print(f"ERROR: Cannot connect to database: {e}")
        print("Make sure lexe-max container is running on port 5436")
        return

    # Process articles
    success = 0
    errors = 0
    start_time = datetime.now()

    try:
        for i, article in enumerate(articles, 1):
            # Add codice if missing
            if 'codice' not in article:
                article['codice'] = codice

            result = await ingest_article(conn, article, args.source)

            if result:
                success += 1
            else:
                errors += 1

            # Progress every 100
            if i % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(articles)}] {rate:.1f} art/sec | Success: {success} | Errors: {errors}")

    finally:
        await conn.close()

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total articles: {len(articles)}")
    print(f"  Inserted: {success}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Rate: {success/elapsed:.1f} articles/sec")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
