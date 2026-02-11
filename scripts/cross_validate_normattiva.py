#!/usr/bin/env python3
"""
Cross-validate normativa articles against Normattiva.it

Confronta gli articoli nel database con la fonte canonica Normattiva.it.
Aggiorna validated_at, validation_status, e canonical_hash.

Usage:
    cd lexe-max
    uv run python scripts/cross_validate_normattiva.py --codice CC --limit 100
    uv run python scripts/cross_validate_normattiva.py --codice CC --sample 0.05  # 5% random
"""

import asyncio
import argparse
import hashlib
import random
import re
from datetime import datetime
from pathlib import Path

import asyncpg
import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger()

# Database connection
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# Normattiva base URL
NORMATTIVA_BASE = "https://www.normattiva.it"

# URN prefixes for codici
URN_BASES = {
    "CC": ("regio.decreto", "1942-03-16", "262"),
    "CP": ("regio.decreto", "1930-10-19", "1398"),
    "CPC": ("regio.decreto", "1940-10-28", "1443"),
    "CPP": ("decreto.presidente.repubblica", "1988-09-22", "447"),
    "CDS": ("decreto.legislativo", "1992-04-30", "285"),
}


def normalize_text(text: str) -> str:
    """Normalizza testo per confronto."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    text = text.replace('–', '-').replace('—', '-')
    text = text.strip()
    return text


def compute_hash(text: str) -> str:
    """SHA256 del testo normalizzato."""
    return hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()


async def fetch_from_normattiva(
    client: httpx.AsyncClient,
    codice: str,
    articolo: str
) -> tuple[str | None, str | None]:
    """
    Fetch articolo da Normattiva.it.

    Returns:
        Tuple di (testo, url) o (None, None) se non trovato
    """
    if codice not in URN_BASES:
        logger.warning("Codice not supported for Normattiva", codice=codice)
        return None, None

    act_type, date, number = URN_BASES[codice]

    # Build URN - remove bis/ter from articolo number for URL
    art_num = articolo.lower().replace("-", "")

    # Normattiva URL format
    url = f"{NORMATTIVA_BASE}/uri-res/N2Ls?urn:nir:stato:{act_type}:{date};{number}~art{art_num}"

    try:
        response = await client.get(url, follow_redirects=True, timeout=30.0)

        if response.status_code != 200:
            logger.debug("Normattiva returned non-200", url=url, status=response.status_code)
            return None, None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try different selectors for article text
        selectors = [
            'div.art-text',
            'div.art-corpo',
            'div.articolo-testo',
            'div#articolo-testo',
            'div.corpo_articolo',
            'div.testo_articolo',
        ]

        content = None
        for sel in selectors:
            elem = soup.select_one(sel)
            if elem:
                content = elem.get_text(strip=True, separator=' ')
                break

        if not content:
            # Fallback: look for any div with substantial text
            for div in soup.find_all('div'):
                text = div.get_text(strip=True)
                if len(text) > 50 and 'art' in str(div.get('id', '')).lower():
                    content = text
                    break

        return content, str(response.url)

    except Exception as e:
        logger.error("Error fetching from Normattiva", url=url, error=str(e))
        return None, None


async def validate_article(
    conn: asyncpg.Connection,
    client: httpx.AsyncClient,
    row: dict
) -> dict:
    """
    Valida singolo articolo contro Normattiva.

    Returns:
        Dict con risultati validazione
    """
    codice = row['codice']
    articolo = row['articolo']
    db_text = row['testo']

    result = {
        'id': row['id'],
        'codice': codice,
        'articolo': articolo,
        'status': 'pending',
        'canonical_url': None,
        'canonical_hash': None,
        'hash_match': False,
        'diff_type': None,
    }

    # Fetch from Normattiva
    canonical_text, canonical_url = await fetch_from_normattiva(client, codice, articolo)

    if canonical_text is None:
        result['status'] = 'review_needed'
        result['diff_type'] = 'not_found_in_canonical'
        return result

    result['canonical_url'] = canonical_url

    # Compare hashes
    db_hash = compute_hash(db_text)
    canonical_hash = compute_hash(canonical_text)

    result['canonical_hash'] = canonical_hash
    result['hash_match'] = (db_hash == canonical_hash)

    if result['hash_match']:
        result['status'] = 'verified'
        result['diff_type'] = 'exact'
    else:
        # Check if it's just formatting difference
        db_words = set(normalize_text(db_text).split())
        can_words = set(normalize_text(canonical_text).split())

        # Jaccard similarity
        intersection = len(db_words & can_words)
        union = len(db_words | can_words)
        similarity = intersection / union if union > 0 else 0

        if similarity > 0.95:
            result['status'] = 'verified'
            result['diff_type'] = 'format_diff'
        elif similarity > 0.80:
            result['status'] = 'format_diff'
            result['diff_type'] = 'minor'
        else:
            result['status'] = 'content_diff'
            result['diff_type'] = 'substantive'

    return result


async def update_validation_status(conn: asyncpg.Connection, result: dict) -> None:
    """Aggiorna stato validazione nel database."""
    diff_summary = f"hash_match={result['hash_match']}, diff_type={result['diff_type']}"

    await conn.execute("""
        UPDATE kb.normativa SET
            validation_status = $1,
            validated_at = NOW(),
            canonical_url = $2,
            canonical_hash = $3,
            validation_diff = $4
        WHERE id = $5
    """, result['status'], result['canonical_url'], result['canonical_hash'],
         diff_summary, result['id'])


async def main():
    parser = argparse.ArgumentParser(description="Cross-validate with Normattiva")
    parser.add_argument("--codice", default="CC", help="Codice to validate")
    parser.add_argument("--limit", type=int, default=0, help="Max articles to validate (0=all)")
    parser.add_argument("--sample", type=float, default=0, help="Random sample rate (0-1)")
    parser.add_argument("--rps", type=float, default=1.0, help="Requests per second to Normattiva")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION WITH NORMATTIVA")
    print(f"  {datetime.now().isoformat()}")
    print(f"  Codice: {args.codice}")
    print(f"  Rate limit: {args.rps} req/sec")
    print(f"{'='*60}\n")

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("Connected to database")

    # Get articles to validate
    query = """
        SELECT id, codice, articolo, testo, validation_status
        FROM kb.normativa
        WHERE codice = $1
        ORDER BY articolo
    """
    rows = await conn.fetch(query, args.codice)
    print(f"Found {len(rows)} articles for {args.codice}")

    # Apply sampling if requested
    if args.sample > 0:
        n_sample = max(1, int(len(rows) * args.sample))
        rows = random.sample(rows, n_sample)
        print(f"Sampling {n_sample} articles ({args.sample*100:.1f}%)")
    elif args.limit > 0:
        rows = rows[:args.limit]
        print(f"Limited to {len(rows)} articles")

    # Stats
    stats = {
        'total': len(rows),
        'verified': 0,
        'format_diff': 0,
        'content_diff': 0,
        'review_needed': 0,
        'errors': 0,
    }

    start_time = datetime.now()
    delay = 1.0 / args.rps  # Seconds between requests

    async with httpx.AsyncClient() as client:
        for i, row in enumerate(rows, 1):
            try:
                result = await validate_article(conn, client, dict(row))

                stats[result['status']] = stats.get(result['status'], 0) + 1

                if not args.dry_run:
                    await update_validation_status(conn, result)

                # Progress
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(rows) - i) / rate if rate > 0 else 0

                status_icon = "=" if result['status'] == 'verified' else "~" if 'diff' in result['status'] else "!"
                print(f"  [{i}/{len(rows)}] {rate:.2f}/sec | ETA: {eta:.0f}s | {status_icon} Art. {row['articolo']}: {result['status']}")

                # Rate limiting
                await asyncio.sleep(delay)

            except Exception as e:
                stats['errors'] += 1
                logger.error("Validation error", articolo=row['articolo'], error=str(e))

    await conn.close()

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total validated: {stats['total']}")
    print(f"  Verified (exact): {stats['verified']}")
    print(f"  Format diff: {stats.get('format_diff', 0)}")
    print(f"  Content diff: {stats.get('content_diff', 0)}")
    print(f"  Review needed: {stats.get('review_needed', 0)}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
