#!/usr/bin/env python3
"""Cross-validate KB normativa against Altalex MD exports.

Confronta gli articoli nel database (da Brocardi) con Altalex.
Altalex è una fonte editoriale autorevole, utile per validazione locale
senza dipendere da Normattiva scraping.

Usage:
    cd lexe-max

    # Test 5 articoli specifici
    uv run python scripts/cross_validate_altalex.py \
        --altalex "C:/Users/Fra/Downloads/Edizione 2025 Altalex.md" \
        --codice CC --articles 1,2043,2059,1218,1321

    # Validazione completa (tutti gli articoli)
    uv run python scripts/cross_validate_altalex.py \
        --altalex "C:/Users/Fra/Downloads/Edizione 2025 Altalex.md" \
        --codice CC

    # Sample 5%
    uv run python scripts/cross_validate_altalex.py \
        --altalex "C:/Users/Fra/Downloads/Edizione 2025 Altalex.md" \
        --codice CC --sample 0.05
"""

import argparse
import asyncio
import random
from datetime import datetime
from pathlib import Path

import asyncpg
import structlog

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexe_api.kb.sources.altalex_adapter import AltalexAdapter, AltalexArticle

logger = structlog.get_logger()

# Database connection
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"


async def get_db_articles(conn: asyncpg.Connection, codice: str) -> dict[str, dict]:
    """Get all articles from DB for a codice."""
    rows = await conn.fetch("""
        SELECT id, codice, articolo, rubrica, testo, validation_status
        FROM kb.normativa
        WHERE codice = $1
        ORDER BY articolo
    """, codice)

    return {row['articolo']: dict(row) for row in rows}


async def update_validation_status(
    conn: asyncpg.Connection,
    article_id: str,
    result: dict,
) -> None:
    """Update validation status in database."""
    diff_summary = (
        f"altalex_match: hash={result['hash_match']}, "
        f"similarity={result['similarity']:.2%}, "
        f"diff_type={result['diff_type']}"
    )

    await conn.execute("""
        UPDATE kb.normativa SET
            validation_status = $1,
            validated_at = NOW(),
            validation_diff = COALESCE(validation_diff, '') || E'\n' || $2
        WHERE id = $3
    """, result['status'], diff_summary, article_id)


async def main():
    parser = argparse.ArgumentParser(description="Cross-validate with Altalex MD")
    parser.add_argument(
        "--altalex",
        required=True,
        help="Path to Altalex MD file"
    )
    parser.add_argument("--codice", default="CC", help="Codice to validate")
    parser.add_argument(
        "--articles",
        help="Comma-separated list of specific articles to validate"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0,
        help="Random sample rate (0-1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max articles to validate (0=all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update database"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION: ALTALEX vs BROCARDI (DB)")
    print(f"  {datetime.now().isoformat()}")
    print(f"  Codice: {args.codice}")
    print(f"  Altalex: {args.altalex}")
    print(f"{'='*60}\n")

    # Parse Altalex file
    adapter = AltalexAdapter()
    altalex_path = Path(args.altalex)

    if not altalex_path.exists():
        print(f"ERROR: File not found: {altalex_path}")
        return

    print("Parsing Altalex MD file...")
    altalex_articles = adapter.parse_and_cache(altalex_path, codice=args.codice)
    altalex_by_num = {art.articolo: art for art in altalex_articles}
    print(f"  Found {len(altalex_articles)} articles in Altalex")

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("Connected to database")

    # Get DB articles
    db_articles = await get_db_articles(conn, args.codice)
    print(f"  Found {len(db_articles)} articles in DB (Brocardi)")

    # Determine which articles to validate
    if args.articles:
        # Specific articles
        article_nums = [a.strip() for a in args.articles.split(',')]
        articles_to_validate = [
            (num, db_articles.get(num))
            for num in article_nums
            if num in db_articles
        ]
        print(f"  Validating {len(articles_to_validate)} specific articles")
    else:
        # All or sampled
        articles_to_validate = list(db_articles.items())

        if args.sample > 0:
            n_sample = max(1, int(len(articles_to_validate) * args.sample))
            articles_to_validate = random.sample(articles_to_validate, n_sample)
            print(f"  Sampling {n_sample} articles ({args.sample*100:.1f}%)")
        elif args.limit > 0:
            articles_to_validate = articles_to_validate[:args.limit]
            print(f"  Limited to {len(articles_to_validate)} articles")

    # Stats
    stats = {
        'total': len(articles_to_validate),
        'verified': 0,
        'format_diff': 0,
        'content_diff': 0,
        'not_in_altalex': 0,
        'errors': 0,
    }

    results_detail = []

    print(f"\nValidating {stats['total']} articles...\n")

    for i, (art_num, db_row) in enumerate(articles_to_validate, 1):
        try:
            altalex_art = altalex_by_num.get(art_num)

            if altalex_art is None:
                stats['not_in_altalex'] += 1
                icon = "??"
                status = "not_in_altalex"
                result = {'status': 'review_needed', 'diff_type': 'not_in_altalex'}
            else:
                # Compare
                result = adapter.compare_with_brocardi(
                    altalex_art,
                    db_row['testo']
                )
                status = result['status']
                stats[status] = stats.get(status, 0) + 1

                if result['hash_match']:
                    icon = "OK"
                elif result['similarity'] > 0.9:
                    icon = "~~"
                else:
                    icon = "!!"

            # Update DB
            if not args.dry_run and db_row:
                await update_validation_status(conn, db_row['id'], result)

            # Progress
            print(f"  [{i}/{stats['total']}] {icon} Art. {art_num}: {status}", end="")
            if 'similarity' in result:
                print(f" ({result['similarity']:.1%} sim)")
            else:
                print()

            results_detail.append({
                'articolo': art_num,
                'status': status,
                **result,
            })

        except Exception as e:
            stats['errors'] += 1
            logger.error("Validation error", articolo=art_num, error=str(e))

    await conn.close()

    # Final stats
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total validated: {stats['total']}")
    print(f"  [OK] Verified (exact/format): {stats['verified']}")
    print(f"  [~~] Format diff: {stats.get('format_diff', 0)}")
    print(f"  [!!] Content diff: {stats.get('content_diff', 0)}")
    print(f"  [??] Not in Altalex: {stats['not_in_altalex']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Show content diffs details
    content_diffs = [r for r in results_detail if r.get('diff_type') == 'substantive']
    if content_diffs:
        print("Content differences found:")
        for r in content_diffs[:5]:
            print(f"  - Art. {r['articolo']}: {r.get('similarity', 0):.1%} similarity")
            print(f"    Altalex: {r.get('altalex_len', 0)} chars, Brocardi: {r.get('brocardi_len', 0)} chars")


if __name__ == "__main__":
    asyncio.run(main())
