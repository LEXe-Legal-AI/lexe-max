#!/usr/bin/env python
"""
Reprocess specific codici with fixed MarkerChunker.

After fixing the chunker patterns, reprocess JSON files to extract
previously missed articles and update the database.

Usage:
    cd lexe-max
    uv run python scripts/reprocess_fixed_chunker.py CCI CAMB
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg

from lexe_api.kb.ingestion.marker_chunker import MarkerChunker
from lexe_api.kb.ingestion.altalex_store import AltalexStore

DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

BATCH_DIR = Path(__file__).parent.parent.parent / "altalex-md" / "batch"

# Mapping codice -> folder name pattern
CODICE_FOLDERS = {
    "CCI": "codice-crisi-impresa",
    "CAMB": "codice-ambiente",
    "CC": "codice-civile",
    "CP": "codice-penale",
    "CPC": "codice-procedura-civile",
    "CPP": "codice-procedura-penale",
}


def find_json_for_codice(codice: str) -> Path | None:
    """Find JSON file for codice."""
    pattern = CODICE_FOLDERS.get(codice)
    if not pattern:
        print(f"Unknown codice: {codice}")
        return None

    for folder in BATCH_DIR.iterdir():
        if folder.is_dir() and pattern in folder.name.lower():
            json_files = list(folder.glob("*.json"))
            for jf in json_files:
                if not jf.name.endswith("_meta.json"):
                    return jf
    return None


async def reprocess_codice(
    codice: str,
    conn: asyncpg.Connection,
    store: AltalexStore,
    dry_run: bool = False,
) -> dict:
    """
    Reprocess a codice with the fixed chunker.

    Returns stats dict.
    """
    json_path = find_json_for_codice(codice)
    if not json_path:
        return {"error": f"JSON not found for {codice}"}

    print(f"\n{'='*60}")
    print(f"Reprocessing {codice}")
    print(f"JSON: {json_path.name}")
    print(f"{'='*60}")

    # Get current counts
    before = await conn.fetchrow("""
        SELECT COUNT(*) as articles,
               COUNT(DISTINCT articolo_num_norm) as distinct_nums
        FROM kb.normativa_altalex
        WHERE codice = $1
    """, codice)

    print(f"Before: {before['articles']} articles, {before['distinct_nums']} distinct nums")

    # Process with fixed chunker
    chunker = MarkerChunker()
    articles = chunker.process_file(json_path, codice)

    print(f"Chunker extracted: {len(articles)} articles")

    # Count distinct article numbers
    distinct = len(set(a.articolo_num_norm for a in articles if a.articolo_num_norm))
    print(f"Distinct nums in extraction: {distinct}")

    if dry_run:
        print("DRY RUN - Not storing to DB")
        return {
            "codice": codice,
            "before_articles": before['articles'],
            "before_nums": before['distinct_nums'],
            "extracted_articles": len(articles),
            "extracted_nums": distinct,
            "dry_run": True,
        }

    # Store articles (without embeddings for now - they'll need regeneration)
    # The store uses UPSERT so existing articles will be updated
    articles_with_embeddings = [(a, None) for a in articles]

    result = await store.store_batch(
        articles=articles_with_embeddings,
        codice=codice,
        embedding_model=None,  # No embedding for now
        source_file=str(json_path),
        batch_size=100,
    )

    print(f"Store result: {result.inserted} inserted, {result.updated} updated, {result.failed} failed")

    # Get after counts
    after = await conn.fetchrow("""
        SELECT COUNT(*) as articles,
               COUNT(DISTINCT articolo_num_norm) as distinct_nums
        FROM kb.normativa_altalex
        WHERE codice = $1
    """, codice)

    print(f"After: {after['articles']} articles, {after['distinct_nums']} distinct nums")
    print(f"Delta: +{after['articles'] - before['articles']} articles, +{after['distinct_nums'] - before['distinct_nums']} nums")

    return {
        "codice": codice,
        "before_articles": before['articles'],
        "before_nums": before['distinct_nums'],
        "after_articles": after['articles'],
        "after_nums": after['distinct_nums'],
        "delta_articles": after['articles'] - before['articles'],
        "delta_nums": after['distinct_nums'] - before['distinct_nums'],
        "inserted": result.inserted,
        "updated": result.updated,
        "failed": result.failed,
    }


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess codici with fixed chunker")
    parser.add_argument("codici", nargs="+", help="Codici to reprocess (CCI, CAMB, etc.)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
    conn = await pool.acquire()
    store = AltalexStore(pool)

    results = []
    for codice in args.codici:
        result = await reprocess_codice(codice, conn, store, dry_run=args.dry_run)
        results.append(result)

    await pool.release(conn)
    await pool.close()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"{r.get('codice', 'UNK')}: ERROR - {r['error']}")
        elif r.get("dry_run"):
            print(f"{r['codice']}: {r['before_articles']} -> {r['extracted_articles']} articles (DRY RUN)")
        else:
            print(f"{r['codice']}: {r['before_articles']} -> {r['after_articles']} articles (+{r['delta_articles']})")


if __name__ == "__main__":
    asyncio.run(main())
