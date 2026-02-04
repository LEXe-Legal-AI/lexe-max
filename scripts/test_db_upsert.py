#!/usr/bin/env python
"""
Test UPSERT senza embeddings.

Testa solo chunk + storage DB per verificare che la pipeline funzioni.
"""

import asyncio
import os
import sys
from pathlib import Path

# Set KB database URL BEFORE importing modules
os.environ.setdefault(
    "LEXE_KB_DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_upsert():
    """Test UPSERT su GDPR senza embeddings."""
    from lexe_api.kb.ingestion.altalex_pipeline import AltalexPipeline, PipelineConfig
    from lexe_api.kb.ingestion.altalex_store import AltalexStore
    from lexe_api.database import get_kb_pool

    # Find GDPR JSON
    json_dir = Path(__file__).parent.parent.parent / "altalex-md" / "batch"
    gdpr_files = [f for f in json_dir.rglob("*gdpr*.json") if not f.name.endswith("_meta.json")]

    if not gdpr_files:
        print(f"ERROR: GDPR JSON non trovato in {json_dir}")
        sys.exit(1)

    gdpr_json = gdpr_files[0]
    print(f"Found GDPR: {gdpr_json}")
    print(f"Size: {gdpr_json.stat().st_size // 1024} KB")
    print()

    print("=" * 60)
    print("TEST 1: Chunk + Store (no embedding)")
    print("=" * 60)

    try:
        # Get DB pool
        pool = await get_kb_pool()
        store = AltalexStore(pool)

        # Create pipeline with store but skip embed
        pipeline = AltalexPipeline(
            config=PipelineConfig(
                quarantine_on_error=True,
            ),
            store=store,
            db_pool=pool,
        )

        # Run pipeline skipping embedding
        result = await pipeline.run(
            json_path=gdpr_json,
            codice="GDPR",
            skip_embed=True,  # Skip embedding
            skip_store=False,  # Enable storage
        )

        print()
        print("=" * 60)
        print("RISULTATI")
        print("=" * 60)
        print(f"Stage: {result.stage.value}")
        print(f"Total articles: {result.stats.total_articles}")
        print(f"Valid: {result.stats.valid_articles}")
        print(f"Invalid: {result.stats.invalid_articles}")
        print(f"Stored: {result.stats.stored_articles}")
        print(f"Quarantine: {result.stats.quarantine_articles}")
        print()
        print(f"Chunk time: {result.stats.chunk_time_ms:.0f}ms")
        print(f"Validate time: {result.stats.validate_time_ms:.0f}ms")
        print(f"Store time: {result.stats.store_time_ms:.0f}ms")
        print(f"Total time: {result.stats.total_time_ms:.0f}ms")

        if result.error_message:
            print(f"\nERROR: {result.error_message}")

        # Get document stats from store
        stats = await store.get_document_stats("GDPR")
        print()
        print("DB Stats (GDPR):")
        print(f"  Total articles in DB: {stats.get('total_articles', 0)}")
        print(f"  Unique articles: {stats.get('unique_articles', 0)}")
        print(f"  Article range: {stats.get('min_article', '?')} - {stats.get('max_article', '?')}")
        print(f"  With text: {stats.get('with_text', 0)}")
        print(f"  With rubrica: {stats.get('with_rubrica', 0)}")

        await pipeline.close()

        # Verify data in DB
        print()
        print("=" * 60)
        print("TEST 2: Verify Data in DB")
        print("=" * 60)

        async with pool.acquire() as conn:
            # Sample row
            row = await conn.fetchrow(
                """
                SELECT id, codice, articolo, global_key, rubrica,
                       LEFT(testo, 100) as testo_preview,
                       articolo_num_norm, articolo_suffix
                FROM kb.normativa_altalex
                WHERE codice = 'GDPR'
                ORDER BY articolo_num_norm
                LIMIT 1
                """
            )
            if row:
                print(f"Sample article:")
                print(f"  ID: {row['id']}")
                print(f"  Global Key: {row['global_key']}")
                print(f"  Articolo: {row['articolo']}")
                print(f"  Num Norm: {row['articolo_num_norm']}")
                print(f"  Suffix: {row['articolo_suffix']}")
                print(f"  Rubrica: {row['rubrica'][:50] if row['rubrica'] else 'N/A'}...")
                print(f"  Testo: {row['testo_preview']}...")

            # Count
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM kb.normativa_altalex WHERE codice = 'GDPR'"
            )
            print(f"\nTotal GDPR articles in DB: {count}")

        # Test UPSERT (run again)
        print()
        print("=" * 60)
        print("TEST 3: UPSERT (re-run same file)")
        print("=" * 60)

        pipeline2 = AltalexPipeline(
            config=PipelineConfig(),
            store=store,
        )
        result2 = await pipeline2.run(
            json_path=gdpr_json,
            codice="GDPR",
            skip_embed=True,
            skip_store=False,
        )

        print(f"Stage: {result2.stage.value}")
        print(f"Second run - should be mostly UPDATEs, not INSERTs")

        await pipeline2.close()

        # Count again (should be same)
        async with pool.acquire() as conn:
            count2 = await conn.fetchval(
                "SELECT COUNT(*) FROM kb.normativa_altalex WHERE codice = 'GDPR'"
            )
            print(f"Total GDPR articles after second run: {count2}")

        if count == count2:
            print("[OK] UPSERT working correctly (count unchanged)")
        else:
            print(f"[WARNING] Count changed from {count} to {count2}")

        print()
        print("=" * 60)
        print("TEST COMPLETATO!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_upsert())
