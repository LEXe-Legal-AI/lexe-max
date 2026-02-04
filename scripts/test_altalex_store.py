#!/usr/bin/env python
"""
Test Altalex Store Pipeline

Test storage su GDPR (98 articoli) per verificare UPSERT.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexe_api.kb.ingestion.altalex_pipeline import (
    AltalexPipeline,
    PipelineConfig,
    create_pipeline_with_db,
)


async def test_store_gdpr():
    """Test storage su GDPR."""
    # Find GDPR JSON - search recursively in batch dir
    json_dir = Path(__file__).parent.parent.parent / "altalex-md" / "batch"
    gdpr_files = list(json_dir.rglob("*gdpr*.json"))

    # Exclude _meta.json files
    gdpr_files = [f for f in gdpr_files if not f.name.endswith("_meta.json")]

    if not gdpr_files:
        print("ERROR: GDPR JSON non trovato in altalex-md/batch/")
        print(f"Cercato in: {json_dir}")
        sys.exit(1)

    gdpr_json = gdpr_files[0]
    print(f"Found GDPR: {gdpr_json}")
    print(f"Size: {gdpr_json.stat().st_size // 1024} KB")
    print()

    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Check KB database URL (LEXE_ prefix used by pydantic settings)
    kb_url = os.environ.get("LEXE_KB_DATABASE_URL")
    if not kb_url:
        print("WARNING: LEXE_KB_DATABASE_URL not set, using default")
        print("Set with: export LEXE_KB_DATABASE_URL=postgresql://lexe_kb:password@localhost:5434/lexe_kb")
        # Try default
        os.environ["LEXE_KB_DATABASE_URL"] = "postgresql://lexe_kb:lexe_kb_dev@localhost:5434/lexe_kb"

    print("=" * 60)
    print("TEST 1: Pipeline con Storage")
    print("=" * 60)

    try:
        # Create pipeline with DB
        pipeline = await create_pipeline_with_db(
            config=PipelineConfig(
                embed_batch_size=50,
                db_batch_size=50,
                quarantine_on_error=True,
            )
        )

        # Run pipeline with storage enabled
        result = await pipeline.run(
            json_path=gdpr_json,
            codice="GDPR",
            skip_embed=False,
            skip_store=False,  # Enable storage!
        )

        print()
        print("=" * 60)
        print("RISULTATI")
        print("=" * 60)
        print(f"Stage: {result.stage.value}")
        print(f"Total articles: {result.stats.total_articles}")
        print(f"Valid: {result.stats.valid_articles}")
        print(f"Embedded: {result.stats.embedded_articles}")
        print(f"Stored: {result.stats.stored_articles}")
        print(f"Quarantine: {result.stats.quarantine_articles}")
        print()
        print(f"Chunk time: {result.stats.chunk_time_ms:.0f}ms")
        print(f"Validate time: {result.stats.validate_time_ms:.0f}ms")
        print(f"Embed time: {result.stats.embed_time_ms:.0f}ms")
        print(f"Store time: {result.stats.store_time_ms:.0f}ms")
        print(f"Total time: {result.stats.total_time_ms:.0f}ms")

        if result.error_message:
            print(f"\nERROR: {result.error_message}")

        # Get document stats from store
        if pipeline._store:
            stats = await pipeline._store.get_document_stats("GDPR")
            emb_count = await pipeline._store.count_embeddings("GDPR")
            print()
            print("DB Stats:")
            print(f"  Total articles in DB: {stats.get('total_articles', 0)}")
            print(f"  Unique articles: {stats.get('unique_articles', 0)}")
            print(f"  With text: {stats.get('with_text', 0)}")
            print(f"  With rubrica: {stats.get('with_rubrica', 0)}")
            print(f"  Embeddings: {emb_count}")

        await pipeline.close()

        # Test 2: Verify UPSERT (run again)
        print()
        print("=" * 60)
        print("TEST 2: UPSERT (re-run same file)")
        print("=" * 60)

        pipeline = await create_pipeline_with_db()
        result2 = await pipeline.run(
            json_path=gdpr_json,
            codice="GDPR",
            skip_embed=False,
            skip_store=False,
        )

        print(f"Stage: {result2.stage.value}")
        print(f"Stored (should be UPDATE, not INSERT): {result2.stats.stored_articles}")

        await pipeline.close()

        print()
        print("=" * 60)
        print("TEST COMPLETATO CON SUCCESSO!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_store_gdpr())
