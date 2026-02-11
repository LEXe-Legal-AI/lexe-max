#!/usr/bin/env python
"""
Batch Normativa Ingestion - Altalex PDFs

Processa tutti i JSON marker nella cartella batch/ con embeddings e storage.
Usa OpenRouter per embeddings (text-embedding-3-small).

Usage:
    # Set API key
    set OPENROUTER_API_KEY=sk-or-v1-xxx

    # Run batch
    python scripts/batch_normativa_ingestion.py

    # Skip already processed
    python scripts/batch_normativa_ingestion.py --skip-existing

    # Dry run (no DB writes)
    python scripts/batch_normativa_ingestion.py --dry-run
"""

import asyncio
import os
import re
import sys
from pathlib import Path

# Set KB database URL BEFORE importing modules
os.environ.setdefault(
    "LEXE_KB_DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mapping nome file -> codice
CODICE_MAP = {
    "gdpr": "GDPR",
    "codice-civile": "CC",
    "codice-penale": "CP",
    "codice-procedura-civile": "CPC",
    "codice-procedura-penale": "CPP",
    "costituzione": "COST",
    "codice-ambiente": "CAMB",
    "codice-amministrazione-digitale": "CAD",
    "codice-antimafia": "CAMAFIA",
    "codice-appalti": "CAPP",
    "codice-beni-culturali": "CBC",
    "codice-crisi-impresa": "CCI",
    "codice-deontologico-forense": "CDF",
    "codice-giustizia-contabile": "CGC",
    "codice-giustizia-sportiva": "CGS",
    "codice-medicinali": "CMED",
    "codice-nautica-diporto": "CND",
    "codice-pari-opportunita": "CPO",
    "codice-processo-amministrativo": "CPA",
    "codice-processo-penale-minorile": "CPPM",
    "codice-proprieta-industriale": "CPI",
    "codice-strada": "CDS",
    "codice-terzo-settore": "CTS",
    "codice-turismo": "CTUR",
    "dichiarazione-universale": "DUDU",
    "legge-divorzio": "LDIV",
    "legge-fallimentare": "LFALL",
    "legge-locazioni": "LLOC",
    "legge-procedimento-amministrativo": "L241",
    "legge-professionale-forense": "LPF",
    "legge-reati-tributari": "LRT",
    "legge-sciopero": "LSCIO",
    "legge-depenalizzazione": "LDEP",
    "legge-diritto-internazionale": "LDIP",
    "mediazione-civile": "LMED",
    "ordinamento-penitenziario": "OP",
    "responsabilita-amministrativa": "D231",
    "riforma-lavoro-biagi": "RBIAGI",
    "riforma-lavoro-fornero": "RFORN",
    "sicurezza-urbana": "LSIC",
    "statuto-contribuente": "SCONTR",
    "statuto-lavoratori": "SLAV",
    "testo-unico-casellario": "TUCAS",
    "testo-unico-documentazione": "TUDOC",
    "testo-unico-edilizia": "TUE",
    "testo-unico-enti-locali": "TUEL",
    "testo-unico-espropriazioni": "TUESP",
    "testo-unico-foreste": "TUF",
    "testo-unico-immigrazione": "TUI",
    "testo-unico-maternita": "TUMAT",
    "testo-unico-previdenza": "TUPREV",
    "testo-unico-radiotelevisione": "TURTV",
    "testo-unico-societa-partecipate": "TUSP",
    "regolamento-esecuzione": "REGCDS",
}


def extract_codice_from_path(json_path: Path) -> str:
    """Estrae codice dal nome file usando mapping."""
    name = json_path.parent.name.lower()

    # Try exact match first
    for key, codice in CODICE_MAP.items():
        if key in name:
            return codice

    # Fallback: use first word capitalized
    words = re.findall(r'[a-z]+', name)
    if words:
        return words[0].upper()[:10]

    return "UNKNOWN"


async def get_existing_codici() -> set[str]:
    """Get list of codici already in database."""
    from lexe_api.database import get_kb_pool

    pool = await get_kb_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT codice FROM kb.normativa_altalex"
        )
    return {row['codice'] for row in rows}


async def run_batch(skip_existing: bool = False, dry_run: bool = False):
    """Run batch ingestion on all JSON files."""
    from lexe_api.kb.ingestion.altalex_pipeline import (
        AltalexPipeline,
        PipelineConfig,
    )
    from lexe_api.kb.ingestion.altalex_store import AltalexStore
    from lexe_api.database import get_kb_pool

    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Set with: set OPENROUTER_API_KEY=sk-or-v1-xxx")
        sys.exit(1)

    # Find all JSON files
    batch_dir = Path(__file__).parent.parent.parent / "altalex-md" / "batch"
    json_files = []
    for subdir in batch_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.json"):
                if not f.name.endswith("_meta.json"):
                    json_files.append(f)

    print(f"Found {len(json_files)} JSON files to process")
    print()

    # Get existing codici if skipping
    existing_codici = set()
    if skip_existing:
        existing_codici = await get_existing_codici()
        print(f"Skipping {len(existing_codici)} already processed codici: {sorted(existing_codici)}")
        print()

    # Setup
    pool = await get_kb_pool()
    store = None if dry_run else AltalexStore(pool)

    results = []
    total_articles = 0
    total_embedded = 0
    total_stored = 0
    total_errors = 0

    for i, json_path in enumerate(sorted(json_files), 1):
        codice = extract_codice_from_path(json_path)

        # Skip if exists
        if codice in existing_codici:
            print(f"[{i}/{len(json_files)}] SKIP {codice} (already exists)")
            continue

        print("=" * 70)
        print(f"[{i}/{len(json_files)}] Processing: {json_path.parent.name}")
        print(f"Codice: {codice}")
        print("=" * 70)

        try:
            pipeline = AltalexPipeline(
                config=PipelineConfig(
                    embed_batch_size=50,
                    db_batch_size=100,
                    quarantine_on_error=True,
                ),
                store=store,
                db_pool=pool,
            )

            result = await pipeline.run(
                json_path=json_path,
                codice=codice,
                skip_embed=False,
                skip_store=dry_run,
            )

            results.append({
                "codice": codice,
                "file": json_path.name,
                "stage": result.stage.value,
                "articles": result.stats.total_articles,
                "valid": result.stats.valid_articles,
                "embedded": result.stats.embedded_articles,
                "stored": result.stats.stored_articles,
                "errors": result.stats.quarantine_articles,
            })

            total_articles += result.stats.total_articles
            total_embedded += result.stats.embedded_articles
            total_stored += result.stats.stored_articles
            total_errors += result.stats.quarantine_articles

            print(f"Result: {result.stage.value}")
            print(f"  Articles: {result.stats.total_articles}")
            print(f"  Embedded: {result.stats.embedded_articles}")
            print(f"  Stored: {result.stats.stored_articles}")
            print(f"  Time: {result.stats.total_time_ms:.0f}ms")

            if result.error_message:
                print(f"  ERROR: {result.error_message}")

            await pipeline.close()

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            total_errors += 1
            results.append({
                "codice": codice,
                "file": json_path.name,
                "stage": "failed",
                "error": str(e),
            })

        print()

    # Summary
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Total files processed: {len(results)}")
    print(f"Total articles: {total_articles}")
    print(f"Total embedded: {total_embedded}")
    print(f"Total stored: {total_stored}")
    print(f"Total errors: {total_errors}")
    print()

    # Results table
    print("Results by document:")
    print("-" * 70)
    for r in results:
        status = "[OK]" if r.get("stage") == "complete" else "[FAIL]"
        print(f"  {status} {r['codice']}: {r.get('stored', 0)} stored, {r.get('errors', 0)} errors")

    # Final DB stats
    if not dry_run:
        async with pool.acquire() as conn:
            total_count = await conn.fetchval(
                "SELECT COUNT(*) FROM kb.normativa_altalex"
            )
            emb_count = await conn.fetchval(
                "SELECT COUNT(*) FROM kb.altalex_embeddings"
            )
        print()
        print("Database totals:")
        print(f"  Total articles in DB: {total_count}")
        print(f"  Total embeddings in DB: {emb_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Normativa Ingestion")
    parser.add_argument("--skip-existing", action="store_true", help="Skip codici already in DB")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    asyncio.run(run_batch(skip_existing=args.skip_existing, dry_run=args.dry_run))
