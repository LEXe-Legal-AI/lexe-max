#!/usr/bin/env python3
"""
Quick CdS Ingestion Test - Run from mirror.

Tests the full pipeline on Codice della Strada (419 articles).
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexe_api.kb.sources.studiocataldi_adapter import StudioCataldiAdapter
from lexe_api.kb.ingestion.legal_numbers_extractor import extract_canonical_ids

MIRROR_PATH = Path("C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa")


async def main():
    print(f"\n{'='*60}")
    print("  CdS INGESTION TEST")
    print(f"  {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    adapter = StudioCataldiAdapter(MIRROR_PATH)

    # Check available codici
    available = await adapter.list_codici()
    print(f"Available codici: {available}")

    if "CDS" not in available:
        print("ERROR: CDS not found!")
        # Try direct path
        cds_path = MIRROR_PATH / "codicedellastrada" / "commentato"
        if cds_path.exists():
            print(f"Found CDS at: {cds_path}")

    # Fetch all CdS articles
    start = datetime.now()
    articles = []
    total_citations = set()
    errors = []

    def progress(current: int, total: int, msg: str | None):
        if current % 50 == 0 or current == total:
            elapsed = (datetime.now() - start).total_seconds()
            rate = current / elapsed if elapsed > 0 else 0
            print(f"  [{current}/{total}] {rate:.1f} art/sec - {msg}")

    print("\nFetching CdS articles...")

    try:
        # Manual fetch since adapter might not find CDS
        cds_commentato = MIRROR_PATH / "codicedellastrada" / "commentato"
        html_files = sorted(cds_commentato.glob("*.html"))
        total = len(html_files)
        print(f"Found {total} HTML files")

        from lexe_api.kb.ingestion.deterministic_cleaner import StudioCataldiCleaner
        from lexe_api.kb.ingestion.structure_extractor import StructureExtractor
        from lexe_api.kb.ingestion.urn_generator import URNGenerator

        cleaner = StudioCataldiCleaner()
        extractor = StructureExtractor()
        urn_gen = URNGenerator()

        for i, html_file in enumerate(html_files):
            try:
                # Clean HTML
                html = html_file.read_text(encoding="utf-8", errors="ignore")
                cleaned, metadata = cleaner.clean_article_page(html)

                if not cleaned.text or len(cleaned.text) < 20:
                    continue

                # Extract structure
                structure = extractor.extract_single_article(cleaned.text)

                # Extract article number from filename
                import re
                match = re.search(r"art[^\d]*(\d+)", html_file.stem, re.I)
                art_num = match.group(1) if match else str(i)

                # Extract citations
                citations = extract_canonical_ids(cleaned.text)
                total_citations.update(citations)

                # Generate URN
                urn = urn_gen.generate_for_codice("CDS", art_num)

                articles.append({
                    "articolo": art_num,
                    "rubrica": structure.rubrica if structure else None,
                    "urn": urn,
                    "text_length": len(cleaned.text),
                    "citations_count": len(citations),
                    "file": html_file.name,
                })

                if (i + 1) % 50 == 0:
                    progress(i + 1, total, f"Art. {art_num}")

            except Exception as e:
                errors.append({"file": html_file.name, "error": str(e)})

        progress(len(html_files), total, "Done!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Stats
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  Articles processed: {len(articles)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Rate: {len(articles)/elapsed:.1f} articles/sec")
    print(f"  Unique citations found: {len(total_citations)}")

    # Sample citations
    print(f"\n  Sample citations (first 10):")
    for cit in sorted(total_citations)[:10]:
        print(f"    - {cit}")

    # Sample articles
    print(f"\n  Sample articles (first 5):")
    for art in articles[:5]:
        print(f"    Art. {art['articolo']}: {art['rubrica'] or '(no rubrica)'}")
        print(f"      URN: {art['urn']}")
        print(f"      Text: {art['text_length']} chars, {art['citations_count']} citations")

    # Save results
    output_file = Path(__file__).parent / "cds_ingestion_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "articles": len(articles),
                "errors": len(errors),
                "elapsed_seconds": elapsed,
                "unique_citations": len(total_citations),
            },
            "citations": sorted(total_citations),
            "articles": articles,
            "errors": errors,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
