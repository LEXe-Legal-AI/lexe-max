#!/usr/bin/env python3
"""
Test Ingestion Pipeline - Verifica su 10 sample files.

Testa la pipeline di ingestion sui file mirror StudioCataldi:
1. Deterministic Cleaner (HTML → testo pulito)
2. Structure Extractor (gerarchia + articoli)
3. URN Generator (URN:NIR)
4. Legal Numbers Extractor (citazioni)
5. StudioCataldi Adapter (full integration)

Usage:
    cd lexe-max
    uv run python scripts/test_ingestion_pipeline.py

    # Con path custom
    uv run python scripts/test_ingestion_pipeline.py --mirror-path /path/to/normativa
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, value: str, indent: int = 2) -> None:
    """Print result line."""
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")


async def test_deterministic_cleaner(sample_file: Path) -> dict:
    """Test HTML cleaner on a sample file."""
    from lexe_api.kb.ingestion.deterministic_cleaner import StudioCataldiCleaner

    print_header("TEST 1: Deterministic Cleaner")
    print(f"  File: {sample_file.name}")

    cleaner = StudioCataldiCleaner()
    html = sample_file.read_text(encoding="utf-8", errors="ignore")

    cleaned, metadata = cleaner.clean_article_page(html)

    print_result("Original size", f"{cleaned.original_size:,} bytes")
    print_result("Cleaned size", f"{cleaned.cleaned_size:,} bytes")
    print_result("Reduction", f"{cleaned.reduction_ratio:.1%}")
    print_result("Token estimate", f"~{cleaned.token_estimate:,}")
    print_result("Title", cleaned.title or "(none)")
    print_result("Metadata codice", metadata.get("codice", "(none)"))
    print_result("Metadata articolo", metadata.get("articolo", "(none)"))

    # Show first 200 chars of cleaned text
    preview = cleaned.text[:200].replace("\n", " ")
    print_result("Preview", f'"{preview}..."')

    return {
        "cleaned_text": cleaned.text,
        "metadata": metadata,
        "reduction": cleaned.reduction_ratio,
    }


async def test_structure_extractor(text: str, filename: str) -> dict:
    """Test structure extraction on cleaned text."""
    from lexe_api.kb.ingestion.structure_extractor import StructureExtractor

    print_header("TEST 2: Structure Extractor")

    extractor = StructureExtractor()
    structure = extractor.extract(text, filename)

    print_result("Document type", structure.doc_type.value)
    print_result("Codice", structure.codice or "(none)")
    print_result("Articles found", str(structure.article_count))
    print_result("Hierarchy nodes", str(len(structure.hierarchy)))

    if structure.articles:
        art = structure.articles[0]
        print_result("First article", f"Art. {art.articolo}")
        print_result("  Rubrica", art.rubrica or "(none)")
        print_result("  Commi", str(len(art.commi)))
        if art.libro:
            print_result("  Libro", art.libro)
        if art.titolo:
            print_result("  Titolo", art.titolo)

    return {
        "doc_type": structure.doc_type.value,
        "codice": structure.codice,
        "articles": structure.articles,
    }


async def test_urn_generator(codice: str, articolo: str) -> dict:
    """Test URN:NIR generation."""
    from lexe_api.kb.ingestion.urn_generator import URNGenerator, CanonicalIdGenerator

    print_header("TEST 3: URN Generator")

    urn_gen = URNGenerator()
    canonical_gen = CanonicalIdGenerator()

    # Generate URN
    urn = urn_gen.generate_for_codice(codice, articolo)
    canonical_id = canonical_gen.generate_for_article(codice, articolo)

    print_result("Codice", codice)
    print_result("Articolo", articolo)
    print_result("URN:NIR", urn or "(not supported)")
    print_result("Canonical ID", canonical_id)

    if urn:
        normattiva_url = urn_gen.get_normattiva_url(urn)
        print_result("Normattiva URL", normattiva_url)

    # Test parsing
    if urn:
        parsed = urn_gen.parse(urn)
        if parsed:
            print_result("Parsed authority", parsed.authority)
            print_result("Parsed act_type", parsed.act_type)
            print_result("Parsed article", parsed.article or "(none)")

    return {
        "urn": urn,
        "canonical_id": canonical_id,
    }


async def test_legal_numbers_extractor(text: str) -> dict:
    """Test legal numbers extraction."""
    from lexe_api.kb.ingestion.legal_numbers_extractor import (
        LegalNumbersExtractor,
        LegalNumberType,
    )

    print_header("TEST 4: Legal Numbers Extractor")

    extractor = LegalNumbersExtractor()
    result = extractor.extract(text)

    print_result("Numbers found", str(result.count))
    print_result("Unique canonical IDs", str(len(result.unique_canonical_ids)))

    # Count by type
    by_type = {}
    for num in result.numbers:
        type_name = num.number_type.value
        by_type[type_name] = by_type.get(type_name, 0) + 1

    for type_name, count in sorted(by_type.items()):
        print_result(f"  {type_name}", str(count))

    # Show first 5 numbers
    print("\n  First 5 numbers:")
    for num in result.numbers[:5]:
        print(f"    - {num.raw_text} -> {num.canonical_id}")

    return {
        "count": result.count,
        "by_type": by_type,
        "canonical_ids": list(result.unique_canonical_ids),
    }


async def test_studiocataldi_adapter(mirror_path: Path, codice: str = "COST") -> dict:
    """Test full adapter integration."""
    from lexe_api.kb.sources.studiocataldi_adapter import StudioCataldiAdapter

    print_header("TEST 5: StudioCataldi Adapter")
    print(f"  Mirror path: {mirror_path}")
    print(f"  Testing codice: {codice}")

    adapter = StudioCataldiAdapter(mirror_path)

    # List available codici
    available = await adapter.list_codici()
    print_result("Available codici", ", ".join(available) if available else "(none)")

    if codice not in available:
        if available:
            codice = available[0]
            print(f"  Switching to: {codice}")
        else:
            print("  ERROR: No codici found!")
            return {"error": "No codici found"}

    # Fetch first 5 articles
    articles = []

    def progress(current: int, total: int, msg: str | None) -> None:
        if current <= 5:
            print(f"    [{current}/{total}] {msg}")

    all_articles = await adapter.fetch_codice(codice, progress_callback=progress)

    print_result("Total articles", str(len(all_articles)))

    if all_articles:
        for art in all_articles[:3]:
            print(f"\n    Art. {art.articolo}:")
            print(f"      Rubrica: {art.rubrica or '(none)'}")
            print(f"      URN: {art.urn_nir or '(none)'}")
            print(f"      Text length: {len(art.testo)} chars")
            if art.citations_raw:
                print(f"      Citations: {len(art.citations_raw)}")

        articles = all_articles[:5]

    return {
        "available_codici": available,
        "tested_codice": codice,
        "article_count": len(all_articles),
        "sample_articles": [
            {
                "articolo": a.articolo,
                "rubrica": a.rubrica,
                "urn": a.urn_nir,
            }
            for a in articles
        ],
    }


async def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ingestion pipeline")
    parser.add_argument(
        "--mirror-path",
        type=Path,
        default=None,
        help="Path to StudioCataldi normativa/ folder",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LEXE KB - INGESTION PIPELINE TEST")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 60)

    # Find mirror path
    mirror_path = args.mirror_path
    if mirror_path is None:
        candidates = [
            Path("C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa"),
            Path("/opt/lexe-platform/data/studiocataldi/normativa"),
            Path.cwd() / "data" / "studiocataldi" / "normativa",
        ]
        for candidate in candidates:
            if candidate.exists():
                mirror_path = candidate
                break

    if mirror_path is None or not mirror_path.exists():
        print(f"\n  ERROR: Mirror path not found!")
        print(f"  Tried: {[str(c) for c in candidates]}")
        print(f"\n  Use --mirror-path to specify the path.")
        return

    print(f"\n  Mirror path: {mirror_path}")

    # Find sample file
    sample_files = list(mirror_path.glob("**/*art*.html"))[:10]
    if not sample_files:
        sample_files = list(mirror_path.glob("**/*.html"))[:10]

    if not sample_files:
        print("\n  ERROR: No HTML files found in mirror path!")
        return

    print(f"  Found {len(sample_files)} sample files")

    # Run tests on first file
    sample_file = sample_files[0]

    results = {}

    # Test 1: Cleaner
    try:
        results["cleaner"] = await test_deterministic_cleaner(sample_file)
    except Exception as e:
        print(f"\n  ERROR in cleaner test: {e}")
        results["cleaner"] = {"error": str(e)}

    # Test 2: Structure Extractor
    if "cleaned_text" in results.get("cleaner", {}):
        try:
            results["extractor"] = await test_structure_extractor(
                results["cleaner"]["cleaned_text"],
                sample_file.name,
            )
        except Exception as e:
            print(f"\n  ERROR in extractor test: {e}")
            results["extractor"] = {"error": str(e)}

    # Test 3: URN Generator
    codice = results.get("extractor", {}).get("codice") or "CC"
    articolo = "2043"  # Default famous article
    if results.get("extractor", {}).get("articles"):
        articolo = results["extractor"]["articles"][0].articolo

    try:
        results["urn"] = await test_urn_generator(codice, articolo)
    except Exception as e:
        print(f"\n  ERROR in URN test: {e}")
        results["urn"] = {"error": str(e)}

    # Test 4: Legal Numbers Extractor
    if "cleaned_text" in results.get("cleaner", {}):
        try:
            results["numbers"] = await test_legal_numbers_extractor(
                results["cleaner"]["cleaned_text"]
            )
        except Exception as e:
            print(f"\n  ERROR in numbers test: {e}")
            results["numbers"] = {"error": str(e)}

    # Test 5: Full Adapter
    try:
        results["adapter"] = await test_studiocataldi_adapter(mirror_path)
    except Exception as e:
        print(f"\n  ERROR in adapter test: {e}")
        import traceback
        traceback.print_exc()
        results["adapter"] = {"error": str(e)}

    # Summary
    print_header("SUMMARY")

    all_ok = all(
        "error" not in r for r in results.values()
    )

    if all_ok:
        print("  ✓ All tests passed!")
    else:
        print("  ✗ Some tests failed:")
        for name, result in results.items():
            if "error" in result:
                print(f"    - {name}: {result['error']}")

    # Stats
    if results.get("cleaner"):
        print(f"\n  Token reduction: {results['cleaner'].get('reduction', 0):.1%}")
    if results.get("numbers"):
        print(f"  Legal numbers found: {results['numbers'].get('count', 0)}")
    if results.get("adapter"):
        print(f"  Articles parsed: {results['adapter'].get('article_count', 0)}")

    print("\n" + "=" * 60)
    print("  Test completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
