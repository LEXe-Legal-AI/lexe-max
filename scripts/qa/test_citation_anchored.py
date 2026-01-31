#!/usr/bin/env python3
"""
Test Citation-Anchored Extraction

Testa la nuova modalità di estrazione sui documenti Civile problematici.

Usage:
    uv run python scripts/qa/test_citation_anchored.py
    uv run python scripts/qa/test_citation_anchored.py --doc "Volume I_2016"
"""

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lexe_api.kb.ingestion.massima_extractor import (
    extract_massime_citation_anchored,
    extract_massime_from_pdf_text,
    find_citation_anchors,
)
from qa_config import PDF_DIR

# Documenti Civile problematici (coverage < 15%)
CIVILE_CRITICAL = [
    "Volume I_2016_Massimario_Civile_1_372.pdf",
    "Volume I_2017_Massimario_Civile_1_372.pdf",
    "2014 Mass civile Vol 1 pagg 408.pdf",
    "Volume II_2024_Massimario_Civile(volume completo).pdf",
    "Volume II_2023_Massimario_Civile(volume completo).pdf",
]


def test_single_page(text: str, page_num: int) -> dict:
    """Test citation-anchored extraction on a single page."""
    anchors = find_citation_anchors(text)

    massime = extract_massime_citation_anchored(
        text=text,
        page_number=page_num,
        window_before=2,
        window_after=1,
        split_on_multiple=True,
        min_length=120,
        max_length=3000,
    )

    return {
        "page": page_num,
        "text_len": len(text),
        "n_anchors": len(anchors),
        "n_massime": len(massime),
        "anchors": [a.match_text[:50] for a in anchors[:5]],
        "massime_preview": [m.testo[:100] + "..." for m in massime[:3]],
    }


def test_document(pdf_path: Path, toc_skip: int = 20) -> dict:
    """Test citation-anchored extraction on full document."""
    doc = fitz.open(pdf_path)

    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        pages.append((i + 1, text))

    doc.close()

    # Gate config per Civile
    gate_config = {
        "min_length": 120,
        "max_length": 2500,
        "citation_window_before": 2,
        "citation_window_after": 1,
        "split_on_multiple_citations": True,
    }

    massime = extract_massime_from_pdf_text(
        pages=pages,
        extraction_mode="citation_anchored",
        toc_skip_pages=toc_skip,
        gate_config=gate_config,
    )

    # Stats per pagina
    page_stats = {}
    for m in massime:
        pg = m.page_start or 0
        if pg not in page_stats:
            page_stats[pg] = 0
        page_stats[pg] += 1

    return {
        "filename": pdf_path.name,
        "total_pages": len(pages),
        "toc_skip": toc_skip,
        "pages_processed": len(pages) - toc_skip,
        "n_massime": len(massime),
        "massime_per_page": len(massime) / max(1, len(pages) - toc_skip),
        "with_complete_citation": sum(1 for m in massime if m.citation_complete),
        "page_distribution": dict(sorted(page_stats.items())[:10]),
    }


def main(doc_filter: str = None):
    print("=" * 80)
    print("TEST CITATION-ANCHORED EXTRACTION")
    print("=" * 80)
    print()

    # Filter documents
    if doc_filter:
        test_docs = [d for d in CIVILE_CRITICAL if doc_filter.lower() in d.lower()]
        if not test_docs:
            # Try to find by partial match in PDF_DIR
            for pdf in PDF_DIR.glob("*.pdf"):
                if doc_filter.lower() in pdf.name.lower():
                    test_docs.append(pdf.name)
    else:
        test_docs = CIVILE_CRITICAL[:2]  # Test first 2 by default

    print(f"Documenti da testare: {len(test_docs)}")
    print()

    for filename in test_docs:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename

        if not pdf_path.exists():
            print(f"[SKIP] {filename} non trovato")
            continue

        print("=" * 80)
        print(f"DOCUMENTO: {filename}")
        print("=" * 80)

        # Test single page first (page 50 to skip TOC)
        print("\n  TEST PAGINA SINGOLA (p. 50):")
        doc = fitz.open(pdf_path)
        if len(doc) > 50:
            page_text = doc[49].get_text()
            single_result = test_single_page(page_text, 50)
            print(f"    Text length: {single_result['text_len']}")
            print(f"    Anchors found: {single_result['n_anchors']}")
            print(f"    Massime extracted: {single_result['n_massime']}")
            if single_result['anchors']:
                print(f"    Sample anchors: {single_result['anchors'][:3]}")
        doc.close()

        # Test full document
        print("\n  TEST DOCUMENTO COMPLETO:")
        result = test_document(pdf_path, toc_skip=25)

        print(f"    Total pages: {result['total_pages']}")
        print(f"    TOC skip: {result['toc_skip']}")
        print(f"    Pages processed: {result['pages_processed']}")
        print(f"    Massime extracted: {result['n_massime']}")
        print(f"    Massime/page: {result['massime_per_page']:.2f}")
        print(f"    With complete citation: {result['with_complete_citation']}")

        # Compare with current
        current_massime = {
            "Volume I_2016_Massimario_Civile_1_372.pdf": 43,
            "Volume I_2017_Massimario_Civile_1_372.pdf": 41,
            "2014 Mass civile Vol 1 pagg 408.pdf": 47,
            "Volume II_2024_Massimario_Civile(volume completo).pdf": 28,
            "Volume II_2023_Massimario_Civile(volume completo).pdf": 35,
        }

        if filename in current_massime:
            current = current_massime[filename]
            improvement = result['n_massime'] / current if current > 0 else 0
            print(f"\n    [COMPARISON]")
            print(f"      Current massime: {current}")
            print(f"      New massime: {result['n_massime']}")
            print(f"      Improvement: {improvement:.1f}x")

        print()

    print("=" * 80)
    print("[TEST COMPLETATO]")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test citation-anchored extraction")
    parser.add_argument("--doc", type=str, help="Filter by document name")
    args = parser.parse_args()

    main(doc_filter=args.doc)
