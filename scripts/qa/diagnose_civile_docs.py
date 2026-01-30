#!/usr/bin/env python3
"""
Diagnosi Civile Documents - Preflight per rielaborazione

Analizza ogni documento Civile problematico e genera report con:
- toc_page_ratio e toc_page_ranges
- citation density per pagina
- chunks candidati vs scartati (se disponibili)
- raccomandazione profilo e gate

Usage:
    uv run python scripts/qa/diagnose_civile_docs.py
    uv run python scripts/qa/diagnose_civile_docs.py --doc "Volume I_2016"
"""

import argparse
import asyncio
import re
from pathlib import Path

import asyncpg
import fitz  # PyMuPDF

from qa_config import DB_URL, PDF_DIR

# Citation patterns
CITATION_PATTERNS = {
    "rv": re.compile(r"Rv\.?\s*\d{6}", re.IGNORECASE),
    "sez": re.compile(r"Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI|L|Lav)", re.IGNORECASE),
    "cass": re.compile(r"Cass\.?\s*(?:civ|pen)?\.?\s*\d+/\d+", re.IGNORECASE),
    "sent": re.compile(r"(?:sent|ord)\.?\s*n\.?\s*\d+", re.IGNORECASE),
}

# TOC patterns
TOC_PATTERNS = [
    re.compile(r"^INDICE", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Indice\s+generale", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^SOMMARIO", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Capitolo\s+[IVX]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE),  # numbered sections
    re.compile(r"\.{3,}\s*\d+$", re.MULTILINE),  # dotted page numbers
]


def analyze_pdf(pdf_path: Path) -> dict:
    """Analyze a PDF for TOC and citation patterns."""
    doc = fitz.open(pdf_path)

    results = {
        "filename": pdf_path.name,
        "total_pages": len(doc),
        "pages": [],
        "toc_pages": [],
        "citation_pages": [],
        "empty_pages": [],
    }

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        page_num = i + 1

        page_info = {
            "page": page_num,
            "char_count": len(text),
            "is_toc": False,
            "citations": {},
            "toc_signals": 0,
        }

        # Check for empty/near-empty pages
        if len(text.strip()) < 100:
            results["empty_pages"].append(page_num)
            page_info["is_empty"] = True

        # Check for TOC patterns
        toc_signals = 0
        for pattern in TOC_PATTERNS:
            matches = len(pattern.findall(text))
            toc_signals += matches

        page_info["toc_signals"] = toc_signals
        if toc_signals >= 3 or (toc_signals >= 1 and page_num <= 30):
            page_info["is_toc"] = True
            results["toc_pages"].append(page_num)

        # Count citations
        total_citations = 0
        for name, pattern in CITATION_PATTERNS.items():
            count = len(pattern.findall(text))
            page_info["citations"][name] = count
            total_citations += count

        page_info["total_citations"] = total_citations
        if total_citations > 0:
            results["citation_pages"].append(page_num)

        results["pages"].append(page_info)

    doc.close()

    # Compute summary stats
    results["toc_page_ratio"] = len(results["toc_pages"]) / results["total_pages"] if results["total_pages"] > 0 else 0
    results["citation_page_ratio"] = len(results["citation_pages"]) / results["total_pages"] if results["total_pages"] > 0 else 0
    results["total_citations"] = sum(p["total_citations"] for p in results["pages"])
    results["citations_per_page"] = results["total_citations"] / results["total_pages"] if results["total_pages"] > 0 else 0

    # Identify TOC ranges (consecutive TOC pages)
    toc_ranges = []
    if results["toc_pages"]:
        start = results["toc_pages"][0]
        end = start
        for p in results["toc_pages"][1:]:
            if p == end + 1:
                end = p
            else:
                toc_ranges.append((start, end))
                start = p
                end = p
        toc_ranges.append((start, end))
    results["toc_ranges"] = toc_ranges

    return results


def recommend_profile(analysis: dict) -> dict:
    """Generate profile and gate recommendations based on analysis."""

    rec = {
        "profile": "structured_parent_child",  # default
        "extraction_mode": "standard",
        "gates": {
            "min_length": 150,
            "citation_ratio_max": 0.03,
            "toc_skip_pages": 0,
        },
        "notes": [],
    }

    # TOC handling
    if analysis["toc_page_ratio"] > 0.05:
        rec["gates"]["toc_skip_pages"] = max(r[1] for r in analysis["toc_ranges"]) if analysis["toc_ranges"] else 15
        rec["notes"].append(f"TOC detected: skip first {rec['gates']['toc_skip_pages']} pages")

        if analysis["toc_page_ratio"] > 0.10:
            rec["profile"] = "baseline_toc_filter"
            rec["notes"].append("High TOC ratio: using baseline_toc_filter profile")

    # Citation density
    if analysis["citations_per_page"] > 5:
        rec["extraction_mode"] = "citation_anchored"
        rec["gates"]["citation_ratio_max"] = 0.06
        rec["notes"].append("High citation density: using citation_anchored extraction")

        if analysis["citations_per_page"] > 10:
            rec["gates"]["citation_ratio_max"] = 0.08
            rec["notes"].append("Very high citation density: raised citation_ratio_max to 0.08")

    # Low citation pages suggest commentary
    if analysis["citation_page_ratio"] < 0.3:
        rec["profile"] = "mixed_hybrid"
        rec["gates"]["min_length"] = 120
        rec["notes"].append("Low citation ratio: likely commentary, using mixed_hybrid")

    # Estimate expected massime
    # Heuristic: ~1 massima per 2-3 citations
    rec["expected_massime_min"] = analysis["total_citations"] // 3
    rec["expected_massime_max"] = analysis["total_citations"] // 2

    return rec


async def get_db_info(conn, filename: str) -> dict:
    """Get database info for a document."""

    row = await conn.fetchrow("""
        SELECT m.id, m.doc_id, m.anno, m.tipo, m.pages,
               count(ma.id) as n_massime,
               s.coverage_pct, s.matched_count, s.unmatched_count
        FROM kb.pdf_manifest m
        LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
        LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
        WHERE m.filename LIKE $1
        GROUP BY m.id, m.doc_id, m.anno, m.tipo, m.pages, s.coverage_pct, s.matched_count, s.unmatched_count
    """, f"%{filename}%")

    if row:
        return dict(row)
    return {}


async def main(doc_filter: str = None):
    print("=" * 80)
    print("DIAGNOSI CIVILE DOCUMENTS - PREFLIGHT")
    print("=" * 80)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get Civile documents with low coverage or low massime
    if doc_filter:
        query = """
            SELECT m.filename, m.anno, m.tipo, m.pages,
                   count(ma.id) as n_massime,
                   s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.filename ILIKE $1
            GROUP BY m.filename, m.anno, m.tipo, m.pages, s.coverage_pct
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query, f"%{doc_filter}%")
    else:
        query = """
            SELECT m.filename, m.anno, m.tipo, m.pages,
                   count(ma.id) as n_massime,
                   s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.tipo = 'civile' OR m.tipo IS NULL
            GROUP BY m.filename, m.anno, m.tipo, m.pages, s.coverage_pct
            HAVING count(ma.id) < 100 OR s.coverage_pct < 60 OR s.coverage_pct IS NULL
            ORDER BY s.coverage_pct ASC NULLS FIRST
            LIMIT 20
        """
        docs = await conn.fetch(query)

    print(f"Documenti da analizzare: {len(docs)}")
    print()

    for doc in docs:
        filename = doc["filename"]
        print("=" * 80)
        print(f"DOCUMENTO: {filename}")
        print("=" * 80)

        # Find PDF
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename

        if not pdf_path.exists():
            print(f"  [ERRORE] PDF non trovato: {filename}")
            print()
            continue

        # Analyze PDF
        print("  Analisi PDF...")
        analysis = analyze_pdf(pdf_path)

        # Get DB info
        db_info = await get_db_info(conn, filename)

        # Generate recommendations
        rec = recommend_profile(analysis)

        # Print results
        print()
        print("  DATABASE INFO:")
        print(f"    Anno: {db_info.get('anno', 'N/A')}")
        print(f"    Tipo: {db_info.get('tipo', 'N/A')}")
        print(f"    Massime estratte: {db_info.get('n_massime', 0)}")
        print(f"    Coverage: {float(db_info.get('coverage_pct') or 0):.1f}%")

        print()
        print("  ANALISI PDF:")
        print(f"    Pagine totali: {analysis['total_pages']}")
        print(f"    Pagine TOC: {len(analysis['toc_pages'])} ({analysis['toc_page_ratio']:.1%})")
        print(f"    TOC ranges: {analysis['toc_ranges']}")
        print(f"    Pagine con citazioni: {len(analysis['citation_pages'])} ({analysis['citation_page_ratio']:.1%})")
        print(f"    Citazioni totali: {analysis['total_citations']}")
        print(f"    Citazioni/pagina: {analysis['citations_per_page']:.1f}")

        print()
        print("  RACCOMANDAZIONI:")
        print(f"    Profile: {rec['profile']}")
        print(f"    Extraction mode: {rec['extraction_mode']}")
        print(f"    Gates:")
        for k, v in rec["gates"].items():
            print(f"      {k}: {v}")
        print(f"    Expected massime: {rec['expected_massime_min']}-{rec['expected_massime_max']}")
        if rec["notes"]:
            print(f"    Note:")
            for note in rec["notes"]:
                print(f"      - {note}")

        # Gap analysis
        current_massime = db_info.get('n_massime', 0)
        if current_massime < rec['expected_massime_min']:
            gap = rec['expected_massime_min'] - current_massime
            print()
            print(f"  [GAP] Mancano almeno {gap} massime (attese: {rec['expected_massime_min']}-{rec['expected_massime_max']})")

        print()

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Civile documents")
    parser.add_argument("--doc", type=str, help="Filter by document name")
    args = parser.parse_args()

    asyncio.run(main(doc_filter=args.doc))
