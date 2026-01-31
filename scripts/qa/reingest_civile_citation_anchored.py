#!/usr/bin/env python3
"""
Re-Ingest Civile Documents with Citation-Anchored Extraction

Rielabora tutti i documenti Civile usando l'estrazione citation-anchored
per migliorare drasticamente il numero di massime estratte.

Risultati attesi:
- Volume I 2016: 43 → 3,035 massime (70x)
- Volume I 2017: 41 → 2,619 massime (64x)
- Coverage: 62% → 85%+

Usage:
    uv run python scripts/qa/reingest_civile_citation_anchored.py
    uv run python scripts/qa/reingest_civile_citation_anchored.py --dry-run
    uv run python scripts/qa/reingest_civile_citation_anchored.py --doc "Volume I_2016"
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import fitz  # PyMuPDF

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lexe_api.kb.ingestion.massima_extractor import extract_massime_from_pdf_text
from lexe_api.kb.ingestion.normalization import compute_simhash64, normalize_v2
from qa_config import DB_URL, PDF_DIR


# Gate configurations per document type
GATE_CONFIGS = {
    # High coverage documents - light tuning
    "high_coverage": {
        "toc_skip_pages": 15,
        "min_length": 130,
        "max_length": 2500,
        "citation_window_before": 2,
        "citation_window_after": 1,
        "split_on_multiple_citations": True,
    },
    # Medium coverage - standard citation-anchored
    "medium_coverage": {
        "toc_skip_pages": 20,
        "min_length": 120,
        "max_length": 2500,
        "citation_window_before": 2,
        "citation_window_after": 1,
        "split_on_multiple_citations": True,
    },
    # Low coverage - aggressive extraction
    "low_coverage": {
        "toc_skip_pages": 25,
        "min_length": 100,
        "max_length": 2000,
        "citation_window_before": 2,
        "citation_window_after": 1,
        "split_on_multiple_citations": True,
    },
    # Very low coverage - maximum extraction
    "critical": {
        "toc_skip_pages": 30,
        "min_length": 80,
        "max_length": 1800,
        "citation_window_before": 3,
        "citation_window_after": 2,
        "split_on_multiple_citations": True,
    },
}


def get_gate_config(coverage_pct: float, n_massime: int) -> dict:
    """Select gate config based on current document stats."""
    if coverage_pct is None or coverage_pct < 15:
        return GATE_CONFIGS["critical"]
    elif coverage_pct < 30:
        return GATE_CONFIGS["low_coverage"]
    elif coverage_pct < 50:
        return GATE_CONFIGS["medium_coverage"]
    else:
        return GATE_CONFIGS["high_coverage"]


async def get_civile_documents(conn, doc_filter: str = None) -> list[dict]:
    """Get all Civile documents that need reprocessing."""
    if doc_filter:
        query = """
            SELECT m.id, m.doc_id, m.filename, m.anno, m.pages,
                   count(ma.id) as n_massime,
                   s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.filename ILIKE $1
            GROUP BY m.id, m.doc_id, m.filename, m.anno, m.pages, s.coverage_pct
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query, f"%{doc_filter}%")
    else:
        query = """
            SELECT m.id, m.doc_id, m.filename, m.anno, m.pages,
                   count(ma.id) as n_massime,
                   s.coverage_pct
            FROM kb.pdf_manifest m
            LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
            LEFT JOIN kb.reference_alignment_summary s ON s.manifest_id = m.id
            WHERE m.tipo = 'civile' OR m.tipo IS NULL
            GROUP BY m.id, m.doc_id, m.filename, m.anno, m.pages, s.coverage_pct
            ORDER BY s.coverage_pct ASC NULLS FIRST
        """
        docs = await conn.fetch(query)

    return [dict(d) for d in docs]


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from all pages of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        pages.append((i + 1, text))
    doc.close()
    return pages


async def reingest_document(
    conn,
    doc: dict,
    batch_id: str,
    dry_run: bool = False,
) -> dict:
    """Re-ingest a single document with citation-anchored extraction."""
    filename = doc["filename"]
    doc_id = doc["doc_id"]
    manifest_id = doc["id"]
    coverage = float(doc["coverage_pct"] or 0)
    current_massime = doc["n_massime"]

    # Find PDF
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        pdf_path = PDF_DIR / "new" / filename
    if not pdf_path.exists():
        return {"status": "error", "error": f"PDF not found: {filename}"}

    # Select gate config
    gate_config = get_gate_config(coverage, current_massime)

    # Extract pages
    pages = extract_pdf_pages(pdf_path)

    # Extract massime with citation-anchored mode
    massime = extract_massime_from_pdf_text(
        pages=pages,
        extraction_mode="citation_anchored",
        toc_skip_pages=gate_config["toc_skip_pages"],
        gate_config=gate_config,
    )

    result = {
        "filename": filename,
        "manifest_id": manifest_id,
        "doc_id": doc_id,
        "pages": len(pages),
        "current_massime": current_massime,
        "new_massime": len(massime),
        "improvement": len(massime) / current_massime if current_massime > 0 else float('inf'),
        "with_citation": sum(1 for m in massime if m.citation_complete),
        "gate_config": gate_config,
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    # Delete old massime for this document
    await conn.execute(
        "DELETE FROM kb.massime WHERE document_id = $1",
        doc_id,
    )

    # Insert new massime
    inserted = 0
    for m in massime:
        # Compute normalization and fingerprint
        testo_norm, _ = normalize_v2(m.testo)
        fingerprint = compute_simhash64(testo_norm)

        try:
            await conn.execute(
                """
                INSERT INTO kb.massime (
                    id, document_id, testo, testo_normalizzato, content_hash,
                    text_fingerprint, sezione, numero_sentenza, anno,
                    data_decisione, rv, relatore, tema,
                    page_start, page_end, section_context, section_path,
                    extraction_mode, batch_id, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11, $12, $13,
                    $14, $15, $16, $17,
                    $18, $19, $20
                )
                """,
                str(uuid4()),           # id
                doc_id,                 # document_id
                m.testo,                # testo
                testo_norm,             # testo_normalizzato
                m.content_hash,         # content_hash
                fingerprint,            # text_fingerprint
                m.citation.normalized_sezione,  # sezione
                m.citation.numero,      # numero_sentenza
                m.citation.anno,        # anno
                m.citation.data_decisione,  # data_decisione
                m.citation.rv,          # rv
                m.citation.relatore,    # relatore
                None,                   # tema (to be extracted later)
                m.page_start,           # page_start
                m.page_end,             # page_end
                m.section_context,      # section_context
                m.section_path,         # section_path
                "citation_anchored",    # extraction_mode
                batch_id,               # batch_id
                datetime.utcnow(),      # created_at
            )
            inserted += 1
        except Exception as e:
            # Log but continue
            print(f"    [WARN] Insert failed: {e}")

    result["inserted"] = inserted
    result["status"] = "success"

    return result


async def main(dry_run: bool = False, doc_filter: str = None):
    print("=" * 80)
    print("RE-INGEST CIVILE DOCUMENTS - CITATION-ANCHORED")
    print("=" * 80)
    print()

    if dry_run:
        print("[DRY RUN MODE - No changes will be made]")
        print()

    conn = await asyncpg.connect(DB_URL)

    # Get documents
    docs = await get_civile_documents(conn, doc_filter)
    print(f"Documenti da processare: {len(docs)}")
    print()

    # Create batch
    batch_id = f"civile_citation_anchored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Batch ID: {batch_id}")
    print()

    # Process each document
    total_current = 0
    total_new = 0
    results = []

    for i, doc in enumerate(docs, 1):
        filename = doc["filename"][:50]
        print(f"[{i}/{len(docs)}] {filename}...")

        result = await reingest_document(conn, doc, batch_id, dry_run)

        if result["status"] == "error":
            print(f"  [ERROR] {result['error']}")
        else:
            total_current += result["current_massime"]
            total_new += result["new_massime"]

            improvement = result["improvement"]
            if improvement == float('inf'):
                imp_str = "NEW (was 0)"
            else:
                imp_str = f"{improvement:.1f}x"

            print(f"  Current: {result['current_massime']:>4}, New: {result['new_massime']:>5}, "
                  f"Improvement: {imp_str}")

        results.append(result)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Documents processed: {len(docs)}")
    print(f"Total current massime: {total_current}")
    print(f"Total new massime: {total_new}")
    if total_current > 0:
        print(f"Overall improvement: {total_new / total_current:.1f}x")
    print()

    if not dry_run:
        print(f"[COMMITTED] Batch: {batch_id}")
    else:
        print("[DRY RUN] No changes made. Run without --dry-run to commit.")

    await conn.close()

    # Write results to file
    results_file = Path(__file__).parent / "REINGEST_RESULTS.md"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"# Re-Ingest Civile Results\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Batch:** {batch_id}\n")
        f.write(f"**Mode:** {'DRY RUN' if dry_run else 'COMMITTED'}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Documents | {len(docs)} |\n")
        f.write(f"| Current Massime | {total_current} |\n")
        f.write(f"| New Massime | {total_new} |\n")
        if total_current > 0:
            f.write(f"| Improvement | {total_new / total_current:.1f}x |\n")
        f.write(f"\n## Per-Document Results\n\n")
        f.write(f"| Document | Current | New | Improvement |\n")
        f.write(f"|----------|---------|-----|-------------|\n")
        for r in results:
            if r["status"] != "error":
                imp = r["improvement"]
                imp_str = "NEW" if imp == float('inf') else f"{imp:.1f}x"
                f.write(f"| {r['filename'][:40]} | {r['current_massime']} | {r['new_massime']} | {imp_str} |\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-ingest Civile with citation-anchored")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing to DB")
    parser.add_argument("--doc", type=str, help="Filter by document name")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, doc_filter=args.doc))
