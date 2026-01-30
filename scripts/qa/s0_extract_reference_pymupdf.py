#!/usr/bin/env python3
"""
QA Protocol - Re-extract Reference Units with PyMuPDF.

Replaces Unstructured extraction with PyMuPDF for fair alignment comparison.
Uses same extraction logic as guided_ingestion_local.py.

Usage:
    uv run python scripts/qa/s0_extract_reference_pymupdf.py
    uv run python scripts/qa/s0_extract_reference_pymupdf.py --clear  # Clear existing first
"""

import argparse
import asyncio
import hashlib
import re
from pathlib import Path

import asyncpg
import fitz  # PyMuPDF

from qa_config import DB_URL, PDF_DIR

# Same patterns as guided_ingestion_local.py
MASSIMA_START_PATTERN = re.compile(
    r"(?:^|\n)"
    r"(?:Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI)|\d{5,6})"
    r"|(?:Cass\.|Cassazione)"
    r"|(?:sent\.\s*n\.?\s*\d+)"
    r"|(?:ord\.\s*n\.?\s*\d+)",
    re.IGNORECASE | re.MULTILINE,
)

CITATION_PATTERN = re.compile(
    r"(?:Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI)\s*,?\s*(?:sent\.?|ord\.?)?\s*n\.?\s*\d+)"
    r"|(?:Rv\.?\s*\d{6})"
    r"|(?:\d{6}\s*,?\s*rv\.?\s*\d+)",
    re.IGNORECASE,
)

TOC_PATTERN = re.compile(r"\.{3,}|\.\\s+\\.\\s+\\.|_{5,}|—{3,}")
BAD_STARTS = [", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO", "Pag.", "pag."]


def normalize_text(text: str) -> str:
    """Normalize text for comparison - same as guided_ingestion."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_hash(text: str) -> str:
    """Compute content hash from normalized text."""
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode()).hexdigest()[:40]


def extract_units_from_pdf(pdf_path: Path, min_length: int = 80) -> list[dict]:
    """
    Extract reference units from PDF using PyMuPDF.

    Uses conservative segmentation (min_length=80) to capture more units
    than the pipeline (min_length=150).
    """
    doc = fitz.open(pdf_path)
    units = []
    current_text = []
    current_start_page = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text:
            continue

        # Check for unit boundaries (massima start patterns)
        if MASSIMA_START_PATTERN.search(text) and current_text:
            # Flush current unit
            combined = "\n".join(current_text)
            if len(combined) >= min_length:
                # Apply basic filtering
                if not any(combined.strip().startswith(bs) for bs in BAD_STARTS):
                    if not (TOC_PATTERN.search(combined) and len(combined) < 300):
                        units.append({
                            "text": combined,
                            "page_start": current_start_page,
                            "page_end": page_num,
                        })
            current_text = [text]
            current_start_page = page_num + 1
        else:
            if not current_text:
                current_start_page = page_num + 1
            current_text.append(text)

    # Flush last unit
    if current_text:
        combined = "\n".join(current_text)
        if len(combined) >= min_length:
            if not any(combined.strip().startswith(bs) for bs in BAD_STARTS):
                if not (TOC_PATTERN.search(combined) and len(combined) < 300):
                    units.append({
                        "text": combined,
                        "page_start": current_start_page,
                        "page_end": len(doc),
                    })

    doc.close()
    return units


async def main(clear_existing: bool = False):
    print("=" * 70)
    print("QA PROTOCOL - REFERENCE UNITS EXTRACTION (PyMuPDF)")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get qa_run_id
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"qa_run_id: {qa_run_id}")

    if clear_existing:
        deleted = await conn.execute(
            "DELETE FROM kb.qa_reference_units WHERE qa_run_id = $1",
            qa_run_id,
        )
        print(f"[CLEAR] Deleted existing reference units for qa_run_id={qa_run_id}")

    # Get manifests
    manifests = await conn.fetch(
        "SELECT id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"PDF da processare: {len(manifests)}")
    print()

    total_units = 0
    processed = 0
    errors = 0

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already has units (skip if not clearing)
        if not clear_existing:
            existing = await conn.fetchval(
                "SELECT count(*) FROM kb.qa_reference_units WHERE manifest_id = $1",
                manifest_id,
            )
            if existing > 0:
                print(f"  [SKIP] {filename[:45]:45} già {existing} units")
                total_units += existing
                processed += 1
                continue

        # Find PDF
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  [MISS] {filename[:45]:45} NOT FOUND")
            errors += 1
            continue

        try:
            # Extract units
            units = extract_units_from_pdf(pdf_path)

            # Insert into database
            for i, unit in enumerate(units):
                text = unit["text"]
                text_norm = normalize_text(text)
                content_hash = compute_hash(text)
                has_citation = bool(CITATION_PATTERN.search(text))

                await conn.execute(
                    """
                    INSERT INTO kb.qa_reference_units
                    (qa_run_id, manifest_id, unit_index, testo, testo_norm,
                     content_hash, char_count, has_citation, extraction_method)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (manifest_id, unit_index) DO UPDATE SET
                    testo = EXCLUDED.testo,
                    testo_norm = EXCLUDED.testo_norm,
                    content_hash = EXCLUDED.content_hash,
                    extraction_method = EXCLUDED.extraction_method
                    """,
                    qa_run_id, manifest_id, i, text[:50000], text_norm[:50000],
                    content_hash, len(text), has_citation, "pymupdf",
                )

            total_units += len(units)
            processed += 1
            print(f"  [OK]   {filename[:45]:45} {len(units):>4} units")

        except Exception as e:
            print(f"  [ERR]  {filename[:45]:45} {str(e)[:30]}")
            errors += 1

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total units extracted: {total_units}")

    # Verify in DB
    db_count = await conn.fetchval(
        "SELECT count(*) FROM kb.qa_reference_units WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"Total units in DB: {db_count}")

    # Compare with old Unstructured extraction
    unstructured_count = await conn.fetchval(
        """
        SELECT count(*) FROM kb.qa_reference_units
        WHERE qa_run_id = $1 AND extraction_method = 'unstructured'
        """,
        qa_run_id,
    )
    pymupdf_count = await conn.fetchval(
        """
        SELECT count(*) FROM kb.qa_reference_units
        WHERE qa_run_id = $1 AND extraction_method = 'pymupdf'
        """,
        qa_run_id,
    )
    print(f"\nBy extraction method:")
    print(f"  Unstructured: {unstructured_count}")
    print(f"  PyMuPDF: {pymupdf_count}")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reference units with PyMuPDF")
    parser.add_argument("--clear", action="store_true", help="Clear existing units first")
    args = parser.parse_args()

    asyncio.run(main(clear_existing=args.clear))
