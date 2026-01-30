#!/usr/bin/env python3
"""
QA Protocol - Extract Reference Units with PyMuPDF (ref_v2)

Replaces Unstructured extraction with PyMuPDF for fair alignment comparison.
Uses INDEPENDENT segmentation (not same as massima_extractor).

Features:
- Segmentation by paragraphs/double newlines (generic, not legal-specific)
- Uses norm_v2 for normalization
- Computes simhash64 fingerprint
- Tracks extraction_engine, reference_version, normalization_version

Usage:
    uv run python scripts/qa/s0_extract_reference_units_pymupdf.py
    uv run python scripts/qa/s0_extract_reference_units_pymupdf.py --clear
    uv run python scripts/qa/s0_extract_reference_units_pymupdf.py --pdf "Volume I_2020.pdf"
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

import asyncpg
import fitz  # PyMuPDF

# Add src to path for normalization module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lexe_api.kb.ingestion.normalization import (
    normalize_v2,
    compute_content_hash,
    compute_simhash64,
    compute_spaced_letters_score,
)

from qa_config import DB_URL, PDF_DIR

# Configuration
MIN_LENGTH = 80  # Conservative, below pipeline's 150
MAX_LENGTH = 50000  # Truncate very long units

# Simple citation pattern for has_citation flag
CITATION_PATTERN = re.compile(
    r"(?:Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI)\s*,?\s*(?:sent\.?|ord\.?)?\s*n\.?\s*\d+)"
    r"|(?:Rv\.?\s*\d{6})"
    r"|(?:Cass\.\s*\d+/\d+)",
    re.IGNORECASE,
)


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text from PDF using PyMuPDF, one dict per page."""
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        pages.append({
            "page_num": i + 1,
            "text": text,
            "char_count": len(text),
        })

    doc.close()
    return pages


def segment_into_units(pages: list[dict], min_length: int = MIN_LENGTH) -> list[dict]:
    """
    Segment pages into reference units using INDEPENDENT logic.

    GUARDRAIL: This must NOT use the same logic as massima_extractor!
    We use PAGE-BASED splitting, not legal-specific patterns.

    Strategy:
    1. Each page with substantial text becomes a candidate unit
    2. Merge very short pages (<500 chars) with next page
    3. Split very long pages (>4000 chars) by double newlines
    4. This produces ~500-1000 units total (vs ~10000 pipeline massime)
    """
    units = []
    current_text = []
    current_start_page = None

    for page in pages:
        page_num = page["page_num"]
        text = page["text"].strip()

        if not text or len(text) < 50:  # Skip nearly empty pages
            continue

        # If page is very long, split by double newlines
        if len(text) > 4000:
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                para = para.strip()
                if len(para) >= min_length:
                    units.append({
                        "text": para,
                        "page_start": page_num,
                        "page_end": page_num,
                    })
        # If page is short, accumulate with next
        elif len(text) < 500:
            if not current_text:
                current_start_page = page_num
            current_text.append(text)
            # Check if accumulated is now substantial
            combined = "\n\n".join(current_text)
            if len(combined) >= 1000:
                units.append({
                    "text": combined,
                    "page_start": current_start_page,
                    "page_end": page_num,
                })
                current_text = []
                current_start_page = None
        # Normal page - flush any accumulated, then add this page
        else:
            # Flush accumulated short pages
            if current_text:
                combined = "\n\n".join(current_text)
                if len(combined) >= min_length:
                    units.append({
                        "text": combined,
                        "page_start": current_start_page,
                        "page_end": page_num - 1,
                    })
                current_text = []
                current_start_page = None

            # Add this page as a unit
            units.append({
                "text": text,
                "page_start": page_num,
                "page_end": page_num,
            })

    # Flush any remaining accumulated text
    if current_text:
        combined = "\n\n".join(current_text)
        if len(combined) >= min_length:
            units.append({
                "text": combined,
                "page_start": current_start_page,
                "page_end": pages[-1]["page_num"] if pages else 1,
            })

    return units


async def main(clear_existing: bool = False, single_pdf: str = None):
    print("=" * 70)
    print("QA PROTOCOL - REFERENCE UNITS EXTRACTION (PyMuPDF ref_v2)")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get or create qa_run
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )

    if not qa_run_id:
        # Create new run
        qa_run_id = await conn.fetchval(
            """
            INSERT INTO kb.qa_runs (name, config_json, status)
            VALUES ('ref_v2_pymupdf', '{"extraction_engine": "pymupdf", "reference_version": "ref_v2"}', 'running')
            RETURNING id
            """
        )
        print(f"[NEW] Created qa_run_id={qa_run_id}")
    else:
        print(f"[OK] Using qa_run_id={qa_run_id}")

    if clear_existing:
        deleted = await conn.execute(
            """
            DELETE FROM kb.qa_reference_units
            WHERE qa_run_id = $1 AND extraction_engine = 'pymupdf'
            """,
            qa_run_id,
        )
        print(f"[CLEAR] Deleted existing PyMuPDF reference units")

    # Get manifests
    if single_pdf:
        manifests = await conn.fetch(
            "SELECT id, filename FROM kb.pdf_manifest WHERE filename = $1",
            single_pdf,
        )
    else:
        manifests = await conn.fetch(
            "SELECT id, filename FROM kb.pdf_manifest ORDER BY filename"
        )

    print(f"PDF da processare: {len(manifests)}")
    print()

    total_units = 0
    processed = 0
    errors = 0

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already has ref_v2 units (skip if not clearing)
        if not clear_existing:
            existing = await conn.fetchval(
                """
                SELECT count(*) FROM kb.qa_reference_units
                WHERE manifest_id = $1 AND reference_version = 'ref_v2'
                """,
                manifest_id,
            )
            if existing > 0:
                print(f"  [SKIP] {filename[:45]:45} già {existing} units ref_v2")
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
            # Extract pages
            pages = extract_pdf_pages(pdf_path)

            # Segment into units (independent logic!)
            units = segment_into_units(pages)

            # Insert into database
            for i, unit in enumerate(units):
                raw_text = unit["text"]

                # Compute normalization
                testo_norm, spaced_score = normalize_v2(raw_text)
                content_hash = compute_content_hash(testo_norm)
                fingerprint = compute_simhash64(testo_norm)
                has_citation = bool(CITATION_PATTERN.search(raw_text))

                await conn.execute(
                    """
                    INSERT INTO kb.qa_reference_units (
                        qa_run_id, manifest_id, unit_index,
                        testo, testo_norm, raw_text,
                        content_hash, text_fingerprint,
                        char_count, has_citation,
                        spaced_letters_score,
                        extraction_method, extraction_engine, reference_version,
                        normalization_version, fingerprint_method
                    ) VALUES (
                        $1, $2, $3,
                        $4, $5, $6,
                        $7, $8,
                        $9, $10,
                        $11,
                        $12, $13, $14,
                        $15, $16
                    )
                    ON CONFLICT (manifest_id, unit_index, qa_run_id)
                    DO UPDATE SET
                        testo = EXCLUDED.testo,
                        testo_norm = EXCLUDED.testo_norm,
                        raw_text = EXCLUDED.raw_text,
                        content_hash = EXCLUDED.content_hash,
                        text_fingerprint = EXCLUDED.text_fingerprint,
                        spaced_letters_score = EXCLUDED.spaced_letters_score,
                        extraction_method = EXCLUDED.extraction_method,
                        extraction_engine = EXCLUDED.extraction_engine,
                        reference_version = EXCLUDED.reference_version
                    """,
                    qa_run_id, manifest_id, i,
                    raw_text[:MAX_LENGTH], testo_norm[:MAX_LENGTH], raw_text[:MAX_LENGTH],
                    content_hash, fingerprint,
                    len(raw_text), has_citation,
                    spaced_score,
                    "pymupdf", "pymupdf", "ref_v2",
                    "norm_v2", "simhash64_v1",
                )

            total_units += len(units)
            processed += 1
            print(f"  [OK]   {filename[:45]:45} {len(units):>4} units")

        except Exception as e:
            print(f"  [ERR]  {filename[:45]:45} {str(e)[:30]}")
            errors += 1

    # Update qa_run status
    await conn.execute(
        "UPDATE kb.qa_runs SET status = 'completed', completed_at = now() WHERE id = $1",
        qa_run_id,
    )

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
        """
        SELECT count(*) FROM kb.qa_reference_units
        WHERE qa_run_id = $1 AND reference_version = 'ref_v2'
        """,
        qa_run_id,
    )
    print(f"Total ref_v2 units in DB: {db_count}")

    # Check spaced letters score (should be ~0 for PyMuPDF)
    spaced_stats = await conn.fetchrow(
        """
        SELECT
            avg(spaced_letters_score) as avg_score,
            sum(CASE WHEN spaced_letters_score > 0.12 THEN 1 ELSE 0 END) as n_spaced
        FROM kb.qa_reference_units
        WHERE qa_run_id = $1 AND reference_version = 'ref_v2'
        """,
        qa_run_id,
    )
    if spaced_stats:
        print(f"Avg spaced_letters_score: {float(spaced_stats['avg_score'] or 0):.4f}")
        print(f"Units with spaced text (>0.12): {spaced_stats['n_spaced']}")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reference units with PyMuPDF")
    parser.add_argument("--clear", action="store_true", help="Clear existing ref_v2 units first")
    parser.add_argument("--pdf", type=str, help="Single PDF to process")
    args = parser.parse_args()

    asyncio.run(main(clear_existing=args.clear, single_pdf=args.pdf))
