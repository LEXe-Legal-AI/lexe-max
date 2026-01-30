#!/usr/bin/env python3
"""Re-extract all already-processed PDFs with new segmentation pattern."""

import asyncio
import hashlib
import re
import time
from pathlib import Path

import asyncpg
import httpx

from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL

MIN_LENGTH_REF = 80


def stable_normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"(?:^|\n)\s*\d{1,4}\s*(?:\n|$)", "\n", t)
    t = re.sub(r"\.{3,}", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_ref_hash(text_norm: str) -> str:
    return hashlib.sha256(text_norm.encode("utf-8")).hexdigest()


def has_citation(text: str) -> bool:
    return bool(re.search(r"(?:Sez\.?|Sezione)\s*[UuLl0-9]", text, re.IGNORECASE))


def segment_reference_units(elements: list[dict]) -> list[dict]:
    """New segmentation with La Corte, In tema patterns."""
    units = []
    current_texts = []
    current_page_start = None
    current_page_end = None

    for elem in elements:
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")
        page = elem.get("metadata", {}).get("page_number")

        if elem_type in ("Header", "Footer", "PageNumber"):
            continue

        # NEW pattern with La Corte, In tema
        if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+|La Corte|In tema)", text, re.IGNORECASE):
            if current_texts:
                full = " ".join(current_texts)
                if len(full) >= MIN_LENGTH_REF:
                    units.append({
                        "testo": full,
                        "page_start": current_page_start,
                        "page_end": current_page_end,
                        "has_citation": has_citation(full),
                    })
            current_texts = [text]
            current_page_start = page
            current_page_end = page
        elif text:
            if not current_texts:
                current_page_start = page
            current_texts.append(text)
            current_page_end = page

    if current_texts:
        full = " ".join(current_texts)
        if len(full) >= MIN_LENGTH_REF:
            units.append({
                "testo": full,
                "page_start": current_page_start,
                "page_end": current_page_end,
                "has_citation": has_citation(full),
            })

    return units


async def reextract_pdf(conn, manifest_id: int, filename: str, qa_run_id: int) -> tuple[int, int]:
    """Re-extract a single PDF. Returns (old_count, new_count)."""

    # Find PDF
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        pdf_path = PDF_DIR / "new" / filename
    if not pdf_path.exists():
        print(f"  [SKIP] Not found: {filename}", flush=True)
        return 0, 0

    # Count old
    old_count = await conn.fetchval(
        "SELECT count(*) FROM kb.qa_reference_units WHERE manifest_id = $1",
        manifest_id
    )

    # Extract
    start = time.time()
    try:
        with open(pdf_path, "rb") as f:
            response = httpx.post(
                UNSTRUCTURED_URL,
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={"strategy": "fast", "output_format": "application/json"},
                timeout=600.0,
            )
    except Exception as e:
        print(f"  [ERROR] {filename}: {e}", flush=True)
        return old_count, 0

    if response.status_code != 200:
        print(f"  [ERROR] {filename}: HTTP {response.status_code}", flush=True)
        return old_count, 0

    elements = response.json()
    elapsed = time.time() - start

    # Segment
    units = segment_reference_units(elements)

    if not units:
        print(f"  [WARN] No units: {filename}", flush=True)
        return old_count, 0

    # Delete old
    await conn.execute(
        "DELETE FROM kb.qa_reference_units WHERE manifest_id = $1",
        manifest_id
    )

    # Insert new
    for idx, unit in enumerate(units):
        testo_norm = stable_normalize(unit["testo"])
        content_hash = compute_ref_hash(testo_norm)

        await conn.execute(
            """
            INSERT INTO kb.qa_reference_units
              (qa_run_id, manifest_id, unit_index, testo, testo_norm,
               content_hash, char_count, page_start, page_end,
               has_citation, extraction_method)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            qa_run_id,
            manifest_id,
            idx,
            unit["testo"],
            testo_norm,
            content_hash,
            len(unit["testo"]),
            unit["page_start"],
            unit["page_end"],
            unit["has_citation"],
            "unstructured_fast_v2",
        )

    print(f"  [OK] {filename[:50]:50} {old_count:4} -> {len(units):4} ({elapsed:.1f}s)", flush=True)
    return old_count, len(units)


async def main():
    print("=" * 70)
    print("RE-EXTRACT ALL PROCESSED PDFs WITH NEW SEGMENTATION")
    print("=" * 70)
    print(flush=True)

    conn = await asyncpg.connect(DB_URL)

    # Get qa_run_id
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )

    # Get all processed PDFs (excluding 2014 Mass civile Vol 1 already done)
    rows = await conn.fetch("""
        SELECT DISTINCT m.id, m.filename
        FROM kb.pdf_manifest m
        JOIN kb.qa_reference_units r ON r.manifest_id = m.id
        WHERE m.filename NOT LIKE '%2014 Mass civile Vol 1%'
        ORDER BY m.filename
    """)

    print(f"PDFs to re-extract: {len(rows)}")
    print(flush=True)

    total_old = 0
    total_new = 0

    for i, row in enumerate(rows):
        manifest_id = row["id"]
        filename = row["filename"]

        print(f"[{i+1}/{len(rows)}] Processing...", flush=True)
        old, new = await reextract_pdf(conn, manifest_id, filename, qa_run_id)
        total_old += old
        total_new += new

    print(flush=True)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total OLD units: {total_old}")
    print(f"Total NEW units: {total_new}")
    print(f"Change: {total_new - total_old:+d} ({100*(total_new-total_old)/total_old:.1f}%)")

    # Final verification
    final_count = await conn.fetchval("SELECT count(*) FROM kb.qa_reference_units")
    final_docs = await conn.fetchval("SELECT count(DISTINCT manifest_id) FROM kb.qa_reference_units")
    print(f"\nDatabase total: {final_count} units across {final_docs} documents")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
