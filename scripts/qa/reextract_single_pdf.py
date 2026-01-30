#!/usr/bin/env python3
"""Re-extract a single PDF with updated segmentation."""

import asyncio
import hashlib
import re
import sys
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


async def main(filename: str):
    print(f"Re-extracting: {filename}")

    conn = await asyncpg.connect(DB_URL)

    # Get manifest
    row = await conn.fetchrow(
        "SELECT id, doc_id, sha256 FROM kb.pdf_manifest WHERE filename = $1",
        filename
    )
    if not row:
        print(f"[ERROR] Not found in manifest: {filename}")
        await conn.close()
        return

    manifest_id = row["id"]
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )

    # Find PDF
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        await conn.close()
        return

    # Extract
    print("Extracting with Unstructured (fast)...")
    with open(pdf_path, "rb") as f:
        response = httpx.post(
            UNSTRUCTURED_URL,
            files={"files": (pdf_path.name, f, "application/pdf")},
            data={"strategy": "fast", "output_format": "application/json"},
            timeout=600.0,
        )

    if response.status_code != 200:
        print(f"[ERROR] {response.status_code}")
        await conn.close()
        return

    elements = response.json()
    print(f"Elements: {len(elements)}")

    # Segment
    units = segment_reference_units(elements)
    print(f"Units: {len(units)}")

    if not units:
        print("[WARN] No units!")
        await conn.close()
        return

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

    print(f"[OK] Inserted {len(units)} reference units")

    # Verify
    count = await conn.fetchval(
        "SELECT count(*) FROM kb.qa_reference_units WHERE manifest_id = $1",
        manifest_id
    )
    print(f"Verified: {count} units in database")

    await conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "2014 Mass civile Vol 1 pagg 408.pdf"

    asyncio.run(main(filename))
