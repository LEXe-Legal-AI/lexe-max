"""
QA Protocol - Phase 0: Extract Reference Units

Independent ground-truth extraction using Unstructured hi_res strategy.
Conservative segmentation, lenient gate (min_length=80), saves everything.
Stable normalization for content_hash.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s0_extract_reference_units.py
"""

import asyncio
import hashlib
import re
from pathlib import Path

import asyncpg
import httpx

# ── Config ────────────────────────────────────────────────────────
from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL
MIN_LENGTH_REF = 80  # Lenient threshold (below standard 150)


def stable_normalize(text: str) -> str:
    """
    Stable normalization for reference units.
    - lowercase
    - collapse whitespace
    - remove isolated page numbers
    - remove TOC dotted lines
    """
    if not text:
        return ""
    t = text.lower()
    # Remove isolated page numbers at line boundaries
    t = re.sub(r"(?:^|\n)\s*\d{1,4}\s*(?:\n|$)", "\n", t)
    # Remove TOC dotted lines
    t = re.sub(r"\.{3,}", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_ref_hash(text_norm: str) -> str:
    """Compute content_hash from normalized text."""
    return hashlib.sha256(text_norm.encode("utf-8")).hexdigest()


def has_citation(text: str) -> bool:
    """Check if text contains Cassazione citation."""
    return bool(re.search(
        r"(?:Sez\.?|Sezione)\s*[UuLl0-9]", text, re.IGNORECASE
    ))


def extract_hi_res_sync(pdf_path: Path) -> list[dict]:
    """Extract with Unstructured hi_res strategy (sync version)."""
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = httpx.post(
            UNSTRUCTURED_URL,
            files=files,
            data={"strategy": "hi_res", "output_format": "application/json"},
            timeout=3600.0,  # 1 hour for very large PDFs
        )
    if response.status_code != 200:
        print(f"  [ERROR] Unstructured API: {response.status_code}", flush=True)
        return []
    return response.json()


async def extract_hi_res(pdf_path: Path) -> list[dict]:
    """Extract with Unstructured hi_res strategy (runs sync in thread)."""
    print(f"  Extracting (hi_res): {pdf_path.name}", flush=True)
    import time
    start = time.time()
    # Run sync httpx in thread to avoid async issues with large responses
    result = await asyncio.to_thread(extract_hi_res_sync, pdf_path)
    elapsed = time.time() - start
    print(f"  Completed in {elapsed/60:.1f} min", flush=True)
    return result


def segment_reference_units(elements: list[dict]) -> list[dict]:
    """
    Conservative segmentation: merge consecutive NarrativeText elements.
    Returns list of {testo, page_start, page_end, has_citation}.
    """
    units = []
    current_texts = []
    current_page_start = None
    current_page_end = None

    for elem in elements:
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")
        page = elem.get("metadata", {}).get("page_number")

        # Skip headers, footers, page numbers
        if elem_type in ("Header", "Footer", "PageNumber"):
            continue

        # Section boundary -> flush
        # Patterns: Sez., SEZIONE, N. XX, La Corte, In tema (per 2014 Mass civile)
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

    # Last unit
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


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 0: EXTRACT REFERENCE UNITS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Get latest qa_run_id
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    if not qa_run_id:
        print("[ERROR] No qa_run found. Run s0_build_manifest.py first.")
        await conn.close()
        return

    print(f"[OK] Using qa_run_id={qa_run_id}")

    # Get manifest entries
    manifests = await conn.fetch(
        "SELECT id, doc_id, filename, sha256 FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    total_units = 0
    total_skipped = 0

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already extracted
        existing = await conn.fetchval(
            "SELECT count(*) FROM kb.qa_reference_units WHERE manifest_id = $1",
            manifest_id,
        )
        if existing > 0:
            print(f"  [SKIP] {filename}: {existing} units already exist", flush=True)
            total_skipped += 1
            continue

        # Find PDF file
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  [WARN] PDF not found: {filename}", flush=True)
            continue

        # Extract with hi_res (sync in thread)
        try:
            elements = await extract_hi_res(pdf_path)
        except Exception as e:
            print(f"  [ERROR] {filename}: {type(e).__name__}: {e}", flush=True)
            continue

        if not elements:
            print(f"  [WARN] No elements: {filename}", flush=True)
            continue

        print(f"  Extracted {len(elements)} elements", flush=True)

        # Segment into reference units
        units = segment_reference_units(elements)
        print(f"  Segmented into {len(units)} reference units", flush=True)

        # Insert
        count = 0
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
                ON CONFLICT (manifest_id, unit_index) DO NOTHING
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
                "unstructured_hi_res",
            )
            count += 1

        total_units += count
        print(f"  [OK] {filename}: {count} reference units inserted", flush=True)

    # Verify
    ref_count = await conn.fetchval("SELECT count(*) FROM kb.qa_reference_units")
    ref_docs = await conn.fetchval(
        "SELECT count(DISTINCT manifest_id) FROM kb.qa_reference_units"
    )

    print(f"\n{'=' * 70}")
    print(f"REFERENCE UNITS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total reference units: {ref_count}")
    print(f"Documents covered: {ref_docs}")
    print(f"Skipped (already done): {total_skipped}")
    print(f"New units this run: {total_units}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
