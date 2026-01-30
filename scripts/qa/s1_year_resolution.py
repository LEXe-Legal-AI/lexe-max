"""
QA Protocol - Phase 1: Year Resolution

Resolves anno for each PDF from three sources:
1. anno_from_filename (parse_filename)
2. anno_from_content (extract_anno_from_text on first pages)
3. anno_from_metadata (PyMuPDF doc.metadata)

Flags conflicts between sources.

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    uv run python scripts/qa/s1_year_resolution.py
"""

import asyncio
import re
from pathlib import Path

import asyncpg

from qa_config import DB_URL, PDF_DIR


def parse_anno_from_filename(filename: str) -> int | None:
    """Extract year from filename (from ingest_staging.py)."""
    match = re.search(r"(20\d{2})", filename)
    return int(match.group(1)) if match else None


def extract_anno_from_text(text: str) -> int | None:
    """Try to extract year from document text (from ingest_recover_anno.py)."""
    if not text:
        return None
    # Rassegna/Massimario pattern
    match = re.search(r"(?:Rassegna|Massimario)[^\d]*(\d{4})", text, re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 2008 <= year <= 2025:
            return year
    # "anno 20XX"
    match = re.search(r"anno\s*(\d{4})", text, re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 2008 <= year <= 2025:
            return year
    # Standalone years
    matches = re.findall(r"\b(20\d{2})\b", text)
    for y_str in matches:
        year = int(y_str)
        if 2008 <= year <= 2025:
            return year
    return None


def extract_anno_from_metadata(pdf_path: Path) -> int | None:
    """Extract year from PDF metadata via PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        meta = doc.metadata
        doc.close()

        # Check CreationDate, ModDate
        for key in ("creationDate", "modDate"):
            val = meta.get(key, "")
            if val:
                match = re.search(r"(20\d{2})", val)
                if match:
                    year = int(match.group(1))
                    if 2008 <= year <= 2025:
                        return year

        # Check for InDesign or pdfmark in producer/creator
        for key in ("producer", "creator"):
            val = meta.get(key, "")
            if val and re.search(r"(?:InDesign|pdfmark)", val, re.IGNORECASE):
                match = re.search(r"(20\d{2})", val)
                if match:
                    year = int(match.group(1))
                    if 2008 <= year <= 2025:
                        return year
    except Exception:
        pass
    return None


def extract_first_pages_text(pdf_path: Path, max_pages: int = 5) -> str:
    """Extract text from first N pages using PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        texts = []
        for i in range(min(max_pages, doc.page_count)):
            texts.append(doc[i].get_text())
        doc.close()
        return " ".join(texts)
    except Exception:
        return ""


def resolve_anno(
    anno_filename: int | None,
    anno_content: int | None,
    anno_metadata: int | None,
) -> tuple[int | None, str, bool, str | None]:
    """
    Resolve final anno from three sources.
    Returns (anno_resolved, resolution_method, has_conflict, conflict_details).
    """
    sources = {}
    if anno_filename:
        sources["filename"] = anno_filename
    if anno_content:
        sources["content"] = anno_content
    if anno_metadata:
        sources["metadata"] = anno_metadata

    unique_years = set(sources.values())
    has_conflict = len(unique_years) > 1

    conflict_details = None
    if has_conflict:
        conflict_details = "; ".join(f"{k}={v}" for k, v in sources.items())

    # Priority: filename > content > metadata
    if anno_filename:
        return anno_filename, "filename", has_conflict, conflict_details
    if anno_content:
        return anno_content, "content", has_conflict, conflict_details
    if anno_metadata:
        return anno_metadata, "metadata", has_conflict, conflict_details

    return None, "none", False, None


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 1: YEAR RESOLUTION")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] Using qa_run_id={qa_run_id}")

    manifests = await conn.fetch(
        "SELECT id, filename, anno FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    total_resolved = 0
    total_conflicts = 0

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT 1 FROM kb.pdf_year_resolution WHERE manifest_id = $1",
            manifest_id,
        )
        if existing:
            continue

        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  [WARN] PDF not found: {filename}")
            continue

        anno_filename = parse_anno_from_filename(filename)
        first_text = extract_first_pages_text(pdf_path)
        anno_content = extract_anno_from_text(first_text)
        anno_metadata = extract_anno_from_metadata(pdf_path)

        anno_resolved, method, has_conflict, conflict_details = resolve_anno(
            anno_filename, anno_content, anno_metadata
        )

        await conn.execute(
            """
            INSERT INTO kb.pdf_year_resolution
              (qa_run_id, manifest_id, anno_from_filename, anno_from_content,
               anno_from_metadata, anno_resolved, resolution_method,
               has_conflict, conflict_details)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (manifest_id) DO NOTHING
            """,
            qa_run_id, manifest_id,
            anno_filename, anno_content, anno_metadata,
            anno_resolved, method, has_conflict, conflict_details,
        )

        status = "CONFLICT" if has_conflict else "OK"
        print(f"  [{status}] {filename}: {anno_resolved} ({method})")
        total_resolved += 1
        if has_conflict:
            total_conflicts += 1
            print(f"    Conflict: {conflict_details}")

    # Summary
    conflict_count = await conn.fetchval(
        "SELECT count(*) FROM kb.pdf_year_resolution WHERE has_conflict = true"
    )
    no_year = await conn.fetchval(
        "SELECT count(*) FROM kb.pdf_year_resolution WHERE anno_resolved IS NULL"
    )

    print(f"\n{'=' * 70}")
    print(f"YEAR RESOLUTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total resolved: {total_resolved}")
    print(f"Conflicts: {conflict_count}")
    print(f"No year found: {no_year}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
