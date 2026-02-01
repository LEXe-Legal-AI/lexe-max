"""
QA Protocol - Phase 0: Build PDF Manifest

Creates qa_run record, scans PDF directory, computes SHA256/pages/bytes,
matches with existing kb.documents, populates kb.pdf_manifest.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s0_build_manifest.py
"""

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import asyncpg

# ── Config ────────────────────────────────────────────────────────
from qa_config import DB_URL, PDF_DIR

# Gate policy defaults (for config_json versioning)
GATE_PARAMS = {
    "min_length": 150,
    "max_citation_ratio": 0.03,
    "bad_starts": [", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO"],
}


def get_git_info() -> dict:
    """Get git sha, branch, dirty status."""
    info = {"git_sha": None, "git_branch": None, "git_dirty": None}
    try:
        info["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["git_dirty"] = len(status) > 0
    except Exception:
        pass
    return info


def get_lib_versions() -> dict:
    """Get versions of key libraries."""
    versions = {}
    try:
        import pymupdf
        versions["pymupdf"] = pymupdf.__version__
    except Exception:
        versions["pymupdf"] = "unknown"
    try:
        import httpx
        versions["httpx"] = httpx.__version__
    except Exception:
        versions["httpx"] = "unknown"
    versions["mistral_embed_model"] = "mistralai/mistral-embed-2312"
    return versions


def parse_filename(filename: str) -> tuple[int, str, int]:
    """Extract anno, tipo, volume from filename (from ingest_staging.py)."""
    vol_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "unico": 1, "1": 1, "2": 2, "3": 3, "4": 4}

    tipo = "unknown"
    if re.search(r"civile", filename, re.IGNORECASE):
        tipo = "civile"
    elif re.search(r"penale", filename, re.IGNORECASE):
        tipo = "penale"

    anno_match = re.search(r"(20\d{2})", filename)
    anno = int(anno_match.group(1)) if anno_match else 0

    volume = 1
    vol_match = re.search(r"Volume[_\s]+(I{1,3}V?|unico|\d)", filename, re.IGNORECASE)
    if vol_match:
        vol_str = vol_match.group(1).upper()
        volume = vol_map.get(vol_str, 1)
    vol_match2 = re.search(r"Vol[._\s]+(I{1,3}V?|\d)", filename, re.IGNORECASE)
    if vol_match2:
        vol_str = vol_match2.group(1).upper()
        volume = vol_map.get(vol_str, 1)
    vol_match3 = re.search(r"-\s*(I{1,3}V?)\s+volume", filename, re.IGNORECASE)
    if vol_match3:
        vol_str = vol_match3.group(1).upper()
        volume = vol_map.get(vol_str, 1)

    match = re.match(r"(\d{4})_MASSIMARIO\s+(CIVILE|PENALE)\s+VOL\.?\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower(), int(match.group(3))

    if tipo != "unknown" and anno > 0:
        return anno, tipo, volume

    return 0, "unknown", 0


def get_page_count(pdf_path: Path) -> int:
    """Get page count using PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 0: BUILD PDF MANIFEST")
    print("=" * 70)

    # ── Connect ───────────────────────────────────────────────────
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # ── Create QA Run ─────────────────────────────────────────────
    git_info = get_git_info()
    lib_versions = get_lib_versions()

    pdfs = sorted(PDF_DIR.glob("*.pdf")) + sorted(PDF_DIR.glob("new/*.pdf"))
    print(f"[OK] Found {len(pdfs)} PDFs in {PDF_DIR}")

    config_json = {
        **git_info,
        "lib_versions": lib_versions,
        "gate_params": GATE_PARAMS,
        "pdf_source": str(PDF_DIR),
        "pdf_count": len(pdfs),
    }

    qa_run_id = await conn.fetchval(
        """
        INSERT INTO kb.qa_runs (run_name, git_sha, pipeline, config_json)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING id
        """,
        f"qa_protocol_{len(pdfs)}pdfs",
        git_info.get("git_sha"),
        "qa_protocol_v1",
        json.dumps(config_json),
    )
    print(f"[OK] Created qa_run id={qa_run_id}")

    # ── Create Ingest Batch ───────────────────────────────────────
    batch_id = await conn.fetchval(
        """
        INSERT INTO kb.ingest_batches (batch_name, pipeline, config_json)
        VALUES ($1, $2, $3::jsonb)
        ON CONFLICT (batch_name) DO UPDATE SET started_at = now()
        RETURNING id
        """,
        "standard_v1",
        "ingest_staging",
        json.dumps({"gate_params": GATE_PARAMS, "extraction_strategy": "fast"}),
    )
    print(f"[OK] Ingest batch id={batch_id} (standard_v1)")

    # ── Build Manifest ────────────────────────────────────────────
    inserted = 0
    skipped = 0
    no_doc = 0

    for pdf_path in pdfs:
        filename = pdf_path.name
        sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        file_bytes = pdf_path.stat().st_size
        pages = get_page_count(pdf_path)
        anno, tipo, volume = parse_filename(filename)

        # Normalize filename for matching
        filename_norm = re.sub(r"[\s_\-]+", "_", filename.lower().replace(".pdf", ""))

        # Check if already in manifest
        exists = await conn.fetchval(
            "SELECT 1 FROM kb.pdf_manifest WHERE sha256 = $1", sha256
        )
        if exists:
            skipped += 1
            continue

        # Match with existing kb.documents via source_hash
        doc_id = await conn.fetchval(
            "SELECT id FROM kb.documents WHERE source_hash = $1", sha256
        )

        if not doc_id:
            print(f"  [WARN] No matching document for {filename}")
            no_doc += 1

        await conn.execute(
            """
            INSERT INTO kb.pdf_manifest
              (qa_run_id, doc_id, filename, filename_norm, sha256, pages, bytes,
               anno, tipo, volume, ingest_batch_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            qa_run_id,
            doc_id,
            filename,
            filename_norm,
            sha256,
            pages,
            file_bytes,
            anno if anno > 0 else None,
            tipo if tipo != "unknown" else None,
            str(volume),
            batch_id,
        )
        inserted += 1

    print(f"\n[RESULT] Manifest: {inserted} inserted, {skipped} skipped, {no_doc} no matching doc")

    # ── Verify ────────────────────────────────────────────────────
    manifest_count = await conn.fetchval("SELECT count(*) FROM kb.pdf_manifest")
    unique_sha = await conn.fetchval("SELECT count(DISTINCT sha256) FROM kb.pdf_manifest")

    print(f"\n[VERIFY] Total manifest entries: {manifest_count}")
    print(f"[VERIFY] Unique SHA256: {unique_sha}")
    print(f"[VERIFY] qa_run_id: {qa_run_id}")
    print(f"[VERIFY] ingest_batch_id: {batch_id}")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
