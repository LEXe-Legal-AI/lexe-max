"""
QA Protocol - Phase 2: Noise Detection

Detects noise patterns in extracted elements:
- Dotted lines (TOC artifacts)
- Page-number-only lines
- Repeated characters
- Parsing artifacts

Updates pdf_extraction_quality with noise counts.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s2_noise_detection.py
"""

import asyncio
import re
from pathlib import Path

import asyncpg
import httpx

from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL

# Noise patterns (from cleaner.py)
DOTTED_LINE = re.compile(r"\.{4,}")
SEPARATOR_LINE = re.compile(r"^[.\-_=]+$")
PAGE_NUMBER_ONLY = re.compile(r"^\s*\d{1,4}\s*$")
REPEATED_CHARS = re.compile(r"(.)\1{3,}")
PARSING_ARTIFACT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def count_noise(elements: list[dict]) -> dict:
    """Count noise patterns in elements."""
    noise_markers = 0
    parsing_artifacts = 0
    page_number_only = 0

    for elem in elements:
        text = elem.get("text", "").strip()
        if not text:
            continue

        # Dotted lines
        if DOTTED_LINE.search(text):
            noise_markers += 1
        # Separator lines
        if SEPARATOR_LINE.match(text):
            noise_markers += 1
        # Page number only
        if PAGE_NUMBER_ONLY.match(text):
            page_number_only += 1
        # Repeated chars
        if REPEATED_CHARS.search(text):
            noise_markers += 1
        # Parsing artifacts
        if PARSING_ARTIFACT.search(text):
            parsing_artifacts += 1

    return {
        "noise_markers_count": noise_markers,
        "parsing_artifacts_count": parsing_artifacts,
        "page_number_only_count": page_number_only,
    }


async def extract_fast(client: httpx.AsyncClient, pdf_path: Path) -> list[dict]:
    """Extract with Unstructured fast strategy."""
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = await client.post(
            UNSTRUCTURED_URL,
            files=files,
            data={"strategy": "fast", "output_format": "application/json"},
            timeout=300.0,
        )
    if response.status_code != 200:
        return []
    return response.json()


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 2: NOISE DETECTION")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] Using qa_run_id={qa_run_id}")

    manifests = await conn.fetch(
        "SELECT id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    total_noise = 0
    total_artifacts = 0

    async with httpx.AsyncClient() as client:
        for m in manifests:
            manifest_id = m["id"]
            filename = m["filename"]

            # Check if noise already recorded (non-zero)
            existing = await conn.fetchrow(
                """
                SELECT noise_markers_count, parsing_artifacts_count
                FROM kb.pdf_extraction_quality
                WHERE manifest_id = $1
                """,
                manifest_id,
            )
            if existing and (existing["noise_markers_count"] > 0 or existing["parsing_artifacts_count"] > 0):
                continue

            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                pdf_path = PDF_DIR / "new" / filename
            if not pdf_path.exists():
                continue

            elements = await extract_fast(client, pdf_path)
            if not elements:
                continue

            noise = count_noise(elements)

            # Update extraction quality
            await conn.execute(
                """
                UPDATE kb.pdf_extraction_quality
                SET noise_markers_count = $1,
                    parsing_artifacts_count = $2,
                    page_number_only_count = $3
                WHERE manifest_id = $4
                """,
                noise["noise_markers_count"],
                noise["parsing_artifacts_count"],
                noise["page_number_only_count"],
                manifest_id,
            )

            total_noise += noise["noise_markers_count"]
            total_artifacts += noise["parsing_artifacts_count"]

            if noise["noise_markers_count"] > 0 or noise["parsing_artifacts_count"] > 0:
                print(
                    f"  [NOISE] {filename}: "
                    f"markers={noise['noise_markers_count']}, "
                    f"artifacts={noise['parsing_artifacts_count']}, "
                    f"page_nums={noise['page_number_only_count']}"
                )

    print(f"\n{'=' * 70}")
    print(f"NOISE DETECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total noise markers: {total_noise}")
    print(f"Total parsing artifacts: {total_artifacts}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
