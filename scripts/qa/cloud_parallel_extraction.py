#!/usr/bin/env python3
"""
Cloud Parallel Extraction - Estrae tutti i 63 PDF via Unstructured Cloud API.

Usa `by_title` chunking per chunk ottimali (~2200 chars) per RAG.
Auto-split dei PDF >300 pagine prima del processing.

Usage:
    uv run python scripts/qa/cloud_parallel_extraction.py
    uv run python scripts/qa/cloud_parallel_extraction.py --dry-run
    uv run python scripts/qa/cloud_parallel_extraction.py --max-concurrent 4
"""

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import fitz  # PyMuPDF
import httpx

from qa_config import DB_URL, PDF_DIR

# ── Config ──────────────────────────────────────────────────────────
CLOUD_API_URL = "https://api.unstructured.io/general/v0/general"
CLOUD_API_KEY = "h7nQP3E52xtFGxLJwuh3guHk4ehtNL"

MAX_PAGES_CLOUD = 300  # Cloud limit
MAX_CONCURRENT = 3     # Parallel requests (be nice to API)
TIMEOUT_SECONDS = 600  # 10 min per PDF

# Cloud chunking parameters (optimal for RAG)
CLOUD_PARAMS = {
    "strategy": "hi_res",
    "chunking_strategy": "by_title",
    "max_characters": 3000,
    "new_after_n_chars": 2500,
    "combine_under_n_chars": 500,
    "output_format": "application/json",
}


@dataclass
class ExtractionResult:
    """Result of a single PDF extraction."""
    filename: str
    original_filename: str  # Original name before split
    part_number: int | None  # None if not split, 1 or 2 if split
    chunks: list[dict]
    elapsed_seconds: float
    error: str | None = None


def stable_normalize(text: str) -> str:
    """Normalizza testo per content hash stabile."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"(?:^|\n)\s*\d{1,4}\s*(?:\n|$)", "\n", t)
    t = re.sub(r"\.{3,}", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_content_hash(text_norm: str) -> str:
    """Calcola hash SHA256 del testo normalizzato."""
    return hashlib.sha256(text_norm.encode("utf-8")).hexdigest()


def has_citation(text: str) -> bool:
    """Verifica se il testo contiene citazioni giuridiche."""
    return bool(re.search(r"(?:Sez\.?|Sezione)\s*[UuLl0-9]", text, re.IGNORECASE))


def get_pdf_pages(pdf_path: Path) -> int:
    """Ritorna il numero di pagine del PDF."""
    doc = fitz.open(pdf_path)
    pages = len(doc)
    doc.close()
    return pages


def find_best_split_point(pdf_path: Path) -> int | None:
    """
    Trova il miglior punto di split vicino alla metà del PDF.
    Cerca inizi di capitolo/sezione.
    """
    chapter_patterns = [
        r"^SEZIONE\s+[IVXLCDM]+",
        r"^Sez\.\s*[IVXLCDM]+",
        r"^CAPITOLO\s+[IVXLCDM\d]+",
        r"^PARTE\s+[IVXLCDM]+",
        r"^TITOLO\s+[IVXLCDM]+",
    ]

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    middle = total_pages / 2

    boundaries = []
    seen = set()

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        lines = text.strip().split('\n')[:15]

        for line in lines:
            line_clean = line.strip()
            if not line_clean or len(line_clean) < 3:
                continue

            for pattern in chapter_patterns:
                if re.match(pattern, line_clean, re.IGNORECASE):
                    chapter_id = line_clean[:30].upper()
                    if chapter_id not in seen:
                        seen.add(chapter_id)
                        boundaries.append(page_num + 1)  # 1-based
                    break

    doc.close()

    if not boundaries:
        # Fallback: split a metà esatta
        return total_pages // 2

    # Trova boundary più vicino alla metà
    best = min(boundaries, key=lambda x: abs(x - middle))
    return best


def split_pdf(pdf_path: Path, split_page: int, output_dir: Path) -> tuple[Path, Path]:
    """Divide il PDF in due parti."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    stem = pdf_path.stem
    # Normalizza nome
    stem = re.sub(r'\s*pagg?\s*\d+', '', stem)
    stem = re.sub(r'[()[\]{}]', '', stem)
    stem = stem.replace(' ', '_')
    stem = re.sub(r'_+', '_', stem).strip('_')

    # Part 1
    part1_path = output_dir / f"{stem}_PART1.pdf"
    part1 = fitz.open()
    part1.insert_pdf(doc, from_page=0, to_page=split_page - 2)
    part1.save(part1_path)
    part1.close()

    # Part 2
    part2_path = output_dir / f"{stem}_PART2.pdf"
    part2 = fitz.open()
    part2.insert_pdf(doc, from_page=split_page - 1, to_page=total_pages - 1)
    part2.save(part2_path)
    part2.close()

    doc.close()

    return part1_path, part2_path


async def extract_cloud_chunks(
    pdf_path: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Estrae chunks via Unstructured Cloud API."""
    async with semaphore:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}

            response = await client.post(
                CLOUD_API_URL,
                files=files,
                data=CLOUD_PARAMS,
                headers={"unstructured-api-key": CLOUD_API_KEY},
                timeout=TIMEOUT_SECONDS,
            )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

        return response.json()


async def process_single_pdf(
    pdf_path: Path,
    original_filename: str,
    part_number: int | None,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> ExtractionResult:
    """Processa un singolo PDF (o parte di PDF)."""
    start = time.time()

    try:
        chunks = await extract_cloud_chunks(pdf_path, client, semaphore)
        elapsed = time.time() - start

        return ExtractionResult(
            filename=pdf_path.name,
            original_filename=original_filename,
            part_number=part_number,
            chunks=chunks,
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start
        return ExtractionResult(
            filename=pdf_path.name,
            original_filename=original_filename,
            part_number=part_number,
            chunks=[],
            elapsed_seconds=elapsed,
            error=str(e),
        )


def prepare_pdf_list(pdf_dir: Path, temp_dir: Path) -> list[tuple[Path, str, int | None]]:
    """
    Prepara la lista di PDF da processare.
    Split automatico per PDF >300 pagine.

    Returns:
        Lista di tuple (pdf_path, original_filename, part_number)
    """
    pdfs_to_process = []

    # Trova tutti i PDF
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    print(f"Trovati {len(pdf_files)} PDF in {pdf_dir}")
    print()

    for pdf_path in pdf_files:
        pages = get_pdf_pages(pdf_path)

        if pages <= MAX_PAGES_CLOUD:
            # PDF piccolo, processa direttamente
            pdfs_to_process.append((pdf_path, pdf_path.name, None))
            print(f"  [OK] {pdf_path.name}: {pages} pagine")
        else:
            # PDF grande, split necessario
            print(f"  [SPLIT] {pdf_path.name}: {pages} pagine > {MAX_PAGES_CLOUD}")

            split_page = find_best_split_point(pdf_path)
            part1, part2 = split_pdf(pdf_path, split_page, temp_dir)

            pages1 = get_pdf_pages(part1)
            pages2 = get_pdf_pages(part2)

            # Verifica che entrambe le parti siano < 300
            if pages1 > MAX_PAGES_CLOUD or pages2 > MAX_PAGES_CLOUD:
                print(f"    [WARN] Parti ancora grandi: {pages1} + {pages2}")
                # Potrebbe servire secondo split, ma per ora procediamo

            pdfs_to_process.append((part1, pdf_path.name, 1))
            pdfs_to_process.append((part2, pdf_path.name, 2))
            print(f"    -> PART1: {pages1} pagine, PART2: {pages2} pagine")

    print()
    print(f"Totale PDF da processare: {len(pdfs_to_process)}")
    return pdfs_to_process


async def save_chunks_to_db(
    conn: asyncpg.Connection,
    results: list[ExtractionResult],
    qa_run_id: int,
):
    """Salva i chunks nel database."""
    for result in results:
        if result.error or not result.chunks:
            continue

        # Trova manifest_id
        manifest_id = await conn.fetchval(
            "SELECT id FROM kb.pdf_manifest WHERE filename = $1",
            result.original_filename,
        )

        if not manifest_id:
            print(f"  [WARN] Manifest non trovato per {result.original_filename}")
            continue

        # Delete existing cloud chunks for this manifest
        await conn.execute(
            "DELETE FROM kb.qa_reference_units WHERE manifest_id = $1 AND extraction_method LIKE 'cloud%'",
            manifest_id,
        )

        # Insert new chunks
        for idx, chunk in enumerate(result.chunks):
            text = chunk.get("text", "")
            if not text or len(text) < 50:
                continue

            text_norm = stable_normalize(text)
            content_hash = compute_content_hash(text_norm)
            page_start = chunk.get("metadata", {}).get("page_number")

            # Adjust unit_index for parts
            if result.part_number:
                unit_idx = idx + (10000 * result.part_number)  # Part 1: 10000+, Part 2: 20000+
            else:
                unit_idx = idx

            await conn.execute(
                """
                INSERT INTO kb.qa_reference_units
                  (qa_run_id, manifest_id, unit_index, testo, testo_norm,
                   content_hash, char_count, page_start, page_end,
                   has_citation, extraction_method)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8, $9, $10)
                """,
                qa_run_id,
                manifest_id,
                unit_idx,
                text,
                text_norm,
                content_hash,
                len(text),
                page_start,
                has_citation(text),
                f"cloud_by_title_part{result.part_number}" if result.part_number else "cloud_by_title",
            )


async def main(dry_run: bool = False, max_concurrent: int = MAX_CONCURRENT):
    """Main entry point."""
    print("=" * 70)
    print("CLOUD PARALLEL EXTRACTION - Unstructured Cloud API")
    print("=" * 70)
    print()
    print(f"API URL: {CLOUD_API_URL}")
    print(f"Chunking: by_title (max 3000, new after 2500)")
    print(f"Concurrent requests: {max_concurrent}")
    print(f"Dry run: {dry_run}")
    print()

    # Create temp directory for split PDFs
    temp_dir = Path(tempfile.mkdtemp(prefix="cloud_extract_"))
    print(f"Temp directory: {temp_dir}")
    print()

    try:
        # Prepare PDF list (with auto-split)
        print("=" * 70)
        print("FASE 1: Preparazione PDF")
        print("=" * 70)
        pdfs_to_process = prepare_pdf_list(PDF_DIR, temp_dir)

        if dry_run:
            print()
            print("[DRY RUN] Avrei processato:")
            for pdf_path, orig_name, part in pdfs_to_process:
                part_str = f" (PART {part})" if part else ""
                print(f"  - {pdf_path.name}{part_str}")
            return

        # Connect to database
        conn = await asyncpg.connect(DB_URL)

        # Get or create qa_run_id
        qa_run_id = await conn.fetchval(
            "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
        )
        if not qa_run_id:
            qa_run_id = await conn.fetchval(
                "INSERT INTO kb.qa_runs (status, config_json) VALUES ('running', $1) RETURNING id",
                json.dumps({"type": "cloud_extraction", "strategy": "by_title"}),
            )

        print()
        print("=" * 70)
        print("FASE 2: Estrazione Cloud Parallela")
        print("=" * 70)
        print()

        # Create HTTP client with retry
        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(max_concurrent)

            # Process all PDFs
            tasks = []
            for pdf_path, orig_name, part_num in pdfs_to_process:
                task = process_single_pdf(
                    pdf_path, orig_name, part_num, client, semaphore
                )
                tasks.append(task)

            # Run with progress tracking
            results: list[ExtractionResult] = []
            total = len(tasks)

            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)

                # Progress
                status = "OK" if not result.error else "ERR"
                part_str = f" (P{result.part_number})" if result.part_number else ""
                chunks_str = f"{len(result.chunks)} chunks" if not result.error else result.error[:40]

                print(
                    f"[{i+1:2}/{total}] [{status}] {result.filename[:45]:45}{part_str}: "
                    f"{chunks_str} ({result.elapsed_seconds:.1f}s)"
                )

        # Summary
        print()
        print("=" * 70)
        print("FASE 3: Salvataggio Database")
        print("=" * 70)

        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]

        total_chunks = sum(len(r.chunks) for r in successful)
        total_chars = sum(
            sum(len(c.get("text", "")) for c in r.chunks)
            for r in successful
        )
        avg_chars = total_chars // total_chunks if total_chunks else 0

        print(f"\nSuccessful: {len(successful)}/{total}")
        print(f"Total chunks: {total_chunks}")
        print(f"Average chunk size: {avg_chars} chars")

        if failed:
            print(f"\nFailed ({len(failed)}):")
            for r in failed:
                print(f"  - {r.filename}: {r.error[:60]}")

        # Save to database
        print("\nSalvando chunks nel database...")
        await save_chunks_to_db(conn, results, qa_run_id)

        # Final verification
        cloud_count = await conn.fetchval(
            "SELECT count(*) FROM kb.qa_reference_units WHERE extraction_method LIKE 'cloud%'"
        )
        print(f"Chunks cloud nel database: {cloud_count}")

        await conn.close()

        # Group results by original file
        print()
        print("=" * 70)
        print("SUMMARY PER DOCUMENTO")
        print("=" * 70)

        by_original: dict[str, list[ExtractionResult]] = {}
        for r in successful:
            by_original.setdefault(r.original_filename, []).append(r)

        for orig_name, parts in sorted(by_original.items()):
            total_chunks = sum(len(p.chunks) for p in parts)
            total_time = sum(p.elapsed_seconds for p in parts)
            parts_str = f" ({len(parts)} parts)" if len(parts) > 1 else ""
            print(f"{orig_name[:55]:55} {total_chunks:5} chunks{parts_str} ({total_time:.0f}s)")

        print()
        print(f"TOTALE: {sum(len(r.chunks) for r in successful)} chunks, {avg_chars} avg chars")

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nTemp directory rimossa: {temp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Parallel Extraction")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostra cosa farebbe")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="Richieste parallele")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, max_concurrent=args.max_concurrent))
