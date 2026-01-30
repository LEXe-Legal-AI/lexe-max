"""
KB Massimari - Staging Ingestion Script
Runs the full pipeline on staging server.

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export PATH=$HOME/.local/bin:$PATH
    uv run python scripts/ingest_staging.py
"""
import asyncio
import hashlib
import os
import re
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx

# Staging config
DB_URL = "postgresql://leo:stage_postgres_2026_secure@localhost:5432/leo"
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
PDF_DIR = Path("/opt/leo-platform/lexe-api/data/massimari")

# Gate policy thresholds
MIN_LENGTH = 150
MAX_CITATION_RATIO = 0.03


def parse_filename(filename: str) -> tuple[int, str, int]:
    """Extract anno, tipo, volume from filename."""
    vol_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "unico": 1, "1": 1, "2": 2, "3": 3, "4": 4}

    # Detect tipo (civile/penale)
    tipo = "unknown"
    if re.search(r"civile", filename, re.IGNORECASE):
        tipo = "civile"
    elif re.search(r"penale", filename, re.IGNORECASE):
        tipo = "penale"

    # Extract anno (4-digit year)
    anno_match = re.search(r"(20\d{2})", filename)
    anno = int(anno_match.group(1)) if anno_match else 0

    # Extract volume
    volume = 1
    # Pattern: Volume I_, Volume_I_, Volume 1, Vol. 1, Vol 1, vol_IV
    vol_match = re.search(r"Volume[_\s]+(I{1,3}V?|unico|\d)", filename, re.IGNORECASE)
    if vol_match:
        vol_str = vol_match.group(1).upper()
        volume = vol_map.get(vol_str, 1)
    # Pattern: Vol. 1, Vol 1, vol_IV
    vol_match2 = re.search(r"Vol[._\s]+(I{1,3}V?|\d)", filename, re.IGNORECASE)
    if vol_match2:
        vol_str = vol_match2.group(1).upper()
        volume = vol_map.get(vol_str, 1)
    # Pattern: - I volume, - II volume
    vol_match3 = re.search(r"-\s*(I{1,3}V?)\s+volume", filename, re.IGNORECASE)
    if vol_match3:
        vol_str = vol_match3.group(1).upper()
        volume = vol_map.get(vol_str, 1)

    # Old pattern: 2023_MASSIMARIO CIVILE VOL. 1
    match = re.match(r"(\d{4})_MASSIMARIO\s+(CIVILE|PENALE)\s+VOL\.?\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower(), int(match.group(3))

    # If we found tipo and anno, return them
    if tipo != "unknown" and anno > 0:
        return anno, tipo, volume

    return 0, "unknown", 0


async def extract_with_unstructured(client: httpx.AsyncClient, pdf_path: Path) -> list[dict]:
    """Extract text from PDF using Unstructured API."""
    print(f"  Extracting: {pdf_path.name}")

    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = await client.post(
            UNSTRUCTURED_URL,
            files=files,
            data={"strategy": "fast", "output_format": "application/json"},
            timeout=300.0,
        )

    if response.status_code != 200:
        print(f"  [ERROR] Unstructured API: {response.status_code}")
        return []

    return response.json()


def is_valid_massima(text: str) -> bool:
    """Gate policy filter."""
    if len(text) < MIN_LENGTH:
        return False

    # Count citation patterns
    citations = len(re.findall(r"Cass\.|Sez\.\s*\d|n\.\s*\d+|Rv\.\s*\d+", text))
    words = len(text.split())
    if words > 0 and citations / words > MAX_CITATION_RATIO:
        return False

    # Bad starts
    bad_starts = [", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO"]
    for bad in bad_starts:
        if text.strip().startswith(bad):
            return False

    return True


def extract_massime_from_elements(elements: list[dict]) -> list[str]:
    """Extract massime from Unstructured elements."""
    massime = []
    current_massima = []

    for elem in elements:
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")

        # Skip headers, footers, page numbers
        if elem_type in ["Header", "Footer", "PageNumber"]:
            continue

        # Massima delimiter pattern (usually starts with section number)
        if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+)", text, re.IGNORECASE):
            if current_massima:
                full_text = " ".join(current_massima)
                if is_valid_massima(full_text):
                    massime.append(full_text)
            current_massima = [text]
        elif text:
            current_massima.append(text)

    # Last massima
    if current_massima:
        full_text = " ".join(current_massima)
        if is_valid_massima(full_text):
            massime.append(full_text)

    return massime


async def ingest_pdf(conn: asyncpg.Connection, client: httpx.AsyncClient, pdf_path: Path):
    """Ingest a single PDF."""
    filename = pdf_path.name
    anno, tipo, volume = parse_filename(filename)

    if anno == 0:
        print(f"  [SKIP] Cannot parse: {filename}")
        return 0

    print(f"\n[{anno} {tipo} vol.{volume}] {filename[:50]}...")

    # Check if already ingested
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    existing = await conn.fetchval(
        "SELECT id FROM kb.documents WHERE source_hash = $1", source_hash
    )
    if existing:
        print(f"  [SKIP] Already ingested")
        return 0

    # Extract with Unstructured
    elements = await extract_with_unstructured(client, pdf_path)
    if not elements:
        print(f"  [ERROR] No elements extracted")
        return 0

    print(f"  Extracted {len(elements)} elements")

    # Create document
    doc_id = uuid4()
    await conn.execute("""
        INSERT INTO kb.documents (id, source_path, source_hash, anno, volume, tipo, titolo, processed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    """, doc_id, str(pdf_path), source_hash, anno, volume, tipo, filename)

    # Extract massime
    massime = extract_massime_from_elements(elements)
    print(f"  Found {len(massime)} massime (after gate policy)")

    # Insert massime
    count = 0
    for testo in massime:
        testo_norm = re.sub(r"\s+", " ", testo.lower().strip())
        content_hash = hashlib.sha256(testo_norm.encode()).hexdigest()

        # Skip duplicates
        exists = await conn.fetchval(
            "SELECT 1 FROM kb.massime WHERE content_hash = $1", content_hash
        )
        if exists:
            continue

        await conn.execute("""
            INSERT INTO kb.massime (id, document_id, testo, testo_normalizzato, content_hash, anno, tipo)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, uuid4(), doc_id, testo, testo_norm, content_hash, anno, tipo)
        count += 1

    print(f"  Inserted {count} new massime")
    return count


async def main():
    print("=" * 70)
    print("KB MASSIMARI - STAGING INGESTION")
    print("=" * 70)

    # Get all PDFs
    pdfs = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("new/*.pdf"))
    print(f"Found {len(pdfs)} PDFs")

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    total_massime = 0

    async with httpx.AsyncClient() as client:
        for pdf_path in sorted(pdfs):
            try:
                count = await ingest_pdf(conn, client, pdf_path)
                total_massime += count
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

    # Final stats
    doc_count = await conn.fetchval("SELECT COUNT(*) FROM kb.documents")
    massime_count = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")

    await conn.close()

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Documents: {doc_count}")
    print(f"Massime: {massime_count}")
    print(f"New massime this run: {total_massime}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
