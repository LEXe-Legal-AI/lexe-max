"""
KB Massimari - Recover Anno from PDFs
Extracts year from first pages for PDFs without year in filename.

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export PATH=$HOME/.local/bin:$PATH
    uv run python scripts/ingest_recover_anno.py
"""
import asyncio
import hashlib
import re
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx

# Config
DB_URL = "postgresql://leo:stage_postgres_2026_secure@localhost:5432/leo"
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
PDF_DIR = Path("/opt/leo-platform/lexe-api/data/massimari")

# Gate policy thresholds
MIN_LENGTH = 150
MAX_CITATION_RATIO = 0.03

# PDFs that need anno recovery (no year in filename)
RECOVER_PDFS = [
    "Civile Volume_1.pdf",
    "Civile Volume_2.pdf",
    "Mass civile Vol 1 pagg 408.pdf",
    "Mass civile Vol 2 pagg 280.pdf",
    "Massimario_Penale pagg 384.pdf",
    "Volume 1 massimario Civile_01_400.pdf",
    "Volume 2 massimario Civile_401_650.pdf",
    "Volume 3 Civile NUOVO.pdf",
    "Volume III_2016_Approfond_Tematici.pdf",
    "Volume III_2017_Approfond_Tematici.pdf",
    "Volume Penale.pdf",
]


def extract_anno_from_text(text: str) -> int:
    """Try to extract year from document text."""
    # Look for patterns like "2014", "anno 2015", "Rassegna 2013"
    # Prioritize years in context like "Rassegna della giurisprudenza 2014"

    # First, try to find year in "Rassegna...20XX" or "Massimario...20XX"
    match = re.search(r"(?:Rassegna|Massimario)[^\d]*(\d{4})", text, re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 2008 <= year <= 2025:
            return year

    # Try "anno 20XX"
    match = re.search(r"anno\s*(\d{4})", text, re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 2008 <= year <= 2025:
            return year

    # Try standalone years with context
    matches = re.findall(r"\b(20\d{2})\b", text)
    for year_str in matches:
        year = int(year_str)
        if 2008 <= year <= 2025:
            return year

    return 0


def parse_tipo_from_filename(filename: str) -> str:
    """Extract tipo (civile/penale) from filename."""
    if re.search(r"civile", filename, re.IGNORECASE):
        return "civile"
    elif re.search(r"penale", filename, re.IGNORECASE):
        return "penale"
    return "unknown"


def parse_volume_from_filename(filename: str) -> int:
    """Extract volume number from filename."""
    vol_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}

    # Pattern: Volume I, Volume_1, Vol 1, etc.
    match = re.search(r"Volume[_\s]*(I{1,3}V?|\d)", filename, re.IGNORECASE)
    if match:
        vol_str = match.group(1).upper()
        return vol_map.get(vol_str, 1)

    match = re.search(r"Vol[._\s]*(I{1,3}V?|\d)", filename, re.IGNORECASE)
    if match:
        vol_str = match.group(1).upper()
        return vol_map.get(vol_str, 1)

    return 1


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

    citations = len(re.findall(r"Cass\.|Sez\.\s*\d|n\.\s*\d+|Rv\.\s*\d+", text))
    words = len(text.split())
    if words > 0 and citations / words > MAX_CITATION_RATIO:
        return False

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

        if elem_type in ["Header", "Footer", "PageNumber"]:
            continue

        if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+)", text, re.IGNORECASE):
            if current_massima:
                full_text = " ".join(current_massima)
                if is_valid_massima(full_text):
                    massime.append(full_text)
            current_massima = [text]
        elif text:
            current_massima.append(text)

    if current_massima:
        full_text = " ".join(current_massima)
        if is_valid_massima(full_text):
            massime.append(full_text)

    return massime


async def ingest_pdf_with_recovery(conn: asyncpg.Connection, client: httpx.AsyncClient, pdf_path: Path):
    """Ingest PDF with anno recovery from content."""
    filename = pdf_path.name

    # Extract tipo and volume from filename
    tipo = parse_tipo_from_filename(filename)
    volume = parse_volume_from_filename(filename)

    print(f"\n[RECOVER] {filename}")
    print(f"  Tipo: {tipo}, Volume: {volume}")

    # Check if already ingested
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    existing = await conn.fetchval(
        "SELECT id FROM kb.documents WHERE source_hash = $1", source_hash
    )
    if existing:
        print(f"  [SKIP] Already ingested")
        return 0

    # Extract content
    elements = await extract_with_unstructured(client, pdf_path)
    if not elements:
        print(f"  [ERROR] No elements extracted")
        return 0

    print(f"  Extracted {len(elements)} elements")

    # Try to recover anno from first 20 elements (first pages)
    first_text = " ".join(elem.get("text", "") for elem in elements[:30])
    anno = extract_anno_from_text(first_text)

    if anno == 0:
        # Try filename patterns that might have year
        match = re.search(r"(20\d{2})", filename)
        if match:
            anno = int(match.group(1))

    if anno == 0:
        print(f"  [WARN] Could not recover anno, using 2014 as default")
        anno = 2014  # Default fallback

    print(f"  Recovered anno: {anno}")

    # Create document
    doc_id = uuid4()
    await conn.execute("""
        INSERT INTO kb.documents (id, source_path, source_hash, anno, volume, tipo, titolo, processed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    """, doc_id, str(pdf_path), source_hash, anno, volume, tipo, filename)

    # Extract and insert massime
    massime = extract_massime_from_elements(elements)
    print(f"  Found {len(massime)} massime (after gate policy)")

    count = 0
    for testo in massime:
        testo_norm = re.sub(r"\s+", " ", testo.lower().strip())
        content_hash = hashlib.sha256(testo_norm.encode()).hexdigest()

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
    print("KB MASSIMARI - RECOVER ANNO FROM PDFs")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    total_massime = 0

    async with httpx.AsyncClient() as client:
        for pdf_name in RECOVER_PDFS:
            pdf_path = PDF_DIR / pdf_name
            if not pdf_path.exists():
                print(f"\n[SKIP] File not found: {pdf_name}")
                continue

            try:
                count = await ingest_pdf_with_recovery(conn, client, pdf_path)
                total_massime += count
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

    # Final stats
    doc_count = await conn.fetchval("SELECT COUNT(*) FROM kb.documents")
    massime_count = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")

    await conn.close()

    print("\n" + "=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)
    print(f"Total documents: {doc_count}")
    print(f"Total massime: {massime_count}")
    print(f"New massime this run: {total_massime}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
