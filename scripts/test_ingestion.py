"""
Test Ingestion Script - KB Massimari
Estrae massime da PDF e le inserisce nel database.
Usa PyMuPDF per estrazione (no OCR necessario per PDF digitali).
"""
import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import fitz  # PyMuPDF

# Database connection
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# PDF directory
PDF_DIR = Path("C:/Users/Fra/Documents/lexe/collezione zecca/New folder (2)/Massimario_PDF/")

# Test files
TEST_FILES = [
    "2021_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2023_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2018_MASSIMARIO CIVILE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTIDELLE SEZIONI CIVILI.pdf",
]


def extract_year_and_type(filename: str) -> tuple[int, str]:
    """Extract year and type (CIVILE/PENALE) from filename."""
    match = re.match(r"(\d{4})_MASSIMARIO\s+(CIVILE|PENALE)", filename)
    if match:
        return int(match.group(1)), match.group(2).lower()
    return 2020, "unknown"


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_text_from_pdf(pdf_path: Path, start_page: int = 20, max_pages: int = 80) -> list[dict]:
    """Extract text from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF
        start_page: Skip intro pages (TOC, authors, etc.)
        max_pages: Maximum pages to process
    """
    doc = fitz.open(pdf_path)
    pages = []

    # Skip intro and limit pages
    end_page = min(doc.page_count, start_page + max_pages)

    for i in range(start_page, end_page):
        page = doc[i]
        text = page.get_text()
        if text.strip() and len(text) > 200:  # Skip nearly empty pages
            pages.append({
                "page_num": i + 1,
                "text": text,
            })

    doc.close()
    return pages


# Mesi italiani per parsing date
MESI_IT = {
    'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4,
    'maggio': 5, 'giugno': 6, 'luglio': 7, 'agosto': 8,
    'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12
}

# Pattern per identificare massime - formato Cassazione reale
MASSIMA_PATTERNS = [
    # Sez. 4ª, n. 6513 del 27/01/2021, Savanelli, Rv. 28093301
    re.compile(
        r"Sez\.?\s*(\d+)[ªa°]?,?\s*n\.?\s*(\d+)\s+del\s+(\d{1,2}[/]\d{1,2}[/]\d{4})",
        re.IGNORECASE
    ),
    # Sez. Un., n. 12345 del 27/01/2021
    re.compile(
        r"Sez\.?\s*(Un\.?|Unite),?\s*n\.?\s*(\d+)\s+del\s+(\d{1,2}[/]\d{1,2}[/]\d{4})",
        re.IGNORECASE
    ),
    # Cass., sez. un., 11 luglio 2011, n. 12345 (formato civile)
    re.compile(
        r"Cass\.?,?\s*sez\.?\s*(un\.?|unite|\d+)[ªa°]?,?\s*"
        r"(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4}),?\s*n\.?\s*(\d+)",
        re.IGNORECASE
    ),
    # Cass. civ., Sez. 3, 12 gennaio 2018, n. 123
    re.compile(
        r"Cass\.\s*(civ|pen)\.?,?\s*[Ss]ez\.?\s*(\d+|[Uu]n\.?|[Uu]nite),?\s*"
        r"(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4}),?\s*n\.?\s*(\d+)",
        re.IGNORECASE
    ),
]

# Pattern per citazioni normative
NORM_PATTERNS = [
    # Art. 123 c.p.c.
    re.compile(r"[Aa]rt\.?\s*(\d+)(?:\s*,?\s*(?:comma|co\.?)\s*(\d+))?\s+(c\.?p\.?c\.?|c\.?p\.?|c\.?c\.?)", re.IGNORECASE),
    # L. 241/1990
    re.compile(r"[Ll]\.?\s*(?:n\.?\s*)?(\d+)[/](\d{4})", re.IGNORECASE),
    # D.Lgs. 150/2011
    re.compile(r"[Dd]\.?\s*[Ll]gs\.?\s*(?:n\.?\s*)?(\d+)[/](\d{4})", re.IGNORECASE),
]


def extract_massime_from_text(text: str, source_type: str) -> list[dict]:
    """Extract massime from raw text."""
    massime = []
    seen_hashes = set()

    # Find all citation matches with context
    for pattern in MASSIMA_PATTERNS:
        for match in pattern.finditer(text):
            # Get context: 200 chars before and 500 after
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 500)
            context = text[start:end].strip()

            # Clean up context
            context = re.sub(r'\s+', ' ', context)

            if len(context) < 50:
                continue

            # Extract info from match
            groups = match.groups()

            # Parse sezione and numero based on pattern
            sezione = "unknown"
            numero = "0"
            data_str = None

            if len(groups) >= 3:
                sezione = str(groups[0]) if groups[0] else "unknown"
                numero = str(groups[1]) if groups[1] else "0"
                data_str = groups[2] if len(groups) > 2 else None

            # Compute hash to avoid duplicates
            content_hash = compute_hash(context)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            massima = {
                "id": str(uuid4()),
                "testo": context[:2000],
                "tipo_pronuncia": "sentenza",
                "sezione": sezione,
                "numero": numero,
                "materia": source_type,
                "hash": content_hash,
            }

            # Parse date - handle both dd/mm/yyyy and Italian written dates
            if data_str:
                try:
                    if '/' in data_str:
                        # Format: dd/mm/yyyy
                        parts = data_str.split('/')
                        if len(parts) == 3:
                            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                            massima["data_decisione"] = date(y, m, d)
                except (ValueError, IndexError):
                    pass

            # Also try to extract Italian date from context
            if "data_decisione" not in massima:
                date_match = re.search(
                    r"(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})",
                    context, re.IGNORECASE
                )
                if date_match:
                    try:
                        d = int(date_match.group(1))
                        m = MESI_IT.get(date_match.group(2).lower(), 1)
                        y = int(date_match.group(3))
                        massima["data_decisione"] = date(y, m, d)
                    except (ValueError, KeyError):
                        pass

            massime.append(massima)

    return massime


def extract_citations_from_text(text: str) -> list[dict]:
    """Extract normative citations from text."""
    citations = []

    for pattern in NORM_PATTERNS:
        for match in pattern.finditer(text):
            citation = {
                "id": str(uuid4()),
                "tipo": "norma",
                "raw_text": match.group(0),
            }
            citations.append(citation)

    return citations


async def insert_document(conn: asyncpg.Connection, filename: str, year: int, doc_type: str, volume: int = 1) -> str:
    """Insert document record."""
    doc_id = str(uuid4())
    source_hash = compute_hash(filename)

    # Check if exists
    existing = await conn.fetchval(
        "SELECT id FROM kb.documents WHERE source_hash = $1", source_hash
    )
    if existing:
        return str(existing)

    await conn.execute("""
        INSERT INTO kb.documents (id, source_path, titolo, anno, volume, tipo, source_hash, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """, doc_id, filename, filename, year, volume, doc_type, source_hash, json.dumps({}))

    return doc_id


def normalize_text(text: str) -> str:
    """Normalize text for hashing and indexing."""
    return re.sub(r'\s+', ' ', text.lower().strip())


async def insert_massima(conn: asyncpg.Connection, doc_id: str, massima: dict) -> str:
    """Insert massima record."""
    testo = massima["testo"]
    testo_normalizzato = normalize_text(testo)

    # Check if exists by hash
    existing = await conn.fetchval(
        "SELECT id FROM kb.massime WHERE content_hash = $1", massima["hash"]
    )
    if existing:
        return str(existing)

    await conn.execute("""
        INSERT INTO kb.massime (
            id, document_id, testo, testo_normalizzato, sezione, numero,
            materia, content_hash, data_decisione, anno, tipo
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    """,
        massima["id"],
        doc_id,
        testo,
        testo_normalizzato,
        massima.get("sezione", "unknown"),
        str(massima.get("numero", "0")),  # varchar, not int
        massima.get("materia", "unknown"),
        massima["hash"],
        massima.get("data_decisione"),
        massima.get("data_decisione", date.today()).year if massima.get("data_decisione") else None,
        massima.get("tipo_pronuncia", "sentenza"),
    )

    return massima["id"]


async def process_pdf(pdf_path: Path, conn: asyncpg.Connection) -> dict:
    """Process a single PDF file."""
    filename = pdf_path.name
    year, doc_type = extract_year_and_type(filename)

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"Year: {year}, Type: {doc_type}")

    # Extract text
    print("Extracting text...")
    pages = extract_text_from_pdf(pdf_path, max_pages=30)  # Limit for testing
    print(f"  Extracted {len(pages)} pages")

    # Insert document
    doc_id = await insert_document(conn, filename, year, doc_type)
    print(f"  Document ID: {doc_id}")

    # Extract and insert massime
    total_massime = 0
    total_citations = 0

    for page in pages:
        massime = extract_massime_from_text(page["text"], doc_type)
        for massima in massime:
            await insert_massima(conn, doc_id, massima)
            total_massime += 1

        citations = extract_citations_from_text(page["text"])
        total_citations += len(citations)

    print(f"  Extracted {total_massime} massime")
    print(f"  Found {total_citations} citations")

    return {
        "filename": filename,
        "doc_id": doc_id,
        "pages": len(pages),
        "massime": total_massime,
        "citations": total_citations,
    }


async def main():
    """Main entry point."""
    print("=" * 60)
    print("KB Massimari - Test Ingestion")
    print("=" * 60)

    # Connect to database
    print("\nConnecting to database...")
    conn = await asyncpg.connect(DB_URL)
    print("Connected!")

    results = []

    try:
        for filename in TEST_FILES:
            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                print(f"SKIP: {filename} not found")
                continue

            result = await process_pdf(pdf_path, conn)
            results.append(result)

    finally:
        await conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'File':<50} {'Pages':>6} {'Massime':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['filename'][:48]:<50} {r['pages']:>6} {r['massime']:>8}")

    total_massime = sum(r['massime'] for r in results)
    print("-" * 70)
    print(f"{'TOTAL':<50} {sum(r['pages'] for r in results):>6} {total_massime:>8}")

    # Verify in database
    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION")
    print("=" * 60)
    conn = await asyncpg.connect(DB_URL)

    doc_count = await conn.fetchval("SELECT COUNT(*) FROM kb.documents")
    massime_count = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")

    print(f"Documents in DB: {doc_count}")
    print(f"Massime in DB: {massime_count}")

    # Sample massima
    sample = await conn.fetchrow("""
        SELECT m.testo, m.sezione, m.materia, d.titolo
        FROM kb.massime m
        JOIN kb.documents d ON m.document_id = d.id
        LIMIT 1
    """)

    if sample:
        print(f"\nSample massima:")
        print(f"  Source: {sample['titolo'][:50]}...")
        print(f"  Sezione: {sample['sezione']}")
        print(f"  Materia: {sample['materia']}")
        print(f"  Text: {sample['testo'][:100]}...")

    await conn.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
