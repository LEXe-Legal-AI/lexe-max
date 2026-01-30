"""
Link Massime to Sections
Estrae le massime con Unstructured per ottenere page_number,
poi collega alle sezioni usando kb.find_section_for_page().
"""
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Optional
from uuid import UUID

import asyncpg
import httpx

# Config
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
PDF_DIR = Path("C:/Users/Fra/Documents/lexe/collezione zecca/New folder (2)/Massimario_PDF/")

TEST_FILES = [
    "2021_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2023_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2018_MASSIMARIO CIVILE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTIDELLE SEZIONI CIVILI.pdf",
]

# Space-tolerant patterns for Unstructured output
MASSIMA_PATTERNS = [
    # Sez. 4, n. 6513 del 27/01/2021 (with spaces)
    re.compile(
        r"Sez\s*\.?\s*(\d+)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+"
        r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.IGNORECASE
    ),
    # Sez. Un., n. 12345 del 27/01/2021
    re.compile(
        r"Sez\s*\.?\s*(Un\.?|Unite)\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+"
        r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.IGNORECASE
    ),
    # Simple: Sez X n Y (no date)
    re.compile(
        r"Sez\s*\.?\s*(\d+)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d{3,})",
        re.IGNORECASE
    ),
]


def compute_hash(text: str) -> str:
    """Compute content hash for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


async def extract_with_unstructured(pdf_path: Path) -> list[dict]:
    """Extract PDF elements using Unstructured API."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            data = {"strategy": "fast", "languages": ["ita"]}
            response = await client.post(UNSTRUCTURED_URL, files=files, data=data)

    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return []

    return response.json()


def extract_massime_with_pages(elements: list[dict]) -> list[dict]:
    """Extract massime from elements with page numbers."""
    massime = []
    seen_hashes = set()

    for e in elements:
        text = e.get("text", "")
        page = e.get("metadata", {}).get("page_number")

        if not text or len(text) < 50:
            continue

        # Try each pattern
        for pattern in MASSIMA_PATTERNS:
            match = pattern.search(text)
            if match:
                # Extract massima info
                groups = match.groups()
                sezione = groups[0] if groups else None
                numero = groups[1] if len(groups) > 1 else None

                # Get the text after the match (the actual massima content)
                start_pos = match.end()
                massima_text = text[start_pos:].strip()

                # Skip if too short
                if len(massima_text) < 30:
                    continue

                # Deduplicate
                content_hash = compute_hash(massima_text)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                massime.append({
                    "sezione": sezione,
                    "numero": numero,
                    "testo": massima_text[:2000],  # Limit length
                    "pagina": page,
                    "content_hash": content_hash,
                })
                break  # Only match first pattern

    return massime


async def find_document_id(conn: asyncpg.Connection, filename: str) -> Optional[UUID]:
    """Find document_id by source_path filename."""
    row = await conn.fetchrow(
        "SELECT id FROM kb.documents WHERE source_path = $1",
        filename
    )
    return row["id"] if row else None


async def update_massima_page_by_text(conn: asyncpg.Connection, doc_id: UUID, testo: str, page: int) -> bool:
    """Update massima with page number by matching text content using FTS."""
    # Extract key words for FTS search
    words = re.findall(r'\b[a-zA-Zàèéìòù]{4,}\b', testo[:200])
    if len(words) < 3:
        return False

    # Build FTS query with first 5 significant words
    search_words = ' & '.join(words[:5])

    # Use FTS for fuzzy matching
    result = await conn.execute("""
        UPDATE kb.massime
        SET pagina_inizio = $1
        WHERE id = (
            SELECT id FROM kb.massime
            WHERE document_id = $2
              AND tsv_italian @@ to_tsquery('italian', $3)
              AND pagina_inizio IS NULL
            LIMIT 1
        )
    """, page, doc_id, search_words)
    return "UPDATE 1" in result


async def link_massima_to_section(conn: asyncpg.Connection, massima_id: UUID, doc_id: UUID, page: int) -> bool:
    """Link massima to section using page number."""
    section_id = await conn.fetchval("""
        SELECT kb.find_section_for_page($1, $2)
    """, doc_id, page)

    if section_id:
        await conn.execute("""
            UPDATE kb.massime SET section_id = $1 WHERE id = $2
        """, section_id, massima_id)
        return True
    return False


async def process_pdf(conn: asyncpg.Connection, pdf_path: Path, filename: str):
    """Process a single PDF: extract massime, update pages, link sections."""
    print(f"\nProcessing: {filename[:60]}...")

    # Find document
    doc_id = await find_document_id(conn, filename)
    if not doc_id:
        print(f"  WARNING: Document not found in DB")
        return 0, 0

    # Extract elements
    elements = await extract_with_unstructured(pdf_path)
    print(f"  Elements extracted: {len(elements)}")

    # Extract massime with pages
    massime = extract_massime_with_pages(elements)
    print(f"  Massime found: {len(massime)}")

    # Update pages and link to sections
    updated = 0
    linked = 0

    for m in massime:
        if m["pagina"] and m["testo"]:
            # Try to update existing massima with page by text match
            if await update_massima_page_by_text(conn, doc_id, m["testo"], m["pagina"]):
                updated += 1

    # Now link all massime with pages to sections
    rows = await conn.fetch("""
        SELECT id, pagina_inizio
        FROM kb.massime
        WHERE document_id = $1
          AND pagina_inizio IS NOT NULL
          AND section_id IS NULL
    """, doc_id)

    for row in rows:
        if await link_massima_to_section(conn, row["id"], doc_id, row["pagina_inizio"]):
            linked += 1

    print(f"  Pages updated: {updated}")
    print(f"  Linked to sections: {linked}")

    return updated, linked


async def main():
    print("="*70)
    print("KB Massimari - Link Massime to Sections")
    print("="*70)

    # Check Unstructured API
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:8500/healthcheck")
            print(f"Unstructured API: {resp.json()['healthcheck']}")
        except Exception as e:
            print(f"ERROR: Unstructured API not available: {e}")
            return

    # Connect to database
    conn = await asyncpg.connect(DB_URL)
    print(f"Database: Connected")

    # Check current state
    stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(pagina_inizio) as with_page,
            COUNT(section_id) as with_section
        FROM kb.massime
    """)
    print(f"\nCurrent state:")
    print(f"  Total massime: {stats['total']}")
    print(f"  With page: {stats['with_page']}")
    print(f"  With section: {stats['with_section']}")

    total_updated = 0
    total_linked = 0

    for filename in TEST_FILES:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            print(f"SKIP: {filename} not found")
            continue

        updated, linked = await process_pdf(conn, pdf_path, filename)
        total_updated += updated
        total_linked += linked

    # Final stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Pages updated: {total_updated}")
    print(f"Sections linked: {total_linked}")

    # Check final state
    final_stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(pagina_inizio) as with_page,
            COUNT(section_id) as with_section
        FROM kb.massime
    """)
    print(f"\nFinal state:")
    print(f"  Total massime: {final_stats['total']}")
    print(f"  With page: {final_stats['with_page']}")
    print(f"  With section: {final_stats['with_section']}")

    await conn.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
