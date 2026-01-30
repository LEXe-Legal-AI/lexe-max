"""
Backfill pagina_inizio per sezioni senza pagina.
Cerca i titoli delle sezioni nel testo del PDF per pagina.

v2: Usa SKIP_PAGES dinamico basato sulla prima pagina delle massime.
"""
import asyncio
import re
from pathlib import Path
from uuid import UUID

import asyncpg
import httpx

# Config
UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
PDF_DIR = Path("C:/Users/Fra/Documents/lexe/collezione zecca/New folder (2)/Massimario_PDF/")

# Pagine da escludere (fallback se nessuna massima)
DEFAULT_SKIP_PAGES = 15
# Buffer prima della prima massima (il contenuto inizia prima delle massime)
CONTENT_START_BUFFER = 10


def normalize_for_match(text: str) -> str:
    """Normalizza testo per matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_match(needle: str, haystack: str, threshold: float = 0.7) -> bool:
    """
    Match fuzzy semplice basato su parole in comune.
    Ritorna True se almeno threshold% delle parole del needle sono nel haystack.
    """
    needle_words = set(normalize_for_match(needle).split())
    haystack_norm = normalize_for_match(haystack)

    if len(needle_words) < 2:
        return False

    matches = sum(1 for w in needle_words if w in haystack_norm and len(w) > 3)
    ratio = matches / len(needle_words)

    return ratio >= threshold


async def extract_pdf_pages(pdf_path: Path) -> dict[int, str]:
    """Estrae testo per pagina dal PDF."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            data = {"strategy": "fast", "languages": ["ita"]}
            response = await client.post(UNSTRUCTURED_URL, files=files, data=data)

    if response.status_code != 200:
        print(f"  ERROR: {response.status_code}")
        return {}

    elements = response.json()

    # Raggruppa testo per pagina
    pages = {}
    for elem in elements:
        page = elem.get("metadata", {}).get("page_number")
        text = elem.get("text", "")
        if page and text:
            if page not in pages:
                pages[page] = []
            pages[page].append(text)

    # Concatena testo per pagina
    return {p: " ".join(texts) for p, texts in pages.items()}


async def get_dynamic_skip_pages(conn: asyncpg.Connection, doc_id: UUID) -> int:
    """
    Calcola SKIP_PAGES dinamico basato sulla prima pagina delle massime.
    Le massime indicano dove inizia il contenuto reale (dopo l'indice).
    """
    first_page = await conn.fetchval("""
        SELECT MIN(pagina_inizio)
        FROM kb.massime
        WHERE document_id = $1 AND pagina_inizio IS NOT NULL
    """, doc_id)

    if first_page:
        # Il contenuto inizia un po' prima delle massime
        skip = max(first_page - CONTENT_START_BUFFER, DEFAULT_SKIP_PAGES)
        return skip

    return DEFAULT_SKIP_PAGES


async def find_section_page(
    section_title: str,
    pages: dict[int, str],
    skip_pages: int = DEFAULT_SKIP_PAGES
) -> int | None:
    """
    Cerca il titolo della sezione nel testo delle pagine.
    Ritorna la prima pagina dove appare (dopo skip_pages).
    """
    # Pulisci titolo
    clean_title = normalize_for_match(section_title)

    # Cerca nelle pagine ordinate
    for page_num in sorted(pages.keys()):
        if page_num <= skip_pages:
            continue

        page_text = pages[page_num]

        # Prima prova match esatto (normalizzato)
        if clean_title in normalize_for_match(page_text):
            return page_num

        # Poi fuzzy match
        if fuzzy_match(section_title, page_text, threshold=0.6):
            return page_num

    return None


async def backfill_document(conn: asyncpg.Connection, doc_id: UUID, pages: dict[int, str]):
    """Backfill pagina_inizio per tutte le sezioni usando SKIP_PAGES dinamico."""

    # Calcola SKIP_PAGES dinamico
    skip_pages = await get_dynamic_skip_pages(conn, doc_id)
    print(f"  SKIP_PAGES dinamico: {skip_pages}")

    # RESET: Prima azzera le sezioni con pagine probabilmente nel TOC
    # (se pagina < skip_pages, probabilmente era nel TOC)
    result = await conn.execute("""
        UPDATE kb.sections
        SET pagina_inizio = NULL, pagina_fine = NULL
        WHERE document_id = $1
          AND pagina_inizio IS NOT NULL
          AND pagina_inizio <= $2
    """, doc_id, skip_pages)

    # Parse "UPDATE N" result
    reset_count = int(result.split()[-1]) if result else 0
    if reset_count > 0:
        print(f"  Reset sezioni in TOC range (<= p.{skip_pages}): {reset_count}")

    # Trova sezioni senza pagina_inizio
    sections = await conn.fetch("""
        SELECT id, titolo, tipo, numero
        FROM kb.sections
        WHERE document_id = $1 AND pagina_inizio IS NULL
        ORDER BY level, id
    """, doc_id)

    print(f"  Sezioni da processare: {len(sections)}")

    updated = 0
    for section in sections:
        title = section["titolo"]

        # Cerca pagina con SKIP_PAGES dinamico
        page = await find_section_page(title, pages, skip_pages=skip_pages)

        if page:
            await conn.execute("""
                UPDATE kb.sections
                SET pagina_inizio = $1
                WHERE id = $2
            """, page, section["id"])
            updated += 1
            print(f"    [{section['tipo']}] {title[:40]}... -> p.{page}")

    print(f"  Aggiornate: {updated}/{len(sections)}")
    return updated


async def recalculate_page_ends(conn: asyncpg.Connection, doc_id: UUID):
    """Ricalcola pagina_fine per le sezioni usando ultima pagina massime."""

    # Prima trova l'ultima pagina delle massime per questo documento
    last_page = await conn.fetchval("""
        SELECT COALESCE(MAX(pagina_inizio), 500)
        FROM kb.massime
        WHERE document_id = $1 AND pagina_inizio IS NOT NULL
    """, doc_id)

    # Ricalcola usando last_page per l'ultima sezione
    await conn.execute("""
        WITH section_ranges AS (
            SELECT
                id,
                pagina_inizio,
                LEAD(pagina_inizio) OVER (
                    PARTITION BY document_id
                    ORDER BY pagina_inizio NULLS LAST, level, id
                ) as next_start
            FROM kb.sections
            WHERE document_id = $1
              AND pagina_inizio IS NOT NULL
        )
        UPDATE kb.sections s
        SET pagina_fine = CASE
            WHEN sr.next_start IS NULL THEN $2  -- Usa ultima pagina massime
            WHEN sr.next_start <= sr.pagina_inizio THEN sr.pagina_inizio
            ELSE sr.next_start - 1
        END
        FROM section_ranges sr
        WHERE s.id = sr.id
    """, doc_id, last_page)


async def relink_massime(conn: asyncpg.Connection, doc_id: UUID) -> int:
    """Re-linka le massime alle sezioni usando le nuove pagine."""

    # Reset section_id per le massime di questo documento
    await conn.execute("""
        UPDATE kb.massime
        SET section_id = NULL
        WHERE document_id = $1
    """, doc_id)

    # Re-link usando find_section_for_page
    result = await conn.execute("""
        UPDATE kb.massime m
        SET section_id = kb.find_section_for_page(m.document_id, m.pagina_inizio)
        WHERE m.document_id = $1
          AND m.pagina_inizio IS NOT NULL
    """, doc_id)

    # Conta linked
    linked = await conn.fetchval("""
        SELECT COUNT(*) FROM kb.massime
        WHERE document_id = $1 AND section_id IS NOT NULL
    """, doc_id)

    return linked


async def main():
    print("=" * 70)
    print("BACKFILL SECTION PAGES")
    print("=" * 70)

    # Check Unstructured
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:8500/healthcheck")
            print(f"[OK] Unstructured API: {resp.json()['healthcheck']}")
        except Exception as e:
            print(f"[ERR] Unstructured API: {e}")
            return

    # Connect DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Trova TUTTI i documenti con sezioni (forza reprocess per fix TOC)
    docs = await conn.fetch("""
        SELECT DISTINCT d.id, d.source_path, d.anno, d.tipo,
            (SELECT COUNT(*) FROM kb.sections s WHERE s.document_id = d.id) as total_sections,
            (SELECT COUNT(*) FROM kb.sections s WHERE s.document_id = d.id AND s.pagina_inizio IS NULL) as missing
        FROM kb.documents d
        JOIN kb.sections s ON s.document_id = d.id
        ORDER BY d.anno DESC
    """)

    print(f"\nDocumenti da processare: {len(docs)}")

    for doc in docs:
        print(f"\n{'-' * 60}")
        print(f"[{doc['anno']} {doc['tipo']}] {doc['total_sections']} sezioni totali, {doc['missing']} senza pagina")
        print(f"{'-' * 60}")

        pdf_path = PDF_DIR / doc["source_path"]
        if not pdf_path.exists():
            print(f"  [SKIP] PDF non trovato")
            continue

        # Estrai pagine
        print("  -> Estrazione pagine PDF...")
        pages = await extract_pdf_pages(pdf_path)
        print(f"     Pagine estratte: {len(pages)}")

        # Backfill
        print("  -> Backfill pagina_inizio...")
        updated = await backfill_document(conn, doc["id"], pages)

        if updated > 0:
            # Ricalcola page_end
            print("  -> Ricalcolo pagina_fine...")
            await recalculate_page_ends(conn, doc["id"])

            # Re-link massime
            print("  -> Re-link massime...")
            linked = await relink_massime(conn, doc["id"])
            print(f"     Massime linkate: {linked}")

    # Report finale
    print(f"\n{'=' * 70}")
    print("REPORT FINALE")
    print("=" * 70)

    stats = await conn.fetch("""
        SELECT d.anno, d.tipo,
            COUNT(s.id) as sections,
            COUNT(s.pagina_inizio) as with_page,
            ROUND(100.0 * COUNT(s.pagina_inizio) / COUNT(s.id), 1) as pct
        FROM kb.documents d
        JOIN kb.sections s ON s.document_id = d.id
        GROUP BY d.anno, d.tipo
        ORDER BY d.anno, d.tipo
    """)

    print("\nCopertura pagina_inizio sezioni:")
    for row in stats:
        print(f"  {row['anno']} {row['tipo']}: {row['with_page']}/{row['sections']} ({row['pct']}%)")

    # QA5 check
    qa5 = await conn.fetch("""
        SELECT d.anno, d.tipo, COUNT(*) AS fuori_range
        FROM kb.massime m
        JOIN kb.sections s ON s.id = m.section_id
        JOIN kb.documents d ON d.id = m.document_id
        WHERE m.pagina_inizio IS NOT NULL
          AND s.pagina_inizio IS NOT NULL
          AND s.pagina_fine IS NOT NULL
          AND (m.pagina_inizio < s.pagina_inizio OR m.pagina_inizio > s.pagina_fine)
        GROUP BY d.anno, d.tipo
    """)

    print("\nQA5 - Massime fuori range:")
    for row in qa5:
        print(f"  {row['anno']} {row['tipo']}: {row['fuori_range']}")

    # QA8 check
    qa8 = await conn.fetch("""
        SELECT d.anno, d.tipo, COUNT(*) AS tot,
            COUNT(m.section_id) AS linked,
            ROUND(100.0 * COUNT(m.section_id) / COUNT(*), 1) AS pct
        FROM kb.massime m
        JOIN kb.documents d ON d.id = m.document_id
        GROUP BY d.anno, d.tipo
        ORDER BY d.anno, d.tipo
    """)

    print("\nQA8 - Linking massime:")
    for row in qa8:
        print(f"  {row['anno']} {row['tipo']}: {row['linked']}/{row['tot']} ({row['pct']}%)")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
