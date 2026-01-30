"""
KB Massimari - Pipeline End-to-End
==================================
Flusso completo: ingestion massime -> link sezioni -> report QA

Segue il piano operativo:
1. Standardizza content_hash (SHA256 completo)
2. Re-ingestion massime con page_number
3. Calcola page_end per sezioni
4. Link massime -> sezioni via pagina
5. Fallback FTS quando pagina manca
6. Report finale per anno/tipo
"""
import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
import httpx

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

UNSTRUCTURED_URL = "http://localhost:8500/general/v0/general"
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
PDF_DIR = Path("C:/Users/Fra/Documents/lexe/collezione zecca/New folder (2)/Massimario_PDF/")

# PDF da processare
PDF_FILES = [
    "2021_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2023_MASSIMARIO PENALE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTI DELLE SEZIONI PENALI.pdf",
    "2018_MASSIMARIO CIVILE VOL. 1 - RASSEGNA DELLA GIURISPRUDENZA DI LEGITTIMITÀ - GLI ORIENTAMENTIDELLE SEZIONI CIVILI.pdf",
]

# Pattern per estrazione massime (space-tolerant per Unstructured)
MASSIMA_PATTERNS = [
    # Sez. 4, n. 6513 del 27/01/2021
    re.compile(
        r"Sez\s*\.?\s*(\d+|Un\.?|Unite)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d+)\s+del\s+"
        r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.IGNORECASE
    ),
    # Sez. 4 n. 12345 (senza data)
    re.compile(
        r"Sez\s*\.?\s*(\d+|Un\.?|Unite)\s*[ªa°]?\s*,?\s*n\s*\.?\s*(\d{4,})",
        re.IGNORECASE
    ),
]


# =============================================================================
# FUNZIONI UTILITY
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalizza testo per hashing e confronto."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def compute_content_hash(text: str) -> str:
    """
    Calcola content_hash STANDARDIZZATO.
    - SHA256 completo (64 caratteri)
    - Da testo normalizzato
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def compute_fingerprint(content_hash: str) -> str:
    """Fingerprint corto per debug (primi 12 caratteri)."""
    return content_hash[:12]


def extract_year_type_volume(filename: str) -> tuple[int, str, int]:
    """Estrae anno, tipo e volume dal nome file."""
    match = re.match(r"(\d{4})_MASSIMARIO\s+(CIVILE|PENALE)\s+VOL\.?\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower(), int(match.group(3))
    return 2020, "unknown", 1


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ExtractedMassima:
    """Massima estratta da PDF."""
    sezione: str
    numero: str
    data_str: Optional[str]
    testo: str
    pagina_inizio: int
    pagina_fine: Optional[int]
    content_hash: str
    fingerprint: str


# =============================================================================
# STEP 1: ESTRAZIONE CON UNSTRUCTURED
# =============================================================================

async def extract_pdf_elements(pdf_path: Path) -> list[dict]:
    """Estrae elementi da PDF usando Unstructured API."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            data = {"strategy": "fast", "languages": ["ita"]}
            response = await client.post(UNSTRUCTURED_URL, files=files, data=data)

    if response.status_code != 200:
        print(f"  ERROR Unstructured: {response.status_code}")
        return []

    return response.json()


def is_valid_massima(testo: str, match_pos: int, full_text: str) -> bool:
    """
    Gate policy: verifica se il testo e' una massima valida.
    Filtra citazioni interne, liste di riferimenti, frammenti.
    """
    # 1. Lunghezza minima
    if len(testo) < 150:
        return False

    # 2. Non deve essere solo una lista di citazioni
    #    Pattern: troppe "Sez." o "Rv." rispetto al testo
    sez_count = len(re.findall(r'Sez\s*\.', testo, re.IGNORECASE))
    rv_count = len(re.findall(r'Rv\s*\.', testo, re.IGNORECASE))
    words = len(testo.split())

    # Se piu' di 1 citazione ogni 30 parole, e' una lista
    if words > 0 and (sez_count + rv_count) / words > 0.03:
        return False

    # 3. Deve contenere parole chiave tipiche di massima
    keywords = [
        r'\b(in tema di|secondo cui|ha affermato|ha precisato|ha ribadito)\b',
        r'\b(il principio|la sentenza|la corte|le sezioni unite)\b',
        r'\b(configurabilit|sussistenza|esclusione|integra|costituisce)\b',
        r'\b(deve ritenersi|non puo|e necessario|si applica)\b',
    ]
    has_keyword = any(re.search(kw, testo, re.IGNORECASE) for kw in keywords)

    # Se < 300 chars e nessuna keyword, probabile frammento
    if len(testo) < 300 and not has_keyword:
        return False

    # 4. Non deve iniziare con pattern tipici di citazione interna
    bad_starts = [
        r'^[\s,;]\s*del\s+\d',  # ", del 28/02/2020"
        r'^[\s,;]\s*dep\.',     # ", dep. 2017"
        r'^[\s,;]\s*Rv\.',      # ", Rv. 123456"
        r'^\s*\d{4,}\s*[;,)]',  # "123456-01)."
    ]
    for bad in bad_starts:
        if re.match(bad, testo, re.IGNORECASE):
            return False

    # 5. Il match deve essere vicino all'inizio del paragrafo
    #    Se "Sez n" appare dopo 200+ caratteri, e' citazione interna
    if match_pos > 200:
        return False

    return True


def extract_massime_from_elements(elements: list[dict]) -> list[ExtractedMassima]:
    """
    Estrae massime dagli elementi con page_number.
    Applica gate policy per filtrare falsi positivi.
    """
    massime = []
    seen_hashes = set()
    rejected = {"short": 0, "citation_list": 0, "no_keyword": 0, "bad_start": 0, "late_match": 0}

    for elem in elements:
        text = elem.get("text", "")
        page = elem.get("metadata", {}).get("page_number")

        if not text or len(text) < 50 or not page:
            continue

        # Prova ogni pattern
        for pattern in MASSIMA_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                sezione = groups[0] if groups else ""
                numero = groups[1] if len(groups) > 1 else ""
                data_str = groups[2] if len(groups) > 2 else None

                # Testo dopo il match
                testo = text[match.end():].strip()

                # GATE POLICY: verifica validita'
                if not is_valid_massima(testo, match.start(), text):
                    continue

                # Limita lunghezza
                testo = testo[:2000]

                # Hash e deduplica
                content_hash = compute_content_hash(testo)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                massime.append(ExtractedMassima(
                    sezione=sezione,
                    numero=numero,
                    data_str=data_str,
                    testo=testo,
                    pagina_inizio=page,
                    pagina_fine=page,
                    content_hash=content_hash,
                    fingerprint=compute_fingerprint(content_hash),
                ))
                break

    return massime


# =============================================================================
# STEP 2: CALCOLA PAGE_END PER SEZIONI
# =============================================================================

async def calculate_section_page_ends(conn: asyncpg.Connection, doc_id: UUID):
    """
    Calcola pagina_fine per ogni sezione.
    pagina_fine = pagina_inizio della prossima sezione - 1
    Se pagina_fine <= pagina_inizio, usa pagina_inizio (sezione di 1 pagina)
    """
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
            WHEN sr.next_start IS NULL THEN sr.pagina_inizio + 10  -- Ultima sezione: assume 10 pagine
            WHEN sr.next_start - 1 < sr.pagina_inizio THEN sr.pagina_inizio  -- Stessa pagina
            ELSE sr.next_start - 1
        END
        FROM section_ranges sr
        WHERE s.id = sr.id
    """, doc_id)


# =============================================================================
# STEP 3: UPSERT MASSIME
# =============================================================================

async def upsert_massima(
    conn: asyncpg.Connection,
    doc_id: UUID,
    massima: ExtractedMassima,
    materia: str
) -> tuple[UUID, bool]:
    """
    Inserisce o aggiorna massima.
    Ritorna (massima_id, is_new).
    """
    # Cerca esistente per content_hash
    existing = await conn.fetchrow("""
        SELECT id, pagina_inizio FROM kb.massime
        WHERE document_id = $1 AND content_hash = $2
    """, doc_id, massima.content_hash)

    if existing:
        # Aggiorna pagina se mancante
        if existing["pagina_inizio"] is None and massima.pagina_inizio:
            await conn.execute("""
                UPDATE kb.massime
                SET pagina_inizio = $1, pagina_fine = $2, updated_at = NOW()
                WHERE id = $3
            """, massima.pagina_inizio, massima.pagina_fine, existing["id"])
        return existing["id"], False

    # Inserisci nuova
    massima_id = uuid4()
    testo_norm = normalize_text(massima.testo)

    await conn.execute("""
        INSERT INTO kb.massime (
            id, document_id, sezione, numero, testo, testo_normalizzato,
            content_hash, pagina_inizio, pagina_fine, materia, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
    """,
        massima_id, doc_id, massima.sezione, massima.numero,
        massima.testo, testo_norm, massima.content_hash,
        massima.pagina_inizio, massima.pagina_fine, materia
    )
    return massima_id, True


# =============================================================================
# STEP 4: LINK MASSIME -> SEZIONI
# =============================================================================

async def link_massima_to_section_by_page(
    conn: asyncpg.Connection,
    massima_id: UUID,
    doc_id: UUID,
    page: int
) -> bool:
    """Link massima a sezione usando pagina."""
    section_id = await conn.fetchval("""
        SELECT kb.find_section_for_page($1, $2)
    """, doc_id, page)

    if section_id:
        await conn.execute("""
            UPDATE kb.massime
            SET section_id = $1, updated_at = NOW()
            WHERE id = $2
        """, section_id, massima_id)
        return True
    return False


async def link_massima_to_section_by_fts(
    conn: asyncpg.Connection,
    massima_id: UUID,
    doc_id: UUID,
    testo: str
) -> bool:
    """
    Fallback: link massima a sezione usando FTS.
    Solo se pagina manca.
    """
    # Estrai parole chiave
    words = re.findall(r'\b[a-zA-Zàèéìòù]{5,}\b', testo[:300])
    if len(words) < 3:
        return False

    # Query FTS
    search_query = ' & '.join(words[:5])

    section_id = await conn.fetchval("""
        SELECT s.id
        FROM kb.sections s
        WHERE s.document_id = $1
          AND to_tsvector('italian', s.titolo) @@ to_tsquery('italian', $2)
        ORDER BY s.level DESC  -- Preferisci sezioni più specifiche
        LIMIT 1
    """, doc_id, search_query)

    if section_id:
        await conn.execute("""
            UPDATE kb.massime
            SET section_id = $1, updated_at = NOW()
            WHERE id = $2
        """, section_id, massima_id)
        return True
    return False


async def link_all_massime(conn: asyncpg.Connection, doc_id: UUID) -> tuple[int, int]:
    """
    Link tutte le massime del documento alle sezioni.
    Ritorna (linked_by_page, linked_by_fts).
    """
    linked_page = 0
    linked_fts = 0

    # Massime senza section_id
    rows = await conn.fetch("""
        SELECT id, pagina_inizio, testo
        FROM kb.massime
        WHERE document_id = $1 AND section_id IS NULL
    """, doc_id)

    for row in rows:
        if row["pagina_inizio"]:
            # Prima prova con pagina
            if await link_massima_to_section_by_page(
                conn, row["id"], doc_id, row["pagina_inizio"]
            ):
                linked_page += 1
                continue

        # Fallback FTS
        if await link_massima_to_section_by_fts(
            conn, row["id"], doc_id, row["testo"]
        ):
            linked_fts += 1

    return linked_page, linked_fts


# =============================================================================
# STEP 5: REPORT QA
# =============================================================================

async def generate_qa_report(conn: asyncpg.Connection) -> str:
    """Genera report QA finale."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("REPORT QA - KB MASSIMARI")
    report.append("=" * 70)

    # Stats per documento
    stats = await conn.fetch("""
        SELECT
            d.anno, d.tipo,
            COUNT(m.id) as massime_tot,
            COUNT(m.pagina_inizio) as con_pagina,
            COUNT(m.section_id) as con_sezione,
            ROUND(100.0 * COUNT(m.section_id) / NULLIF(COUNT(m.id), 0), 1) as pct_linked
        FROM kb.documents d
        LEFT JOIN kb.massime m ON m.document_id = d.id
        GROUP BY d.anno, d.tipo
        ORDER BY d.anno, d.tipo
    """)

    report.append("\nMASSIME PER DOCUMENTO:")
    report.append("-" * 70)
    report.append(f"{'Anno':<6} {'Tipo':<8} {'Totale':>8} {'ConPag':>8} {'ConSez':>8} {'%Link':>8}")
    report.append("-" * 70)

    for row in stats:
        report.append(
            f"{row['anno']:<6} {row['tipo']:<8} {row['massime_tot']:>8} "
            f"{row['con_pagina']:>8} {row['con_sezione']:>8} {row['pct_linked'] or 0:>7.1f}%"
        )

    # Sezioni per documento
    section_stats = await conn.fetch("""
        SELECT
            d.anno, d.tipo,
            COUNT(s.id) as sezioni_tot,
            COUNT(s.pagina_inizio) as con_pagina,
            ROUND(100.0 * COUNT(s.pagina_inizio) / NULLIF(COUNT(s.id), 0), 1) as pct_page
        FROM kb.documents d
        LEFT JOIN kb.sections s ON s.document_id = d.id
        GROUP BY d.anno, d.tipo
        ORDER BY d.anno, d.tipo
    """)

    report.append("\nSEZIONI PER DOCUMENTO:")
    report.append("-" * 70)
    report.append(f"{'Anno':<6} {'Tipo':<8} {'Totale':>8} {'ConPag':>8} {'%Page':>8}")
    report.append("-" * 70)

    for row in section_stats:
        report.append(
            f"{row['anno']:<6} {row['tipo']:<8} {row['sezioni_tot']:>8} "
            f"{row['con_pagina']:>8} {row['pct_page'] or 0:>7.1f}%"
        )

    # Controlli qualità
    report.append("\nCONTROLLI QUALITA:")
    report.append("-" * 70)

    # Massime fuori range sezione
    out_of_range = await conn.fetchval("""
        SELECT COUNT(*)
        FROM kb.massime m
        JOIN kb.sections s ON m.section_id = s.id
        WHERE m.pagina_inizio IS NOT NULL
          AND s.pagina_inizio IS NOT NULL
          AND s.pagina_fine IS NOT NULL
          AND (m.pagina_inizio < s.pagina_inizio OR m.pagina_inizio > s.pagina_fine)
    """)
    report.append(f"  Massime fuori range sezione: {out_of_range}")

    # Massime duplicate (stesso hash)
    duplicates = await conn.fetchval("""
        SELECT COUNT(*) - COUNT(DISTINCT content_hash)
        FROM kb.massime
    """)
    report.append(f"  Massime duplicate (hash): {duplicates}")

    # Sezioni senza massime
    empty_sections = await conn.fetchval("""
        SELECT COUNT(*)
        FROM kb.sections s
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.massime m WHERE m.section_id = s.id
        )
    """)
    report.append(f"  Sezioni senza massime: {empty_sections}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)


# =============================================================================
# PIPELINE PRINCIPALE
# =============================================================================

async def process_document(conn: asyncpg.Connection, pdf_path: Path, filename: str):
    """Processa un singolo documento."""
    print(f"\n{'-' * 60}")
    print(f"[PDF] {filename[:55]}...")
    print(f"{'-' * 60}")

    # Trova documento in DB
    doc_row = await conn.fetchrow("""
        SELECT id FROM kb.documents WHERE source_path = $1
    """, filename)

    if not doc_row:
        print(f"  [WARN] Documento non trovato in DB")
        return

    doc_id = doc_row["id"]
    anno, tipo, volume = extract_year_type_volume(filename)

    # Step 1: Estrai elementi da PDF
    print(f"  -> Estrazione PDF con Unstructured...")
    elements = await extract_pdf_elements(pdf_path)
    print(f"    Elementi: {len(elements)}")

    # Step 2: Estrai massime
    massime = extract_massime_from_elements(elements)
    print(f"    Massime trovate: {len(massime)}")

    # Step 3: Calcola page_end per sezioni
    print(f"  -> Calcolo page_end sezioni...")
    await calculate_section_page_ends(conn, doc_id)

    # Step 4: Upsert massime
    print(f"  -> Upsert massime...")
    new_count = 0
    updated_count = 0
    for m in massime:
        _, is_new = await upsert_massima(conn, doc_id, m, tipo)
        if is_new:
            new_count += 1
        else:
            updated_count += 1

    print(f"    Nuove: {new_count}, Aggiornate: {updated_count}")

    # Step 5: Link a sezioni
    print(f"  -> Link massime -> sezioni...")
    linked_page, linked_fts = await link_all_massime(conn, doc_id)
    print(f"    Via pagina: {linked_page}, Via FTS: {linked_fts}")


async def main():
    print("\n" + "=" * 70)
    print("KB MASSIMARI - PIPELINE END-TO-END")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check Unstructured
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:8500/healthcheck")
            status = resp.json().get('healthcheck', 'unknown')
            print(f"\n[OK] Unstructured API: {status}")
        except Exception as e:
            print(f"\n[ERR] Unstructured API non disponibile: {e}")
            return

    # Connect DB
    conn = await asyncpg.connect(DB_URL)
    print(f"[OK] Database: Connected")

    # Stato iniziale
    initial = await conn.fetchrow("""
        SELECT COUNT(*) as massime, COUNT(section_id) as linked
        FROM kb.massime
    """)
    print(f"\nStato iniziale: {initial['massime']} massime, {initial['linked']} linked")

    # Processa ogni PDF
    for filename in PDF_FILES:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            print(f"\n[SKIP] {filename[:50]}... (non trovato)")
            continue

        await process_document(conn, pdf_path, filename)

    # Report finale
    report = await generate_qa_report(conn)
    print(report)

    await conn.close()
    print("\n[DONE] Pipeline completata!")


if __name__ == "__main__":
    asyncio.run(main())
