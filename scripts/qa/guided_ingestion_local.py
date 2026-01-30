#!/usr/bin/env python3
"""
Guided Ingestion - Versione locale con PyMuPDF.

Estrae massime da tutti i 63 PDF usando PyMuPDF e le salva in kb.massime.
Applica i profili assegnati dal Doc Intelligence A/B test.

Usage:
    uv run python scripts/qa/guided_ingestion_local.py
    uv run python scripts/qa/guided_ingestion_local.py --pdf "Volume I_2020.pdf"
    uv run python scripts/qa/guided_ingestion_local.py --limit 5
"""

import argparse
import asyncio
import hashlib
import re
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qa_config import DB_URL, PDF_DIR

# Gate configuration per profile
PROFILE_GATE_CONFIG = {
    "structured_by_title": {"min_length": 150, "max_citation_ratio": 0.03},
    "structured_parent_child": {"min_length": 150, "max_citation_ratio": 0.03},
    "baseline_toc_filter": {"min_length": 150, "max_citation_ratio": 0.03, "skip_toc": True},
    "legacy_layout": {"min_length": 120, "max_citation_ratio": 0.03},
    "list_pure": {"min_length": 100, "max_citation_ratio": 0.05},
    "mixed_hybrid": {"min_length": 150, "max_citation_ratio": 0.03},
}

# Patterns for detecting massime
MASSIMA_START_PATTERN = re.compile(
    r"(?:^|\n)"
    r"(?:Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI)|\d{5,6})"
    r"|(?:Cass\.|Cassazione)"
    r"|(?:sent\.\s*n\.?\s*\d+)"
    r"|(?:ord\.\s*n\.?\s*\d+)",
    re.IGNORECASE | re.MULTILINE,
)

CITATION_PATTERN = re.compile(
    r"(?:Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI)\s*,?\s*(?:sent\.?|ord\.?)?\s*n\.?\s*\d+)"
    r"|(?:Rv\.?\s*\d{6})"
    r"|(?:\d{6}\s*,?\s*rv\.?\s*\d+)",
    re.IGNORECASE,
)

TOC_PATTERN = re.compile(r"\.{3,}|\.\s+\.\s+\.|_{5,}|—{3,}")

BAD_STARTS = [", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO", "Pag.", "pag."]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_hash(text: str) -> str:
    """Compute content hash."""
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode()).hexdigest()[:40]


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_num": i + 1,
            "text": text,
            "char_count": len(text),
        })

    doc.close()
    return pages


def detect_toc_pages(pages: list[dict], max_pages: int = 15) -> set[int]:
    """Detect TOC pages."""
    toc_pages = set()

    for page in pages:
        if page["page_num"] > max_pages:
            break

        text = page["text"]
        toc_matches = len(TOC_PATTERN.findall(text))

        if toc_matches >= 5:
            toc_pages.add(page["page_num"])

        if re.search(r"(?:INDICE|SOMMARIO|CONTENTS)", text, re.IGNORECASE):
            toc_pages.add(page["page_num"])

    return toc_pages


def segment_into_massime(
    pages: list[dict],
    config: dict,
    skip_pages: set[int] | None = None,
) -> list[dict]:
    """Segment pages into massime."""
    skip_pages = skip_pages or set()
    min_length = config.get("min_length", 150)
    max_citation_ratio = config.get("max_citation_ratio", 0.03)

    massime = []
    current_text = []
    current_start_page = None

    for page in pages:
        page_num = page["page_num"]

        # Skip TOC pages if configured
        if page_num in skip_pages:
            continue

        text = page["text"].strip()
        if not text:
            continue

        # Check for massima start patterns
        if MASSIMA_START_PATTERN.search(text) and current_text:
            # Flush current massima
            combined = "\n".join(current_text)
            if len(combined) >= min_length:
                massime.append({
                    "text": combined,
                    "page_start": current_start_page,
                    "page_end": page_num - 1,
                })
            current_text = [text]
            current_start_page = page_num
        else:
            if not current_text:
                current_start_page = page_num
            current_text.append(text)

    # Flush last
    if current_text:
        combined = "\n".join(current_text)
        if len(combined) >= min_length:
            massime.append({
                "text": combined,
                "page_start": current_start_page,
                "page_end": pages[-1]["page_num"] if pages else 1,
            })

    # Apply gate policy to each massima
    filtered = []
    for m in massime:
        text = m["text"]

        # Check bad starts
        if any(text.strip().startswith(bs) for bs in BAD_STARTS):
            continue

        # Check citation ratio
        citations = CITATION_PATTERN.findall(text)
        citation_chars = sum(len(c) for c in citations)
        ratio = citation_chars / len(text) if text else 0

        if ratio > max_citation_ratio and len(text) < 500:
            continue  # Likely a citation list

        # Check TOC infiltration
        if TOC_PATTERN.search(text) and len(text) < 300:
            continue

        filtered.append(m)

    return filtered


async def process_document(
    conn: asyncpg.Connection,
    manifest_id: int,
    filename: str,
    profile: str,
    batch_id: int,
) -> dict:
    """Process a single document."""
    # Find PDF
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        pdf_path = PDF_DIR / "new" / filename
    if not pdf_path.exists():
        return {"status": "not_found", "massime": 0}

    # Get or create document
    doc_id = await conn.fetchval(
        "SELECT doc_id FROM kb.pdf_manifest WHERE id = $1",
        manifest_id,
    )

    if not doc_id:
        # Create new document entry
        doc_id = str(uuid4())

        # Parse filename for metadata
        anno = 2020  # Default
        volume = 1   # Default
        tipo = "misto"  # Default

        year_match = re.search(r"(20\d{2})", filename)
        if year_match:
            anno = int(year_match.group(1))

        # Volume detection - multiple patterns
        vol_match = re.search(r"Vol(?:ume)?[._\s]*([IVX]+|\d+)", filename, re.IGNORECASE)
        if vol_match:
            vol_str = vol_match.group(1).upper()
            roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
            if vol_str in roman_to_int:
                volume = roman_to_int[vol_str]
            elif vol_str.isdigit():
                volume = int(vol_str)

        # Also check for "Vol 1", "Vol. 2" etc
        vol_match2 = re.search(r"[Vv]ol\.?\s*(\d+)", filename)
        if vol_match2:
            volume = int(vol_match2.group(1))

        # Type detection
        if "penale" in filename.lower():
            tipo = "penale"
        elif "civile" in filename.lower():
            tipo = "civile"
        elif "rassegna" in filename.lower():
            tipo = "rassegna"
        elif "approfond" in filename.lower():
            tipo = "approfondimenti"

        # Get source_hash from manifest
        source_hash = await conn.fetchval(
            "SELECT sha256 FROM kb.pdf_manifest WHERE id = $1",
            manifest_id,
        )

        await conn.execute(
            """
            INSERT INTO kb.documents (id, source_path, source_hash, anno, volume, tipo, titolo)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO NOTHING
            """,
            doc_id, filename, source_hash, anno, volume, tipo, filename[:100],
        )

        await conn.execute(
            "UPDATE kb.pdf_manifest SET doc_id = $1 WHERE id = $2",
            doc_id, manifest_id,
        )

    # Extract PDF
    pages = extract_pdf_text(pdf_path)

    # Get profile config
    config = PROFILE_GATE_CONFIG.get(profile, {"min_length": 150, "max_citation_ratio": 0.03})

    # Detect TOC pages if configured
    skip_pages = set()
    if config.get("skip_toc"):
        skip_pages = detect_toc_pages(pages)

    # Segment into massime
    massime = segment_into_massime(pages, config, skip_pages)

    # Save massime to database
    inserted = 0
    for i, m in enumerate(massime):
        text = m["text"]
        text_norm = normalize_text(text)
        content_hash = compute_hash(text)

        # Check for citations
        has_citation = bool(CITATION_PATTERN.search(text))

        # Check if already exists
        existing = await conn.fetchval(
            "SELECT id FROM kb.massime WHERE content_hash = $1",
            content_hash,
        )

        if existing:
            continue

        await conn.execute(
            """
            INSERT INTO kb.massime (
                id, document_id, testo, testo_normalizzato, content_hash,
                pagina_inizio, pagina_fine, citation_extracted,
                created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8,
                now()
            )
            """,
            str(uuid4()), doc_id, text[:50000], text_norm[:50000], content_hash,
            m["page_start"], m["page_end"], has_citation,
        )
        inserted += 1

    return {
        "status": "ok",
        "pages": len(pages),
        "massime_found": len(massime),
        "massime_inserted": inserted,
        "skipped_toc_pages": len(skip_pages),
    }


async def main(single_pdf: str | None = None, limit: int | None = None):
    """Main entry point."""
    print("=" * 70)
    print("GUIDED INGESTION - Local PyMuPDF Extraction")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Create or get batch
    batch_id = await conn.fetchval(
        """
        INSERT INTO kb.ingest_batches (batch_name, pipeline, status, started_at)
        VALUES ('guided_local_v1', 'pymupdf_local', 'running', now())
        ON CONFLICT (batch_name) DO UPDATE SET status = 'running', started_at = now()
        RETURNING id
        """
    )
    print(f"Batch ID: {batch_id}")

    # Get documents with profiles
    if single_pdf:
        query = """
            SELECT m.id, m.filename, r.profile
            FROM kb.pdf_manifest m
            LEFT JOIN kb.doc_intel_ab_results r ON r.manifest_id = m.id AND r.run_name = 'A'
            WHERE m.filename = $1
        """
        rows = await conn.fetch(query, single_pdf)
    else:
        query = """
            SELECT m.id, m.filename, r.profile
            FROM kb.pdf_manifest m
            LEFT JOIN kb.doc_intel_ab_results r ON r.manifest_id = m.id AND r.run_name = 'A'
            ORDER BY m.filename
        """
        rows = await conn.fetch(query)
        if limit:
            rows = rows[:limit]

    print(f"Documenti da processare: {len(rows)}")
    print()

    total_massime = 0
    processed = 0
    errors = 0

    for i, row in enumerate(rows):
        manifest_id = row["id"]
        filename = row["filename"]
        profile = row["profile"] or "structured_parent_child"

        print(f"[{i+1}/{len(rows)}] {filename[:45]:45} profile={profile[:20]}", end=" ")

        try:
            result = await process_document(
                conn, manifest_id, filename, profile, batch_id
            )

            if result["status"] == "ok":
                print(f"OK: {result['massime_inserted']} massime")
                total_massime += result["massime_inserted"]
                processed += 1
            else:
                print(f"SKIP: {result['status']}")

        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")
            errors += 1

    # Update batch
    await conn.execute(
        """
        UPDATE kb.ingest_batches
        SET status = 'completed', completed_at = now()
        WHERE id = $1
        """,
        batch_id,
    )

    # Final stats
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total massime created: {total_massime}")

    # Get total massime in DB
    total_db = await conn.fetchval("SELECT count(*) FROM kb.massime")
    print(f"Total massime in DB: {total_db}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided Ingestion (Local)")
    parser.add_argument("--pdf", type=str, help="Single PDF to process")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs")
    args = parser.parse_args()

    asyncio.run(main(single_pdf=args.pdf, limit=args.limit))
