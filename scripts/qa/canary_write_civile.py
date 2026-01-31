#!/usr/bin/env python3
"""
Canary Write - Civile Citation-Anchored

Scrive le massime estratte nel DB per i documenti canary.
Usa la configurazione validata: window 1+1, soft_cap 1700, hard_cap 2000.

IMPORTANTE:
- Crea nuovo ingest_batch_id (non sovrascrive)
- Logga tutte le cut_decisions per audit
- Non elimina le massime esistenti (confronto possibile)

Usage:
    uv run python scripts/qa/canary_write_civile.py --dry-run
    uv run python scripts/qa/canary_write_civile.py --commit
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lexe_api.kb.ingestion.massima_extractor import (
    extract_citation,
    find_citation_anchors,
    split_into_sentences,
)
from lexe_api.kb.ingestion.cut_validator import (
    SOFT_CAP,
    HARD_CAP,
    MIN_CHAR,
    choose_cut_sync,
)
from lexe_api.kb.ingestion.cleaner import clean_legal_text, compute_content_hash, normalize_for_hash
from lexe_api.kb.ingestion.normalization import compute_simhash64, normalize_v2
from qa_config import DB_URL, PDF_DIR


# ============================================================
# CONFIGURATION (validated by dry-run v2)
# ============================================================

GATES = {
    "min_char": MIN_CHAR,  # 180
    "soft_cap": SOFT_CAP,  # 1700
    "hard_cap": HARD_CAP,  # 2000
    "window_before": 1,
    "window_after": 1,
    "toc_skip_pages": 25,
}

BATCH_NAME = "civile_anchor_canary_v1"

# Canary documents
CANARY_DOCS = [
    "Volume I_2016_Massimario_Civile_1_372.pdf",
    "Volume II_2024_Massimario_Civile(volume completo)_.pdf",
]


def is_toc_like(text: str) -> bool:
    import re
    dotted_lines = len(re.findall(r"\.{3,}\s*\d+", text))
    if dotted_lines >= 3:
        return True
    section_headers = len(re.findall(r"^(?:Capitolo|Sezione|Parte|INDICE)", text, re.MULTILINE | re.IGNORECASE))
    return section_headers >= 2


def is_citation_list_like(text: str) -> bool:
    import re
    citations = len(re.findall(r"(?:Sez\.|Cass\.|Rv\.|n\.)\s*\d+", text, re.IGNORECASE))
    words = len(text.split())
    if words < 20:
        return False
    citation_ratio = (citations * 5) / words
    return citation_ratio > 0.4


def assess_anchor_quality_local(text: str) -> bool:
    """Check if anchor context has at least 2 quality signals."""
    import re
    signals = 0
    if re.search(r"Sez\.?\s*(?:U|L|[IVX0-9]+)", text, re.IGNORECASE):
        signals += 1
    if re.search(r"n\.?\s*\d+", text, re.IGNORECASE):
        signals += 1
    if re.search(r"Rv\.?\s*\d{5,6}", text, re.IGNORECASE):
        signals += 1
    if re.search(r"(?:19|20)\d{2}", text):
        signals += 1
    return signals >= 2


def extract_massime_from_page(text: str, page_num: int) -> tuple[list[dict], list[dict]]:
    """
    Estrai massime da una pagina con smart cut.

    Returns:
        (massime, cut_decisions)
    """
    massime = []
    cut_decisions = []

    anchors = find_citation_anchors(text)
    if not anchors:
        return [], []

    # Filtra ancore per qualità
    valid_anchors = []
    for anchor in anchors:
        start = max(0, anchor.start_pos - 200)
        end = min(len(text), anchor.end_pos + 200)
        context = text[start:end]

        if assess_anchor_quality_local(context):
            valid_anchors.append(anchor)

    if not valid_anchors:
        return [], []

    sentences = split_into_sentences(text)
    seen_hashes = set()

    for anchor_idx, anchor in enumerate(valid_anchors):
        # Trova la frase che contiene la citazione
        citation_sentence_idx = -1
        for i, (sent, start, end) in enumerate(sentences):
            if start <= anchor.start_pos < end:
                citation_sentence_idx = i
                break

        if citation_sentence_idx == -1:
            continue

        # Calcola range frasi
        start_idx = max(0, citation_sentence_idx - GATES["window_before"])
        end_idx = min(len(sentences), citation_sentence_idx + GATES["window_after"] + 1)

        # Estrai window grezzo
        window_start = sentences[start_idx][1]
        window_end = sentences[end_idx - 1][2]
        raw_window = text[window_start:window_end].strip()

        original_len = len(raw_window)

        # Check lunghezza minima
        if original_len < GATES["min_char"]:
            continue

        # Smart cut se troppo lungo
        if original_len > GATES["hard_cap"]:
            decision = choose_cut_sync(raw_window, GATES["soft_cap"], GATES["hard_cap"])
            window_text = raw_window[:decision.offset]

            # Log cut decision
            cut_decisions.append({
                "chunk_temp_id": f"{page_num}:{anchor_idx}",
                "page_number": page_num,
                "method": decision.method,
                "trigger_type": decision.trigger_type,
                "soft_cap": GATES["soft_cap"],
                "hard_cap": GATES["hard_cap"],
                "original_len": original_len,
                "chosen_cut_offset": decision.offset,
                "chosen_candidate_index": decision.candidate_index,
                "forced_cut": decision.forced_cut,
                "candidates_json": [
                    {"offset": c.offset, "kind": c.kind, "reason": c.reason, "score": c.score}
                    for c in decision.candidates
                ],
            })
        else:
            window_text = raw_window

        # Check TOC-like
        if is_toc_like(window_text):
            continue

        # Check citation-list-like
        if is_citation_list_like(window_text):
            continue

        # Dedupe per hash
        testo = clean_legal_text(window_text)
        testo_norm = normalize_for_hash(testo)
        content_hash = compute_content_hash(testo)

        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Normalizzazione v2 per alignment
        testo_norm_v2, _ = normalize_v2(testo)
        fingerprint = compute_simhash64(testo_norm_v2)

        # Estrai citazione
        citation = extract_citation(window_text)

        massima = {
            "id": str(uuid4()),
            "testo": testo,
            "testo_normalizzato": testo_norm_v2,
            "content_hash": content_hash,
            "text_fingerprint": fingerprint,
            "testo_con_contesto": window_text,
            "sezione": citation.normalized_sezione,
            "numero": citation.numero,
            "anno": citation.anno,
            "data_decisione": citation.data_decisione,
            "rv": citation.rv,
            "relatore": citation.relatore,
            "pagina_inizio": page_num,
            "pagina_fine": page_num,
            "citation_complete": citation.is_complete,
            "extraction_mode": "citation_anchored",
            "anchor_text": anchor.match_text[:100],
        }

        massime.append(massima)

    return massime, cut_decisions


async def get_document_info(conn, filename: str) -> dict:
    """Get document info from manifest."""
    row = await conn.fetchrow("""
        SELECT m.id, m.doc_id, m.filename, m.anno, m.pages,
               count(ma.id) as current_massime
        FROM kb.pdf_manifest m
        LEFT JOIN kb.massime ma ON ma.document_id = m.doc_id
        WHERE m.filename = $1
        GROUP BY m.id, m.doc_id, m.filename, m.anno, m.pages
    """, filename)
    return dict(row) if row else None


async def check_batch_exists(conn) -> int | None:
    """Check if batch already exists."""
    return await conn.fetchval("""
        SELECT id FROM kb.ingest_batches WHERE batch_name = $1
    """, BATCH_NAME)


async def create_or_reset_batch(conn) -> int:
    """Create batch or reset if exists (idempotent)."""
    existing = await check_batch_exists(conn)

    if existing:
        print(f"  [IDEMPOTENT] Batch {BATCH_NAME} exists (id={existing}), deleting old data...")
        # Delete old data from this batch
        await conn.execute("""
            DELETE FROM kb.cut_decisions WHERE ingest_batch_id = $1
        """, existing)
        await conn.execute("""
            DELETE FROM kb.massime WHERE ingest_batch_id = $1
        """, existing)
        # Update batch status
        await conn.execute("""
            UPDATE kb.ingest_batches
            SET status = 'running', started_at = $1, completed_at = NULL
            WHERE id = $2
        """, datetime.now(tz=None), existing)
        return existing

    # Create new batch
    batch_id = await conn.fetchval("""
        INSERT INTO kb.ingest_batches (batch_name, pipeline, started_at, status)
        VALUES ($1, $2, $3, 'running')
        RETURNING id
    """, BATCH_NAME, "citation_anchored_v1", datetime.now(tz=None))
    return batch_id


async def write_massime(conn, doc_id: str, massime: list[dict], batch_id: int):
    """Write massime to database."""
    for m in massime:
        await conn.execute("""
            INSERT INTO kb.massime (
                id, document_id, testo, testo_normalizzato, content_hash,
                text_fingerprint, sezione, numero, anno,
                data_decisione, rv, relatore,
                pagina_inizio, pagina_fine,
                citation_complete, extraction_mode, ingest_batch_id, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12,
                $13, $14,
                $15, $16, $17, $18
            )
        """,
            m["id"], doc_id, m["testo"], m["testo_normalizzato"], m["content_hash"],
            m["text_fingerprint"], m["sezione"], m["numero"], m["anno"],
            m["data_decisione"], m["rv"], m["relatore"],
            m["pagina_inizio"], m["pagina_fine"],
            m["citation_complete"], m["extraction_mode"], batch_id, datetime.now(tz=None),
        )


async def write_cut_decisions(conn, manifest_id: int, decisions: list[dict], batch_id: int):
    """Write cut decisions to audit table."""
    import json
    for d in decisions:
        await conn.execute("""
            INSERT INTO kb.cut_decisions (
                ingest_batch_id, manifest_id, chunk_temp_id, page_number,
                method, trigger_type, soft_cap, hard_cap,
                original_len, chosen_cut_offset, chosen_candidate_index, forced_cut,
                candidates_json, created_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10, $11, $12,
                $13, $14
            )
        """,
            batch_id, manifest_id, d["chunk_temp_id"], d["page_number"],
            d["method"], d["trigger_type"], d["soft_cap"], d["hard_cap"],
            d["original_len"], d["chosen_cut_offset"], d["chosen_candidate_index"], d["forced_cut"],
            json.dumps(d["candidates_json"]), datetime.now(tz=None),
        )


async def main(commit: bool = False):
    print("=" * 80)
    print("CANARY WRITE - CIVILE CITATION-ANCHORED")
    print("=" * 80)
    print()

    if not commit:
        print("[DRY-RUN MODE - No changes will be made]")
    else:
        print("[COMMIT MODE - Writing to database]")
    print()

    print("CONFIGURATION:")
    for k, v in GATES.items():
        print(f"  {k}: {v}")
    print(f"  batch_name: {BATCH_NAME}")
    print()

    conn = await asyncpg.connect(DB_URL)

    # Create batch if committing
    batch_id = None
    if commit:
        # Run migration first (outside transaction)
        migration_file = Path(__file__).parent / "migrations" / "003_cut_decisions.sql"
        if migration_file.exists():
            print("Running migration 003_cut_decisions.sql...")
            with open(migration_file) as f:
                await conn.execute(f.read())
            print("Migration complete.")
            print()

        # Idempotent batch creation
        batch_id = await create_or_reset_batch(conn)
        print(f"Using batch: {batch_id} ({BATCH_NAME})")
        print()

    results = []

    for filename in CANARY_DOCS:
        print(f"Processing: {filename}...")

        # Get document info
        doc_info = await get_document_info(conn, filename)
        if not doc_info:
            print(f"  [SKIP] Document not found in manifest")
            continue

        # Find PDF
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  [SKIP] PDF not found")
            continue

        # Extract from PDF
        doc = fitz.open(pdf_path)
        all_massime = []
        all_cut_decisions = []

        for i in range(len(doc)):
            page_num = i + 1

            if page_num <= GATES["toc_skip_pages"]:
                continue

            text = doc[i].get_text()
            if len(text.strip()) < 100:
                continue

            massime, cuts = extract_massime_from_page(text, page_num)
            all_massime.extend(massime)
            all_cut_decisions.extend(cuts)

        doc.close()

        # Dedupe globale
        seen_hashes = set()
        unique_massime = []
        for m in all_massime:
            if m["content_hash"] not in seen_hashes:
                seen_hashes.add(m["content_hash"])
                unique_massime.append(m)

        result = {
            "filename": filename,
            "manifest_id": doc_info["id"],
            "doc_id": doc_info["doc_id"],
            "current_massime": doc_info["current_massime"],
            "new_massime": len(unique_massime),
            "cut_decisions": len(all_cut_decisions),
            "cit_complete": sum(1 for m in unique_massime if m["citation_complete"]),
        }

        improvement = result["new_massime"] / result["current_massime"] if result["current_massime"] > 0 else float('inf')
        imp_str = f"{improvement:.1f}x" if improvement != float('inf') else "NEW"

        print(f"  Current: {result['current_massime']}, New: {result['new_massime']}, Improvement: {imp_str}")
        print(f"  Cut decisions logged: {result['cut_decisions']}")
        print(f"  With complete citation: {result['cit_complete']} ({result['cit_complete']/result['new_massime']*100:.1f}%)")

        # Store for later atomic write
        result["_massime"] = unique_massime
        result["_cut_decisions"] = all_cut_decisions

        print()
        results.append(result)

    # ATOMIC WRITE in single transaction
    if commit and results:
        print("=" * 80)
        print("ATOMIC WRITE (single transaction)")
        print("=" * 80)

        try:
            async with conn.transaction():
                for r in results:
                    print(f"  Writing {r['filename'][:40]}...")
                    await write_massime(conn, r["doc_id"], r["_massime"], batch_id)
                    await write_cut_decisions(conn, r["manifest_id"], r["_cut_decisions"], batch_id)
                    print(f"    {r['new_massime']} massime + {r['cut_decisions']} cut_decisions")

            print("  [TRANSACTION COMMITTED]")
        except Exception as e:
            print(f"  [TRANSACTION ROLLED BACK] Error: {e}")
            raise

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_current = sum(r["current_massime"] for r in results)
    total_new = sum(r["new_massime"] for r in results)
    total_cuts = sum(r["cut_decisions"] for r in results)

    print(f"Documents processed: {len(results)}")
    print(f"Total current massime: {total_current}")
    print(f"Total new massime: {total_new}")
    print(f"Total cut decisions: {total_cuts}")
    if total_current > 0:
        print(f"Overall improvement: {total_new / total_current:.1f}x")

    if commit:
        # Update batch status
        await conn.execute("""
            UPDATE kb.ingest_batches
            SET status = 'completed', completed_at = $1
            WHERE id = $2
        """, datetime.utcnow(), batch_id)

        print()
        print(f"[COMMITTED] Batch: {batch_id} ({BATCH_NAME})")
        print()
        print("Next steps:")
        print("  1. Run reference alignment: uv run python scripts/qa/s6_reference_alignment_v2.py")
        print("  2. Verify coverage: uv run python scripts/qa/verify_alignment_fix.py")
    else:
        print()
        print("[DRY-RUN] No changes made. Run with --commit to write to DB.")

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canary write Civile citation-anchored")
    parser.add_argument("--commit", action="store_true", help="Actually write to database")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing (default)")
    args = parser.parse_args()

    asyncio.run(main(commit=args.commit))
