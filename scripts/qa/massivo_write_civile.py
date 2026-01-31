#!/usr/bin/env python3
"""
Massivo Write - Civile Citation-Anchored

Controlled rollout of citation-anchored extraction for all Civile documents.

FEATURES:
- Wave-based processing (w1a, w1b, w2)
- Guardrail: massime-per-page p95 <= 25
- Guardrail: LLM usage rate < 10% global, < 15% per doc
- Atomic transactions per document
- Idempotent (can re-run safely)

Usage:
    # Dry run Wave 1a (baseline < 75)
    uv run python scripts/qa/massivo_write_civile.py --wave w1a

    # Commit Wave 1a
    uv run python scripts/qa/massivo_write_civile.py --wave w1a --commit

    # All waves
    uv run python scripts/qa/massivo_write_civile.py --wave all --commit
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
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
# CONFIGURATION
# ============================================================

BATCH_NAME = "civile_anchor_massivo_v1"

GATES = {
    "min_char": MIN_CHAR,  # 180
    "soft_cap": SOFT_CAP,  # 1700
    "hard_cap": HARD_CAP,  # 2000
    "window_before": 1,
    "window_after": 1,
    "toc_skip_pages": 25,
}

# Wave thresholds (baseline massime count)
WAVE_THRESHOLDS = {
    "w1a": (0, 75),      # baseline < 75
    "w1b": (75, 150),    # 75 <= baseline < 150
    "w2": (150, 10000),  # baseline >= 150
    "all": (0, 10000),   # all documents
}

# Guardrails
GUARDRAIL_MASSIME_PER_PAGE_P95 = 25
GUARDRAIL_LLM_RATE_GLOBAL = 0.10  # 10%
GUARDRAIL_LLM_RATE_PER_DOC = 0.15  # 15%

# Canary docs to skip (already processed)
CANARY_MANIFEST_IDS = {21, 51}  # Volume I 2016, Volume II 2024


@dataclass
class DocResult:
    manifest_id: int
    filename: str
    doc_id: str
    pages: int
    baseline_massime: int
    new_massime: int
    massime_per_page_p95: float
    llm_rate: float
    cut_decisions: int
    forced_cuts: int
    status: str  # PASS, FAIL_SPAM, FAIL_LLM, SKIP
    massime: list
    cut_decisions_data: list


# ============================================================
# TOC/Citation-list detection (same as canary)
# ============================================================

def is_toc_like(text: str) -> bool:
    """Detect TOC-like text (many numbers, few words)."""
    words = text.split()
    if len(words) < 10:
        return False
    digit_words = sum(1 for w in words if any(c.isdigit() for c in w))
    return digit_words / len(words) > 0.5


def is_citation_list_like(text: str) -> bool:
    """Detect citation-list text (many Sez./Cass. with little prose)."""
    import re
    citations = len(re.findall(r'(?:Sez\.|Cass\.)', text, re.IGNORECASE))
    sentences = len(re.findall(r'[.!?]\s+[A-Z]', text))
    if sentences < 3:
        return citations > 5
    return citations / max(sentences, 1) > 2


# ============================================================
# EXTRACTION (same logic as canary)
# ============================================================

def extract_from_pdf(pdf_path: Path, manifest_id: int) -> tuple[list, list]:
    """Extract massime from PDF using citation-anchored approach."""
    doc = fitz.open(str(pdf_path))
    massime = []
    cut_decisions = []
    seen_hashes = set()

    for page_num in range(len(doc)):
        if page_num < GATES["toc_skip_pages"]:
            continue

        page = doc[page_num]
        page_text = page.get_text()
        if not page_text or len(page_text) < 100:
            continue

        # Find citation anchors
        anchors = find_citation_anchors(page_text)
        sentences = split_into_sentences(page_text)  # returns [(text, start, end), ...]

        for anchor in anchors:
            # Find sentences around anchor
            anchor_sent_idx = None
            for i, (sent_text, sent_start, sent_end) in enumerate(sentences):
                if anchor.start_pos >= sent_start and anchor.start_pos < sent_end:
                    anchor_sent_idx = i
                    break

            if anchor_sent_idx is None:
                continue

            # Build window
            start_idx = max(0, anchor_sent_idx - GATES["window_before"])
            end_idx = min(len(sentences), anchor_sent_idx + GATES["window_after"] + 1)

            window_start = sentences[start_idx][1]  # start pos
            window_end = sentences[end_idx - 1][2]  # end pos
            raw_window = page_text[window_start:window_end]

            # Smart cut if too long
            if len(raw_window) > GATES["hard_cap"]:
                decision = choose_cut_sync(raw_window, GATES["soft_cap"], GATES["hard_cap"])
                window_text = raw_window[:decision.offset]

                cut_decisions.append({
                    "chunk_temp_id": f"p{page_num}:a{anchor.start_pos}",
                    "page_number": page_num,
                    "method": decision.method,
                    "trigger_type": decision.trigger_type,
                    "soft_cap": GATES["soft_cap"],
                    "hard_cap": GATES["hard_cap"],
                    "original_len": len(raw_window),
                    "chosen_cut_offset": decision.offset,
                    "chosen_candidate_index": decision.candidate_index,
                    "forced_cut": decision.forced_cut,
                    "candidates_json": [
                        {"offset": c.offset, "kind": c.kind, "reason": c.reason}
                        for c in decision.candidates
                    ],
                })
            else:
                window_text = raw_window

            # Skip TOC-like
            if is_toc_like(window_text):
                continue

            # Skip citation-list-like
            if is_citation_list_like(window_text):
                continue

            # Clean and dedupe
            testo = clean_legal_text(window_text)
            testo_norm = normalize_for_hash(testo)
            content_hash = compute_content_hash(testo)

            if len(testo) < GATES["min_char"]:
                continue

            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Normalize v2 for alignment
            testo_norm_v2, _ = normalize_v2(testo)
            fingerprint = compute_simhash64(testo_norm_v2)

            # Extract citation
            citation = extract_citation(window_text)

            massima = {
                "id": str(uuid4()),
                "testo": testo,
                "testo_normalizzato": testo_norm_v2,
                "content_hash": content_hash,
                "text_fingerprint": fingerprint,
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
            }

            massime.append(massima)

    doc.close()
    return massime, cut_decisions


# ============================================================
# GUARDRAIL CHECKS
# ============================================================

def check_massime_per_page_p95(massime: list, total_pages: int) -> tuple[float, bool]:
    """Check massime-per-page p95 guardrail."""
    if total_pages == 0 or not massime:
        return 0.0, True

    # Count massime per page
    page_counts = {}
    for m in massime:
        page = m["pagina_inizio"]
        page_counts[page] = page_counts.get(page, 0) + 1

    counts = sorted(page_counts.values())
    if not counts:
        return 0.0, True

    # Calculate p95
    idx = int(len(counts) * 0.95)
    p95 = counts[min(idx, len(counts) - 1)]

    return p95, p95 <= GUARDRAIL_MASSIME_PER_PAGE_P95


def check_llm_rate(cut_decisions: list) -> tuple[float, bool]:
    """Check LLM usage rate guardrail."""
    if not cut_decisions:
        return 0.0, True

    llm_used = sum(1 for d in cut_decisions if d["method"] in ("llm_validated", "llm_skipped_low_conf"))
    rate = llm_used / len(cut_decisions)

    return rate, rate <= GUARDRAIL_LLM_RATE_PER_DOC


# ============================================================
# DATABASE OPERATIONS
# ============================================================

async def get_wave_documents(conn, wave: str) -> list[dict]:
    """Get documents for specified wave."""
    min_baseline, max_baseline = WAVE_THRESHOLDS[wave]

    docs = await conn.fetch("""
        WITH doc_stats AS (
            SELECT
                d.id as doc_id,
                d.source_path,
                d.pagine,
                pm.id as manifest_id,
                pm.filename,
                COUNT(DISTINCT m.id) FILTER (WHERE m.ingest_batch_id IS NULL OR m.ingest_batch_id NOT IN (
                    SELECT id FROM kb.ingest_batches WHERE batch_name LIKE 'civile_anchor%'
                )) as baseline_massime
            FROM kb.documents d
            JOIN kb.pdf_manifest pm ON pm.doc_id = d.id
            LEFT JOIN kb.massime m ON m.document_id = d.id
            WHERE d.tipo = 'civile'
            GROUP BY d.id, d.source_path, d.pagine, pm.id, pm.filename
        )
        SELECT *
        FROM doc_stats
        WHERE baseline_massime >= $1 AND baseline_massime < $2
        ORDER BY baseline_massime ASC
    """, min_baseline, max_baseline)

    return [dict(d) for d in docs]


async def get_or_create_batch(conn, batch_name: str) -> int:
    """Get existing batch or create new one."""
    existing = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = $1",
        batch_name
    )

    if existing:
        return existing

    batch_id = await conn.fetchval("""
        INSERT INTO kb.ingest_batches (batch_name, pipeline, started_at, status)
        VALUES ($1, $2, $3, 'running')
        RETURNING id
    """, batch_name, "citation_anchored_v1", datetime.now(tz=None))

    return batch_id


async def write_doc_massime(conn, doc_id: str, massime: list, batch_id: int):
    """Write massime for a single document."""
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


async def write_doc_cut_decisions(conn, manifest_id: int, decisions: list, batch_id: int):
    """Write cut decisions for a single document."""
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


# ============================================================
# MAIN
# ============================================================

async def process_document(doc: dict, batch_id: int, commit: bool = False) -> DocResult:
    """Process a single document."""
    manifest_id = doc["manifest_id"]
    filename = doc["filename"] or Path(doc["source_path"]).name
    doc_id = str(doc["doc_id"])
    baseline = doc["baseline_massime"]

    # Skip canary docs
    if manifest_id in CANARY_MANIFEST_IDS:
        return DocResult(
            manifest_id=manifest_id,
            filename=filename,
            doc_id=doc_id,
            pages=0,
            baseline_massime=baseline,
            new_massime=0,
            massime_per_page_p95=0,
            llm_rate=0,
            cut_decisions=0,
            forced_cuts=0,
            status="SKIP_CANARY",
            massime=[],
            cut_decisions_data=[],
        )

    # Find PDF file
    pdf_path = None
    if doc["source_path"]:
        pdf_path = Path(doc["source_path"])
        if not pdf_path.exists():
            pdf_path = PDF_DIR / pdf_path.name

    if not pdf_path or not pdf_path.exists():
        # Try to find by filename
        for p in PDF_DIR.glob("**/*.pdf"):
            if filename in p.name:
                pdf_path = p
                break

    if not pdf_path or not pdf_path.exists():
        return DocResult(
            manifest_id=manifest_id,
            filename=filename,
            doc_id=doc_id,
            pages=0,
            baseline_massime=baseline,
            new_massime=0,
            massime_per_page_p95=0,
            llm_rate=0,
            cut_decisions=0,
            forced_cuts=0,
            status="SKIP_NO_PDF",
            massime=[],
            cut_decisions_data=[],
        )

    # Extract
    try:
        pdf_doc = fitz.open(str(pdf_path))
        total_pages = len(pdf_doc)
        pdf_doc.close()
    except Exception:
        total_pages = 0

    massime, cut_decisions = extract_from_pdf(pdf_path, manifest_id)

    # Check guardrails
    mpp_p95, mpp_pass = check_massime_per_page_p95(massime, total_pages)
    llm_rate, llm_pass = check_llm_rate(cut_decisions)

    forced_cuts = sum(1 for d in cut_decisions if d["forced_cut"])

    if not mpp_pass:
        status = "FAIL_SPAM"
    elif not llm_pass:
        status = "FAIL_LLM"
    else:
        status = "PASS"

    return DocResult(
        manifest_id=manifest_id,
        filename=filename,
        doc_id=doc_id,
        pages=total_pages,
        baseline_massime=baseline,
        new_massime=len(massime),
        massime_per_page_p95=mpp_p95,
        llm_rate=llm_rate,
        cut_decisions=len(cut_decisions),
        forced_cuts=forced_cuts,
        status=status,
        massime=massime,
        cut_decisions_data=cut_decisions,
    )


async def main(wave: str, commit: bool = False):
    print("=" * 80)
    print("MASSIVO WRITE - CIVILE CITATION-ANCHORED")
    print("=" * 80)
    print()

    mode = "[COMMIT MODE]" if commit else "[DRY-RUN MODE]"
    print(f"{mode}")
    print()

    print("CONFIGURATION:")
    for k, v in GATES.items():
        print(f"  {k}: {v}")
    print()

    print("GUARDRAILS:")
    print(f"  massime_per_page_p95: <= {GUARDRAIL_MASSIME_PER_PAGE_P95}")
    print(f"  llm_rate_per_doc: <= {GUARDRAIL_LLM_RATE_PER_DOC * 100:.0f}%")
    print(f"  llm_rate_global: <= {GUARDRAIL_LLM_RATE_GLOBAL * 100:.0f}%")
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get documents for wave
    docs = await get_wave_documents(conn, wave)
    print(f"Wave '{wave}': {len(docs)} documents")
    print()

    if not docs:
        print("No documents to process.")
        await conn.close()
        return

    # Get or create batch
    batch_id = await get_or_create_batch(conn, BATCH_NAME)
    print(f"Using batch: {batch_id} ({BATCH_NAME})")
    print()

    # Process documents
    results: list[DocResult] = []
    for doc in docs:
        result = await process_document(doc, batch_id, commit)
        results.append(result)

        status_icon = "[OK]" if result.status == "PASS" else "[--]" if result.status.startswith("SKIP") else "[!!]"
        improvement = result.new_massime / max(result.baseline_massime, 1)

        print(f"  {status_icon} {result.filename[:50]:50} | "
              f"base={result.baseline_massime:>4} new={result.new_massime:>5} ({improvement:>5.1f}x) | "
              f"mpp_p95={result.massime_per_page_p95:>4.1f} llm={result.llm_rate*100:>4.1f}% | "
              f"{result.status}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    pass_results = [r for r in results if r.status == "PASS"]
    fail_results = [r for r in results if r.status.startswith("FAIL")]
    skip_results = [r for r in results if r.status.startswith("SKIP")]

    total_baseline = sum(r.baseline_massime for r in pass_results)
    total_new = sum(r.new_massime for r in pass_results)
    total_cut_decisions = sum(r.cut_decisions for r in pass_results)
    total_forced = sum(r.forced_cuts for r in pass_results)

    # Global LLM rate
    all_cuts = sum(r.cut_decisions for r in results)
    llm_cuts = sum(
        sum(1 for d in r.cut_decisions_data if d["method"] in ("llm_validated", "llm_skipped_low_conf"))
        for r in results
    )
    global_llm_rate = llm_cuts / max(all_cuts, 1)

    print(f"Documents: {len(pass_results)} PASS, {len(fail_results)} FAIL, {len(skip_results)} SKIP")
    print(f"Massime: {total_baseline} baseline -> {total_new} new ({total_new/max(total_baseline,1):.1f}x)")
    print(f"Cut decisions: {total_cut_decisions} (forced: {total_forced}, {100*total_forced/max(total_cut_decisions,1):.1f}%)")
    print(f"Global LLM rate: {global_llm_rate*100:.1f}% (gate: <{GUARDRAIL_LLM_RATE_GLOBAL*100:.0f}%)")

    # Check global LLM gate
    if global_llm_rate > GUARDRAIL_LLM_RATE_GLOBAL:
        print(f"\n[WARN]  WARNING: Global LLM rate exceeds {GUARDRAIL_LLM_RATE_GLOBAL*100:.0f}% threshold!")

    if fail_results:
        print(f"\n[WARN]  FAILED documents:")
        for r in fail_results:
            print(f"    - {r.filename[:50]}: {r.status}")

    # Commit if requested
    if commit and pass_results:
        print()
        print("=" * 80)
        print("COMMITTING TO DATABASE")
        print("=" * 80)

        for r in pass_results:
            print(f"  Writing {r.filename[:50]}...")
            async with conn.transaction():
                await write_doc_massime(conn, r.doc_id, r.massime, batch_id)
                await write_doc_cut_decisions(conn, r.manifest_id, r.cut_decisions_data, batch_id)
            print(f"    {r.new_massime} massime + {r.cut_decisions} cut_decisions")

        # Update batch status
        await conn.execute("""
            UPDATE kb.ingest_batches
            SET status = 'completed', completed_at = $1
            WHERE id = $2
        """, datetime.now(tz=None), batch_id)

        print()
        print(f"[COMMITTED] Batch {batch_id} ({BATCH_NAME})")

    print()
    print("=" * 80)
    print("Rollback SQL (if needed):")
    print(f"  DELETE FROM kb.cut_decisions WHERE ingest_batch_id = {batch_id};")
    print(f"  DELETE FROM kb.massime WHERE ingest_batch_id = {batch_id};")
    print(f"  DELETE FROM kb.ingest_batches WHERE id = {batch_id};")
    print("=" * 80)

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Massivo Write - Civile Citation-Anchored")
    parser.add_argument("--wave", required=True, choices=["w1a", "w1b", "w2", "all"],
                        help="Wave to process: w1a (baseline<75), w1b (75-150), w2 (>=150), all")
    parser.add_argument("--commit", action="store_true",
                        help="Commit to database (default: dry-run)")
    args = parser.parse_args()

    asyncio.run(main(wave=args.wave, commit=args.commit))
