"""
QA Protocol - Phase 11: Guided Ingestion

Re-ingests PDFs using profile-specific configurations:
- clean_standard: pipeline standard (fast, min_length=150)
- legacy_layout_2010_2013: aggressive cleaning, min_length=120
- toc_heavy: TOC removal pre-chunking, skip TOC pages
- citation_dense: block segmentation, citation_ratio=5%
- ocr_needed: hi_res extraction, OCR

After guided ingestion, re-run Phases 4-10 with new ingest_batch_id.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/guided_ingestion.py
"""

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lexe_api.kb.ingestion.gate_policy import GateConfig, evaluate_gate

from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL

# Profile-specific configs
PROFILE_CONFIGS = {
    "clean_standard": GateConfig(min_length=150, max_citation_ratio=0.03),
    "legacy_layout_2010_2013": GateConfig(min_length=120, max_citation_ratio=0.03),
    "toc_heavy": GateConfig(min_length=150, max_citation_ratio=0.03),
    "citation_dense": GateConfig(min_length=150, max_citation_ratio=0.05),
    "ocr_needed": GateConfig(min_length=150, max_citation_ratio=0.03),
}

PROFILE_STRATEGIES = {
    "clean_standard": "fast",
    "legacy_layout_2010_2013": "fast",
    "toc_heavy": "fast",
    "citation_dense": "fast",
    "ocr_needed": "hi_res",
}


async def extract_pdf(
    client: httpx.AsyncClient, pdf_path: Path, strategy: str
) -> list[dict]:
    """Extract PDF with given strategy."""
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = await client.post(
            UNSTRUCTURED_URL,
            files=files,
            data={"strategy": strategy, "output_format": "application/json"},
            timeout=600.0,
        )
    if response.status_code != 200:
        return []
    return response.json()


def detect_toc_pages(elements: list[dict], max_pages: int = 10) -> set[int]:
    """Detect TOC pages from elements."""
    toc_pages = set()
    for elem in elements:
        page = elem.get("metadata", {}).get("page_number")
        if not page or page > max_pages:
            continue
        text = elem.get("text", "")
        # TOC heuristics
        dotted = len(re.findall(r"\.{3,}", text))
        if dotted >= 3:
            toc_pages.add(page)
        if re.search(r"(?:INDICE|SOMMARIO)", text, re.IGNORECASE):
            toc_pages.add(page)
    return toc_pages


def segment_elements(
    elements: list[dict],
    config: GateConfig,
    skip_pages: set[int] | None = None,
) -> list[dict]:
    """Segment elements into massime with gate policy."""
    massime = []
    current = []
    skip_pages = skip_pages or set()

    for elem in elements:
        text = elem.get("text", "").strip()
        elem_type = elem.get("type", "")
        page = elem.get("metadata", {}).get("page_number", 0)

        if page in skip_pages:
            continue

        result = evaluate_gate(text, element_type=elem_type, page=page, config=config)

        if elem_type in ("Header", "Footer", "PageNumber"):
            continue

        if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+)", text, re.IGNORECASE):
            if current:
                full = " ".join(current)
                gate = evaluate_gate(full, config=config)
                if gate.accepted:
                    massime.append({"testo": full, "page": page})
            current = [text]
        elif text:
            current.append(text)

    if current:
        full = " ".join(current)
        gate = evaluate_gate(full, config=config)
        if gate.accepted:
            massime.append({"testo": full})

    return massime


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 11: GUIDED INGESTION")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )

    # Create guided batch
    guided_batch_id = await conn.fetchval(
        """
        INSERT INTO kb.ingest_batches (batch_name, pipeline, config_json)
        VALUES ($1, $2, $3::jsonb)
        ON CONFLICT (batch_name) DO UPDATE SET started_at = now()
        RETURNING id
        """,
        "guided_v1",
        "guided_ingestion",
        json.dumps({
            "profiles": list(PROFILE_CONFIGS.keys()),
            "strategies": PROFILE_STRATEGIES,
        }),
    )
    print(f"[OK] Guided batch id={guided_batch_id}")

    # Get profiles
    profiles = await conn.fetch(
        """
        SELECT ip.manifest_id, ip.profile,
               pm.filename, pm.doc_id
        FROM kb.qa_ingestion_profiles ip
        JOIN kb.pdf_manifest pm ON pm.id = ip.manifest_id
        WHERE ip.qa_run_id = $1
        ORDER BY ip.profile
        """,
        qa_run_id,
    )
    print(f"[OK] Found {len(profiles)} PDFs with profiles")

    profile_stats = {}
    total_new = 0

    async with httpx.AsyncClient() as client:
        for p in profiles:
            manifest_id = p["manifest_id"]
            profile = p["profile"]
            filename = p["filename"]
            doc_id = p["doc_id"]

            config = PROFILE_CONFIGS.get(profile, GateConfig())
            strategy = PROFILE_STRATEGIES.get(profile, "fast")

            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                pdf_path = PDF_DIR / "new" / filename
            if not pdf_path.exists():
                continue

            print(f"\n  [{profile}] {filename} (strategy={strategy})")

            # Extract
            try:
                elements = await extract_pdf(client, pdf_path, strategy)
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

            if not elements:
                continue

            # Detect TOC pages for toc_heavy profile
            skip_pages = set()
            if profile == "toc_heavy":
                skip_pages = detect_toc_pages(elements)
                config = GateConfig(
                    **{**config.__dict__, "skip_pages": skip_pages}
                )
                if skip_pages:
                    print(f"  Skipping TOC pages: {sorted(skip_pages)}")

            # Segment
            massime = segment_elements(elements, config, skip_pages)
            print(f"  Segmented: {len(massime)} massime")

            # Insert (skip duplicates)
            count = 0
            for ms in massime:
                testo = ms["testo"]
                testo_norm = re.sub(r"\s+", " ", testo.lower().strip())
                content_hash = hashlib.sha256(testo_norm.encode()).hexdigest()

                exists = await conn.fetchval(
                    "SELECT 1 FROM kb.massime WHERE content_hash = $1",
                    content_hash,
                )
                if exists:
                    continue

                # Get anno/tipo from document
                doc_info = await conn.fetchrow(
                    "SELECT anno, tipo FROM kb.documents WHERE id = $1",
                    doc_id,
                )
                anno = doc_info["anno"] if doc_info else None
                tipo = doc_info["tipo"] if doc_info else None

                await conn.execute(
                    """
                    INSERT INTO kb.massime (id, document_id, testo, testo_normalizzato, content_hash, anno, tipo)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    uuid4(), doc_id, testo, testo_norm, content_hash, anno, tipo,
                )
                count += 1

            total_new += count
            profile_stats[profile] = profile_stats.get(profile, 0) + count
            print(f"  [OK] {count} new massime inserted")

    # Update batch status
    await conn.execute(
        "UPDATE kb.ingest_batches SET status = 'completed', completed_at = now() WHERE id = $1",
        guided_batch_id,
    )

    # Summary
    total_massime = await conn.fetchval("SELECT count(*) FROM kb.massime")

    print(f"\n{'=' * 70}")
    print(f"GUIDED INGESTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"New massime: {total_new}")
    print(f"Total massime: {total_massime}")
    print(f"\nBy profile:")
    for p, cnt in sorted(profile_stats.items()):
        print(f"  {p}: {cnt} new")
    print(f"\n[!] Re-run Phases 4-10 with ingest_batch_id={guided_batch_id}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
