"""
QA Protocol - Phase 3: Gate Policy Audit

Re-processes all 63 PDFs through evaluate_gate() and logs EVERY decision
(accepted/rejected) with structured reason codes and numeric details.

Uses gate_policy.py module.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s3_gate_policy_audit.py
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import asyncpg
import httpx

# Add src to path for gate_policy import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lexe_api.kb.ingestion.gate_policy import GateConfig, evaluate_gate

from qa_config import DB_URL, PDF_DIR, UNSTRUCTURED_URL


async def extract_fast(client: httpx.AsyncClient, pdf_path: Path) -> list[dict]:
    """Extract with Unstructured fast strategy."""
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = await client.post(
            UNSTRUCTURED_URL,
            files=files,
            data={"strategy": "fast", "output_format": "application/json"},
            timeout=300.0,
        )
    if response.status_code != 200:
        return []
    return response.json()


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 3: GATE POLICY AUDIT")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] Using qa_run_id={qa_run_id}")

    # Get batch id
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )

    manifests = await conn.fetch(
        "SELECT id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    config = GateConfig()
    total_accepted = 0
    total_rejected = 0
    rejection_reasons = {}

    async with httpx.AsyncClient() as client:
        for m in manifests:
            manifest_id = m["id"]
            filename = m["filename"]

            # Check if already done
            existing = await conn.fetchval(
                "SELECT count(*) FROM kb.gate_decisions WHERE manifest_id = $1 AND qa_run_id = $2",
                manifest_id, qa_run_id,
            )
            if existing > 0:
                print(f"  [SKIP] {filename}: {existing} decisions exist")
                continue

            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                pdf_path = PDF_DIR / "new" / filename
            if not pdf_path.exists():
                continue

            elements = await extract_fast(client, pdf_path)
            if not elements:
                continue

            # Build merged texts (same as ingest_staging segmentation)
            current_texts = []
            merged_elements = []  # list of (text, elem_index, page, category)
            elem_idx = 0

            for elem in elements:
                text = elem.get("text", "").strip()
                elem_type = elem.get("type", "")
                page = elem.get("metadata", {}).get("page_number", 0)

                # Also evaluate raw elements that are skip types
                if elem_type in ("Header", "Footer", "PageNumber"):
                    merged_elements.append((text, elem_idx, page, elem_type))
                    elem_idx += 1
                    continue

                if re.match(r"^(Sez\.|SEZIONE|N\.\s*\d+)", text, re.IGNORECASE):
                    if current_texts:
                        full = " ".join(t for t, _, _, _ in current_texts)
                        first_page = current_texts[0][2]
                        first_cat = current_texts[0][3]
                        merged_elements.append((full, current_texts[0][1], first_page, first_cat))
                    current_texts = [(text, elem_idx, page, elem_type)]
                elif text:
                    current_texts.append((text, elem_idx, page, elem_type))

                elem_idx += 1

            if current_texts:
                full = " ".join(t for t, _, _, _ in current_texts)
                first_page = current_texts[0][2]
                first_cat = current_texts[0][3]
                merged_elements.append((full, current_texts[0][1], first_page, first_cat))

            # Evaluate each merged element
            doc_accepted = 0
            doc_rejected = 0

            for text, e_idx, page, category in merged_elements:
                result = evaluate_gate(text, element_type=category, page=page, config=config)

                await conn.execute(
                    """
                    INSERT INTO kb.gate_decisions
                      (qa_run_id, manifest_id, ingest_batch_id, element_index,
                       page_number, char_count, word_count,
                       decision, rejection_reason, rejection_details,
                       element_category, text_preview)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::kb.qa_decision,
                            $9, $10::jsonb, $11, $12)
                    """,
                    qa_run_id, manifest_id, batch_id,
                    e_idx, page,
                    len(text), len(text.split()),
                    result.decision,
                    result.reason,
                    json.dumps(result.details) if result.details else "{}",
                    category,
                    text[:200],
                )

                if result.accepted:
                    doc_accepted += 1
                    total_accepted += 1
                else:
                    doc_rejected += 1
                    total_rejected += 1
                    if result.reason:
                        rejection_reasons[result.reason] = rejection_reasons.get(result.reason, 0) + 1

            total = doc_accepted + doc_rejected
            rate = doc_accepted / total * 100 if total > 0 else 0
            print(f"  {filename}: {doc_accepted}/{total} accepted ({rate:.1f}%)")

    # Summary
    grand_total = total_accepted + total_rejected
    print(f"\n{'=' * 70}")
    print(f"GATE POLICY AUDIT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total decisions: {grand_total}")
    print(f"Accepted: {total_accepted} ({total_accepted/grand_total*100:.1f}%)" if grand_total > 0 else "Accepted: 0")
    print(f"Rejected: {total_rejected}")
    print(f"\nRejection reasons:")
    for reason, cnt in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
