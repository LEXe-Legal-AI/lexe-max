"""
QA Protocol - Phase 6: Reference Alignment

Aligns reference units (Phase 0, hi_res) with pipeline massime.
Match types (ENUM): exact, partial, split, merged, unmatched.
Computes coverage_pct, fragmentation_score, fusion_score per document.

Uses jaccard_similarity from deduplicator.py.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s6_reference_alignment.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lexe_api.kb.ingestion.deduplicator import jaccard_similarity

from qa_config import DB_URL

JACCARD_EXACT = 0.9
JACCARD_PARTIAL = 0.5
JACCARD_SPLIT_MIN = 0.3


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 6: REFERENCE ALIGNMENT")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'guided_local_v1'"
    )
    print(f"[OK] qa_run_id={qa_run_id}, batch_id={batch_id}")

    manifests = await conn.fetch(
        "SELECT id, doc_id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    total_matched = 0
    total_unmatched = 0

    for m in manifests:
        manifest_id = m["id"]
        doc_id = m["doc_id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT 1 FROM kb.reference_alignment_summary WHERE manifest_id = $1 AND ingest_batch_id = $2",
            manifest_id, batch_id,
        )
        if existing:
            continue

        # Get reference units
        ref_units = await conn.fetch(
            """
            SELECT id, testo, testo_norm, content_hash
            FROM kb.qa_reference_units
            WHERE manifest_id = $1
            ORDER BY unit_index
            """,
            manifest_id,
        )

        # Get pipeline massime
        massime = await conn.fetch(
            """
            SELECT id, testo, testo_normalizzato, content_hash
            FROM kb.massime
            WHERE document_id = $1
            """,
            doc_id,
        )

        if not ref_units:
            print(f"  [SKIP] {filename}: no reference units")
            continue

        # Build match index for massime
        massima_by_hash = {r["content_hash"]: r for r in massime}

        doc_matched = 0
        doc_unmatched = 0
        doc_fragmented = 0
        doc_fused = 0
        overlaps = []

        for ref in ref_units:
            ref_id = ref["id"]
            ref_text = ref["testo_norm"] or ref["testo"].lower()
            ref_hash = ref["content_hash"]

            best_match = None
            best_type = "unmatched"
            best_jaccard = 0.0
            best_massima_id = None
            fragment_count = 0
            fusion_count = 0

            # 1. Exact match via content_hash
            if ref_hash in massima_by_hash:
                best_type = "exact"
                best_jaccard = 1.0
                best_massima_id = massima_by_hash[ref_hash]["id"]
            else:
                # 2. Jaccard similarity search
                best_sim = 0.0
                matching_massime = []

                for ms in massime:
                    ms_text = ms["testo_normalizzato"] or ms["testo"].lower()
                    sim = jaccard_similarity(ref_text, ms_text)

                    if sim > best_sim:
                        best_sim = sim
                        best_match = ms

                    if sim >= JACCARD_SPLIT_MIN:
                        matching_massime.append((ms, sim))

                if best_sim >= JACCARD_EXACT:
                    best_type = "exact"
                    best_jaccard = best_sim
                    best_massima_id = best_match["id"]
                elif best_sim >= JACCARD_PARTIAL:
                    best_type = "partial"
                    best_jaccard = best_sim
                    best_massima_id = best_match["id"]
                elif len(matching_massime) >= 2:
                    # Split: reference fragmented across multiple massime
                    best_type = "split"
                    best_jaccard = best_sim
                    best_massima_id = best_match["id"] if best_match else None
                    fragment_count = len(matching_massime)
                    doc_fragmented += 1
                else:
                    best_type = "unmatched"
                    best_jaccard = best_sim

            # Check fusion: multiple refs map to same massima
            # (handled in summary pass below)

            await conn.execute(
                """
                INSERT INTO kb.reference_alignment
                  (qa_run_id, manifest_id, ingest_batch_id,
                   ref_unit_id, matched_massima_id,
                   match_type, overlap_ratio, jaccard_similarity,
                   fragment_count, fusion_count)
                VALUES ($1, $2, $3, $4, $5, $6::kb.qa_match_type, $7, $8, $9, $10)
                """,
                qa_run_id, manifest_id, batch_id,
                ref_id, best_massima_id,
                best_type, best_jaccard, best_jaccard,
                fragment_count, fusion_count,
            )

            if best_type != "unmatched":
                doc_matched += 1
                overlaps.append(best_jaccard)
            else:
                doc_unmatched += 1

        # Detect fusions: multiple ref_units → same massima
        aligned = await conn.fetch(
            """
            SELECT matched_massima_id, count(*) as cnt
            FROM kb.reference_alignment
            WHERE manifest_id = $1 AND ingest_batch_id = $2
              AND matched_massima_id IS NOT NULL
            GROUP BY matched_massima_id
            HAVING count(*) > 1
            """,
            manifest_id, batch_id,
        )
        fusion_count_total = sum(r["cnt"] for r in aligned)

        # Update fusion counts
        for r in aligned:
            await conn.execute(
                """
                UPDATE kb.reference_alignment
                SET fusion_count = $1
                WHERE manifest_id = $2 AND ingest_batch_id = $3
                  AND matched_massima_id = $4
                """,
                r["cnt"], manifest_id, batch_id, r["matched_massima_id"],
            )

        # Summary
        total_ref = doc_matched + doc_unmatched
        coverage_pct = (doc_matched / total_ref * 100) if total_ref > 0 else 0
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

        # Fragmentation score: avg fragments per matched ref (1.0 = no fragmentation)
        frag_rows = await conn.fetch(
            """
            SELECT fragment_count FROM kb.reference_alignment
            WHERE manifest_id = $1 AND ingest_batch_id = $2 AND match_type = 'split'
            """,
            manifest_id, batch_id,
        )
        frag_score = (
            sum(r["fragment_count"] for r in frag_rows) / len(frag_rows)
            if frag_rows else 1.0
        )

        fusion_score = fusion_count_total / total_ref if total_ref > 0 else 0

        await conn.execute(
            """
            INSERT INTO kb.reference_alignment_summary
              (qa_run_id, manifest_id, ingest_batch_id,
               total_ref_units, matched_count, unmatched_count,
               coverage_pct, fragmentation_score, fusion_score, avg_overlap)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (manifest_id, ingest_batch_id) DO NOTHING
            """,
            qa_run_id, manifest_id, batch_id,
            total_ref, doc_matched, doc_unmatched,
            round(coverage_pct, 2), round(frag_score, 2),
            round(fusion_score, 4), round(avg_overlap, 4),
        )

        total_matched += doc_matched
        total_unmatched += doc_unmatched

        status = "OK" if coverage_pct >= 85 else "WARN" if coverage_pct >= 70 else "LOW"
        print(f"  [{status}] {filename}: coverage={coverage_pct:.1f}%, frag={frag_score:.2f}")

    # Global summary
    summary = await conn.fetchrow(
        """
        SELECT
            avg(coverage_pct) as avg_coverage,
            avg(fragmentation_score) as avg_frag,
            avg(fusion_score) as avg_fusion
        FROM kb.reference_alignment_summary
        WHERE ingest_batch_id = $1
        """,
        batch_id,
    )

    print(f"\n{'=' * 70}")
    print(f"REFERENCE ALIGNMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total matched: {total_matched}")
    print(f"Total unmatched: {total_unmatched}")
    if summary:
        print(f"Avg coverage: {float(summary['avg_coverage'] or 0):.1f}%")
        print(f"Avg fragmentation: {float(summary['avg_frag'] or 0):.2f}")
        print(f"Avg fusion: {float(summary['avg_fusion'] or 0):.4f}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
