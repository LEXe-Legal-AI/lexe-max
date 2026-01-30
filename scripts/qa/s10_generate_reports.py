"""
QA Protocol - Phase 10: Generate Reports

Per-document risk score + risk grade, and global summary.
Depends on all previous phases.

Risk score (0=healthy, 1=broken):
  risk = (1 - extraction_quality) * 0.25
       + gate_penalty * 0.20
       + (100 - coverage_pct)/100 * 0.30
       + (1 - recall_5) * 0.15
       + health_flags * 0.02 (cap 0.10)

Grades: A (0-0.2), B (0.2-0.4), C (0.4-0.6), D (0.6-0.8), F (0.8-1.0)

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    uv run python scripts/qa/s10_generate_reports.py
"""

import asyncio
import json

import asyncpg

from qa_config import DB_URL


def compute_risk_grade(score: float) -> str:
    """Map risk score to grade."""
    if score <= 0.2:
        return "A"
    if score <= 0.4:
        return "B"
    if score <= 0.6:
        return "C"
    if score <= 0.8:
        return "D"
    return "F"


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 10: GENERATE REPORTS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )
    print(f"[OK] qa_run_id={qa_run_id}, batch_id={batch_id}")

    manifests = await conn.fetch(
        "SELECT id, doc_id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    grade_dist = {}
    all_risks = []

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT 1 FROM kb.qa_document_reports WHERE manifest_id = $1 AND ingest_batch_id = $2",
            manifest_id, batch_id,
        )
        if existing:
            continue

        # ── Gather metrics ────────────────────────────────────────
        # Extraction quality
        eq = await conn.fetchrow(
            "SELECT overall_quality_score FROM kb.pdf_extraction_quality WHERE manifest_id = $1",
            manifest_id,
        )
        extraction_quality = float(eq["overall_quality_score"] or 0) if eq else 0.0

        # Gate acceptance rate
        gate = await conn.fetchrow(
            """
            SELECT count(*) as total,
                   count(*) FILTER (WHERE decision = 'accepted') as accepted
            FROM kb.gate_decisions
            WHERE manifest_id = $1 AND qa_run_id = $2
            """,
            manifest_id, qa_run_id,
        )
        gate_total = gate["total"] if gate else 0
        gate_accepted = gate["accepted"] if gate else 0
        gate_rate = gate_accepted / gate_total if gate_total > 0 else 1.0

        # Gate penalty: high rejection is bad, but not a direct inversion
        gate_penalty = max(0, 1.0 - gate_rate)

        # Reference coverage
        ref_summary = await conn.fetchrow(
            """
            SELECT coverage_pct
            FROM kb.reference_alignment_summary
            WHERE manifest_id = $1 AND ingest_batch_id = $2
            """,
            manifest_id, batch_id,
        )
        coverage_pct = float(ref_summary["coverage_pct"] or 0) if ref_summary else 0.0

        # Retrieval self-recall@5
        recall = await conn.fetchrow(
            """
            SELECT avg(recall_at_5) as avg_r5
            FROM kb.retrieval_eval_results rr
            JOIN kb.retrieval_eval_queries rq ON rq.id = rr.query_id
            JOIN kb.massime ms ON ms.id = rq.source_massima_id
            JOIN kb.pdf_manifest pm ON pm.doc_id = ms.document_id
            WHERE pm.id = $1 AND rr.method = 'R1_hybrid'
            """,
            manifest_id,
        )
        recall_5 = float(recall["avg_r5"] or 0) if recall else 0.0

        # Health flags count
        health_count = await conn.fetchval(
            "SELECT count(*) FROM kb.pdf_health_flags WHERE manifest_id = $1",
            manifest_id,
        )
        health_count = health_count or 0

        # Chunking quality: from chunk features
        chunk_quality = await conn.fetchrow(
            """
            SELECT avg(quality_score) as avg_q,
                   count(*) FILTER (WHERE toc_infiltration_score > 0.6) as toc_count,
                   count(*) as total
            FROM kb.chunk_features
            WHERE manifest_id = $1 AND qa_run_id = $2
            """,
            manifest_id, qa_run_id,
        )
        chunking_quality = float(chunk_quality["avg_q"] or 0) if chunk_quality else 0.0

        # Profile
        profile_row = await conn.fetchval(
            "SELECT profile FROM kb.qa_ingestion_profiles WHERE manifest_id = $1",
            manifest_id,
        )
        profile = profile_row or "unknown"

        # ── Compute Risk Score ────────────────────────────────────
        risk = (
            (1 - extraction_quality) * 0.25
            + gate_penalty * 0.20
            + (100 - coverage_pct) / 100 * 0.30
            + (1 - recall_5) * 0.15
            + min(health_count * 0.02, 0.10)
        )
        risk = min(max(risk, 0.0), 1.0)
        grade = compute_risk_grade(risk)
        grade_dist[grade] = grade_dist.get(grade, 0) + 1
        all_risks.append(risk)

        # Recommended actions
        actions = []
        if extraction_quality < 0.6:
            actions.append("re-extract with hi_res OCR")
        if gate_rate < 0.5:
            actions.append("review gate policy thresholds")
        if coverage_pct < 85:
            actions.append("investigate low reference coverage")
        if health_count > 3:
            actions.append("review health flags")
        if profile == "ocr_needed":
            actions.append("OCR re-processing required")
        if profile == "toc_heavy":
            actions.append("TOC removal pre-processing")

        report_json = {
            "extraction_quality": extraction_quality,
            "gate_acceptance_rate": gate_rate,
            "gate_total": gate_total,
            "coverage_pct": coverage_pct,
            "recall_5": recall_5,
            "health_flags": health_count,
            "chunking_quality": chunking_quality,
            "profile": profile,
        }

        await conn.execute(
            """
            INSERT INTO kb.qa_document_reports
              (qa_run_id, manifest_id, ingest_batch_id,
               extraction_quality_score, gate_acceptance_rate,
               chunking_quality_score, reference_coverage_pct,
               retrieval_self_recall_5, composite_risk_score, risk_grade,
               profile, health_flag_count, recommended_actions, report_json)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
            ON CONFLICT (manifest_id, ingest_batch_id) DO NOTHING
            """,
            qa_run_id, manifest_id, batch_id,
            round(extraction_quality, 4), round(gate_rate, 4),
            round(chunking_quality, 4), round(coverage_pct, 2),
            round(recall_5, 4), round(risk, 4), grade,
            profile, health_count, actions,
            json.dumps(report_json),
        )

        print(f"  [{grade}] {filename}: risk={risk:.3f} profile={profile}")

    # ── Global Report ─────────────────────────────────────────────
    total_docs = len(manifests)
    total_massime = await conn.fetchval("SELECT count(*) FROM kb.massime")

    avg_eq = await conn.fetchval(
        "SELECT avg(extraction_quality_score) FROM kb.qa_document_reports WHERE ingest_batch_id = $1",
        batch_id,
    )
    avg_gate = await conn.fetchval(
        "SELECT avg(gate_acceptance_rate) FROM kb.qa_document_reports WHERE ingest_batch_id = $1",
        batch_id,
    )
    avg_cov = await conn.fetchval(
        "SELECT avg(reference_coverage_pct) FROM kb.qa_document_reports WHERE ingest_batch_id = $1",
        batch_id,
    )
    avg_r5 = await conn.fetchval(
        "SELECT avg(retrieval_self_recall_5) FROM kb.qa_document_reports WHERE ingest_batch_id = $1",
        batch_id,
    )

    # Top risk docs
    top_risk = await conn.fetch(
        """
        SELECT pm.doc_id
        FROM kb.qa_document_reports qdr
        JOIN kb.pdf_manifest pm ON pm.id = qdr.manifest_id
        WHERE qdr.ingest_batch_id = $1
        ORDER BY qdr.composite_risk_score DESC
        LIMIT 10
        """,
        batch_id,
    )
    top_risk_ids = [r["doc_id"] for r in top_risk if r["doc_id"] is not None]

    # Profile distribution
    profiles = await conn.fetch(
        """
        SELECT profile, count(*) as cnt
        FROM kb.qa_ingestion_profiles
        WHERE qa_run_id = $1
        GROUP BY profile
        """,
        qa_run_id,
    )
    profile_dist = {r["profile"]: r["cnt"] for r in profiles}

    await conn.execute(
        """
        INSERT INTO kb.qa_global_reports
          (qa_run_id, ingest_batch_id, total_documents, total_massime,
           avg_extraction_quality, avg_gate_acceptance_rate,
           avg_reference_coverage, avg_retrieval_recall_5,
           grade_distribution, profile_distribution,
           top_risk_documents)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::uuid[])
        ON CONFLICT (ingest_batch_id) DO NOTHING
        """,
        qa_run_id, batch_id, total_docs, total_massime,
        float(avg_eq or 0), float(avg_gate or 0),
        float(avg_cov or 0), float(avg_r5 or 0),
        json.dumps(grade_dist), json.dumps(profile_dist),
        top_risk_ids,
    )

    # Summary
    print(f"\n{'=' * 70}")
    print(f"REPORTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Documents: {total_docs}")
    print(f"Massime: {total_massime}")
    print(f"\nGrade distribution:")
    for g in ("A", "B", "C", "D", "F"):
        print(f"  {g}: {grade_dist.get(g, 0)}")
    print(f"\nAverages:")
    print(f"  Extraction quality: {float(avg_eq or 0):.3f}")
    print(f"  Gate acceptance: {float(avg_gate or 0):.3f}")
    print(f"  Reference coverage: {float(avg_cov or 0):.1f}%")
    print(f"  Retrieval R@5: {float(avg_r5 or 0):.3f}")

    if all_risks:
        avg_risk = sum(all_risks) / len(all_risks)
        print(f"  Avg risk score: {avg_risk:.3f}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
