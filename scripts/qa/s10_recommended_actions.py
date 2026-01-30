"""
QA Protocol - Phase 10: Recommended Actions (Smoke Tests)

Runs smoke tests on QA results and prints actionable recommendations:
- Top 10 docs by reject_rate
- Top 10 docs by toc_score
- Coverage thresholds (guardrail 85%, target 90%, excellent 95%)
- Retrieval comparison by method

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    uv run python scripts/qa/s10_recommended_actions.py
"""

import asyncio

import asyncpg

from qa_config import DB_URL


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 10: RECOMMENDED ACTIONS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)

    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )

    # ── 1. Top 10 by reject rate ─────────────────────────────────
    print("\n### TOP 10 PDFs BY REJECTION RATE")
    print("-" * 60)
    top_reject = await conn.fetch(
        """
        SELECT pm.filename,
               qdr.gate_acceptance_rate,
               qdr.composite_risk_score,
               qdr.risk_grade
        FROM kb.qa_document_reports qdr
        JOIN kb.pdf_manifest pm ON pm.id = qdr.manifest_id
        WHERE qdr.ingest_batch_id = $1
        ORDER BY qdr.gate_acceptance_rate ASC
        LIMIT 10
        """,
        batch_id,
    )
    for r in top_reject:
        print(f"  [{r['risk_grade']}] {r['filename'][:50]}: accept={r['gate_acceptance_rate']:.1%}")

    # ── 2. Top 10 by TOC infiltration ────────────────────────────
    print("\n### TOP 10 PDFs BY TOC INFILTRATION")
    print("-" * 60)
    top_toc = await conn.fetch(
        """
        SELECT pm.filename,
               count(*) FILTER (WHERE cf.toc_infiltration_score > 0.6) as toc_chunks,
               count(*) as total_chunks
        FROM kb.chunk_features cf
        JOIN kb.pdf_manifest pm ON pm.id = cf.manifest_id
        GROUP BY pm.filename
        HAVING count(*) FILTER (WHERE cf.toc_infiltration_score > 0.6) > 0
        ORDER BY toc_chunks DESC
        LIMIT 10
        """,
    )
    for r in top_toc:
        pct = r["toc_chunks"] / r["total_chunks"] * 100 if r["total_chunks"] > 0 else 0
        print(f"  {r['filename'][:50]}: {r['toc_chunks']}/{r['total_chunks']} toc ({pct:.1f}%)")

    # ── 3. Coverage thresholds ────────────────────────────────────
    print("\n### COVERAGE THRESHOLDS")
    print("-" * 60)

    for threshold, label in [(85, "GUARDRAIL"), (90, "TARGET"), (95, "EXCELLENT")]:
        count = await conn.fetchval(
            """
            SELECT count(*)
            FROM kb.reference_alignment_summary
            WHERE ingest_batch_id = $1 AND coverage_pct >= $2
            """,
            batch_id, threshold,
        )
        total = await conn.fetchval(
            "SELECT count(*) FROM kb.reference_alignment_summary WHERE ingest_batch_id = $1",
            batch_id,
        )
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label} (>={threshold}%): {count}/{total} docs ({pct:.1f}%)")

    # ── 4. Retrieval comparison by method ─────────────────────────
    print("\n### RETRIEVAL COMPARISON BY METHOD")
    print("-" * 60)
    methods = await conn.fetch(
        """
        SELECT method,
               avg(avg_recall_10) as r10,
               avg(avg_mrr) as mrr,
               avg(avg_noise_rate_10) as noise
        FROM kb.retrieval_eval_summary
        WHERE ingest_batch_id = $1
        GROUP BY method
        ORDER BY r10 DESC
        """,
        batch_id,
    )
    for m in methods:
        print(
            f"  {m['method']}: R@10={float(m['r10'] or 0):.3f}, "
            f"MRR={float(m['mrr'] or 0):.3f}, "
            f"noise@10={float(m['noise'] or 0):.3f}"
        )

    # ── 5. Profile distribution ───────────────────────────────────
    print("\n### PROFILE DISTRIBUTION")
    print("-" * 60)
    profiles = await conn.fetch(
        """
        SELECT profile, count(*) as cnt
        FROM kb.qa_ingestion_profiles
        GROUP BY profile
        ORDER BY cnt DESC
        """,
    )
    for p in profiles:
        print(f"  {p['profile']}: {p['cnt']}")

    # ── 6. Label distribution ─────────────────────────────────────
    print("\n### LABEL DISTRIBUTION")
    print("-" * 60)
    labels = await conn.fetch(
        """
        SELECT final_label, count(*) as cnt
        FROM kb.chunk_labels
        GROUP BY final_label
        ORDER BY cnt DESC
        """,
    )
    total_labels = sum(r["cnt"] for r in labels)
    for l in labels:
        pct = l["cnt"] / total_labels * 100 if total_labels > 0 else 0
        print(f"  {l['final_label']}: {l['cnt']} ({pct:.1f}%)")

    # ── 7. Risk grade summary ─────────────────────────────────────
    print("\n### RISK GRADE SUMMARY")
    print("-" * 60)
    grades = await conn.fetch(
        """
        SELECT risk_grade, count(*) as cnt,
               avg(composite_risk_score) as avg_risk
        FROM kb.qa_document_reports
        WHERE ingest_batch_id = $1
        GROUP BY risk_grade
        ORDER BY risk_grade
        """,
        batch_id,
    )
    for g in grades:
        print(f"  {g['risk_grade']}: {g['cnt']} docs (avg risk={float(g['avg_risk'] or 0):.3f})")

    # ── 8. Actionable recommendations ─────────────────────────────
    print("\n### ACTIONABLE RECOMMENDATIONS")
    print("-" * 60)

    # High risk docs
    high_risk = await conn.fetchval(
        """
        SELECT count(*)
        FROM kb.qa_document_reports
        WHERE ingest_batch_id = $1 AND risk_grade IN ('D', 'F')
        """,
        batch_id,
    )
    if high_risk > 0:
        print(f"  [!] {high_risk} documents at risk grade D/F - prioritize re-ingestion")

    # Year conflicts
    conflicts = await conn.fetchval(
        "SELECT count(*) FROM kb.pdf_year_resolution WHERE has_conflict = true"
    )
    if conflicts > 0:
        print(f"  [!] {conflicts} documents with year conflicts - review manually")

    # Low coverage
    low_cov = await conn.fetchval(
        """
        SELECT count(*)
        FROM kb.reference_alignment_summary
        WHERE ingest_batch_id = $1 AND coverage_pct < 85
        """,
        batch_id,
    )
    if low_cov > 0:
        print(f"  [!] {low_cov} documents below 85% coverage guardrail")

    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS COMPLETE")
    print(f"{'=' * 70}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
