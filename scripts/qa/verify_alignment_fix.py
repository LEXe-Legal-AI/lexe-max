#!/usr/bin/env python3
"""
QA Protocol - Verify Alignment Fix

Runs verification queries E1-E6 to confirm the alignment fix worked.
Checks all guardrails and reports pass/fail status.

Usage:
    uv run python scripts/qa/verify_alignment_fix.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

from qa_config import DB_URL

# Guardrails
GUARDRAILS = {
    "coverage_pct": {"min": 60, "target": 85},
    "alignment_trust": {"min": 0.90, "target": 0.95},
    "embedding_pct": {"max": 10, "target": 5},
    "collision_rate": {"max": 5, "target": 3},
}


async def main():
    print("=" * 70)
    print("QA PROTOCOL - VERIFY ALIGNMENT FIX")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get latest qa_run_id and batch_id
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'guided_local_v1'"
    )

    print(f"qa_run_id: {qa_run_id}")
    print(f"batch_id: {batch_id}")
    print()

    results = []

    # =========================================================================
    # E1. Spaced Letters Score per Engine
    # =========================================================================
    print("=" * 70)
    print("E1. SPACED LETTERS SCORE PER ENGINE")
    print("-" * 70)

    e1 = await conn.fetch(
        """
        SELECT extraction_engine,
               count(*) as n_units,
               round(avg(spaced_letters_score)::numeric, 4) as avg_score,
               sum(CASE WHEN spaced_letters_score > 0.12 THEN 1 ELSE 0 END) as n_spaced
        FROM kb.qa_reference_units
        WHERE qa_run_id = $1
        GROUP BY extraction_engine
        ORDER BY extraction_engine
        """,
        qa_run_id,
    )

    for row in e1:
        engine = row["extraction_engine"]
        n_units = row["n_units"]
        avg_score = float(row["avg_score"] or 0)
        n_spaced = row["n_spaced"]
        status = "OK" if (engine == "pymupdf" and n_spaced == 0) else ("WARN" if n_spaced < 10 else "FAIL")
        print(f"  {engine:25} {n_units:>6} units, avg_score={avg_score:.4f}, n_spaced={n_spaced} {status}")

    pymupdf_spaced = next((r["n_spaced"] for r in e1 if r["extraction_engine"] == "pymupdf"), 0)
    results.append(("E1: PyMuPDF n_spaced", pymupdf_spaced == 0, f"{pymupdf_spaced} (expected 0)"))
    print()

    # =========================================================================
    # E2. Coverage Prima vs Dopo
    # =========================================================================
    print("=" * 70)
    print("E2. COVERAGE BY BATCH")
    print("-" * 70)

    e2 = await conn.fetch(
        """
        SELECT ingest_batch_id,
               count(*) as n_docs,
               round(avg(coverage_pct)::numeric, 1) as avg_cov,
               round(min(coverage_pct)::numeric, 1) as min_cov,
               round(max(coverage_pct)::numeric, 1) as max_cov
        FROM kb.reference_alignment_summary
        WHERE qa_run_id = $1
        GROUP BY ingest_batch_id
        ORDER BY ingest_batch_id
        """,
        qa_run_id,
    )

    for row in e2:
        batch = row["ingest_batch_id"]
        n_docs = row["n_docs"]
        avg_cov = float(row["avg_cov"] or 0)
        min_cov = float(row["min_cov"] or 0)
        max_cov = float(row["max_cov"] or 0)
        status = "OK" if avg_cov >= 60 else "FAIL"
        print(f"  batch={batch}: {n_docs:>3} docs, avg={avg_cov:5.1f}%, min={min_cov:5.1f}%, max={max_cov:5.1f}% {status}")

    avg_coverage = float(e2[0]["avg_cov"]) if e2 else 0
    results.append(("E2: avg_coverage_pct", avg_coverage >= 60, f"{avg_coverage:.1f}% (>=60%)"))
    print()

    # =========================================================================
    # E3. Match Stages Distribution
    # =========================================================================
    print("=" * 70)
    print("E3. MATCH STAGES DISTRIBUTION")
    print("-" * 70)

    e3 = await conn.fetch(
        """
        SELECT match_stage,
               count(*) as n,
               round(100.0 * count(*) / sum(count(*)) over(), 1) as pct
        FROM kb.reference_alignment
        WHERE ingest_batch_id = $1 AND qa_run_id = $2
        GROUP BY match_stage
        ORDER BY n DESC
        """,
        batch_id, qa_run_id,
    )

    for row in e3:
        stage = row["match_stage"] or "unmatched"
        n = row["n"]
        pct = float(row["pct"] or 0)
        print(f"  {stage:20} {n:>6} ({pct:5.1f}%)")

    embedding_pct = next((float(r["pct"]) for r in e3 if r["match_stage"] == "embedding"), 0)
    results.append(("E3: embedding_pct", embedding_pct < 10, f"{embedding_pct:.1f}% (<10%)"))
    print()

    # =========================================================================
    # E4. Documenti Critici (Coverage Bassa)
    # =========================================================================
    print("=" * 70)
    print("E4. DOCUMENTS WITH LOW COVERAGE (<60%)")
    print("-" * 70)

    e4 = await conn.fetch(
        """
        SELECT m.filename,
               s.coverage_pct,
               s.total_ref_units,
               s.unmatched_count
        FROM kb.reference_alignment_summary s
        JOIN kb.pdf_manifest m ON m.id = s.manifest_id
        WHERE s.coverage_pct < 60 AND s.qa_run_id = $1 AND s.ingest_batch_id = $2
        ORDER BY s.coverage_pct ASC
        LIMIT 10
        """,
        qa_run_id, batch_id,
    )

    if e4:
        for row in e4:
            filename = row["filename"][:45]
            cov = float(row["coverage_pct"] or 0)
            total = row["total_ref_units"]
            unmatched = row["unmatched_count"]
            print(f"  {filename:45} {cov:5.1f}% ({unmatched}/{total} unmatched)")
    else:
        print("  [None - all documents have coverage >=60%]")

    low_coverage_count = len(e4)
    results.append(("E4: low_coverage_docs", low_coverage_count == 0, f"{low_coverage_count} docs <60%"))
    print()

    # =========================================================================
    # E5. Alignment Trust
    # =========================================================================
    print("=" * 70)
    print("E5. ALIGNMENT TRUST")
    print("-" * 70)

    e5 = await conn.fetch(
        """
        WITH stage_counts AS (
            SELECT ingest_batch_id, qa_run_id,
                   count(*) FILTER (WHERE match_stage IN ('exact_hash','token_jaccard','char_ngram')) as trusted,
                   count(*) FILTER (WHERE match_stage IS NOT NULL) as total_matched,
                   count(*) FILTER (WHERE match_stage = 'embedding') as embedding_count
            FROM kb.reference_alignment
            WHERE qa_run_id = $1
            GROUP BY ingest_batch_id, qa_run_id
        )
        SELECT ingest_batch_id, qa_run_id,
               round(100.0 * trusted / NULLIF(total_matched, 0), 1) as alignment_trust_pct,
               round(100.0 * embedding_count / NULLIF(total_matched, 0), 1) as embedding_pct
        FROM stage_counts
        """,
        qa_run_id,
    )

    for row in e5:
        batch = row["ingest_batch_id"]
        trust = float(row["alignment_trust_pct"] or 0)
        emb = float(row["embedding_pct"] or 0)
        status = "OK" if trust >= 90 else "FAIL"
        print(f"  batch={batch}: alignment_trust={trust:5.1f}%, embedding={emb:5.1f}% {status}")

    alignment_trust = float(e5[0]["alignment_trust_pct"]) if e5 else 0
    results.append(("E5: alignment_trust", alignment_trust >= 90, f"{alignment_trust:.1f}% (>=90%)"))
    print()

    # =========================================================================
    # E6. Collision Rate
    # =========================================================================
    print("=" * 70)
    print("E6. COLLISION RATE (top 10 by collision)")
    print("-" * 70)

    e6 = await conn.fetch(
        """
        WITH collision_counts AS (
            SELECT manifest_id, qa_run_id,
                   count(*) as total_matches,
                   count(DISTINCT matched_massima_id) as unique_matches
            FROM kb.reference_alignment
            WHERE matched_massima_id IS NOT NULL AND qa_run_id = $1
            GROUP BY manifest_id, qa_run_id
        )
        SELECT m.filename,
               c.total_matches,
               c.unique_matches,
               round(100.0 * (c.total_matches - c.unique_matches) / NULLIF(c.total_matches, 0), 1) as collision_rate_pct
        FROM collision_counts c
        JOIN kb.pdf_manifest m ON m.id = c.manifest_id
        WHERE (c.total_matches - c.unique_matches) > 0
        ORDER BY collision_rate_pct DESC
        LIMIT 10
        """,
        qa_run_id,
    )

    if e6:
        for row in e6:
            filename = row["filename"][:40]
            total = row["total_matches"]
            unique = row["unique_matches"]
            rate = float(row["collision_rate_pct"] or 0)
            status = "OK" if rate < 3 else ("WARN" if rate < 5 else "FAIL")
            print(f"  {filename:40} {total:>4} matches, {unique:>4} unique, collision={rate:5.1f}% {status}")
    else:
        print("  [None - no collisions detected]")

    # Get average collision rate
    avg_collision = await conn.fetchval(
        """
        SELECT round(avg(collision_rate)::numeric, 2)
        FROM kb.reference_alignment_summary
        WHERE qa_run_id = $1 AND ingest_batch_id = $2
        """,
        qa_run_id, batch_id,
    )
    avg_collision = float(avg_collision or 0)
    results.append(("E6: avg_collision_rate", avg_collision < 5, f"{avg_collision:.1f}% (<5%)"))
    print()

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    all_passed = True
    for name, passed, value in results:
        status = "OK PASS" if passed else "FAIL FAIL"
        print(f"  {name:30} {status:10} {value}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  +------------------------------------------+")
        print("  |  ALL GUARDRAILS PASSED                   |")
        print("  |  Alignment fix is SUCCESSFUL             |")
        print("  +------------------------------------------+")
    else:
        print("  +------------------------------------------+")
        print("  |  SOME GUARDRAILS FAILED                  |")
        print("  |  Review issues above                     |")
        print("  +------------------------------------------+")

    await conn.close()
    print()


if __name__ == "__main__":
    asyncio.run(main())
