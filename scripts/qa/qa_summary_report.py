#!/usr/bin/env python3
"""
QA Protocol - Summary Report

Generates a summary report from available QA data:
- Document Intelligence (A/B test results)
- Guided Ingestion (massime count)
- Reference Alignment (coverage)

Usage:
    uv run python scripts/qa/qa_summary_report.py
"""

import asyncio
from collections import Counter
from datetime import datetime

import asyncpg

from qa_config import DB_URL


async def main():
    print("=" * 70)
    print("QA PROTOCOL - SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: Document Inventory
    # ═══════════════════════════════════════════════════════════════════
    print("1. DOCUMENT INVENTORY")
    print("-" * 70)

    manifest_count = await conn.fetchval("SELECT count(*) FROM kb.pdf_manifest")
    ref_units = await conn.fetchval("SELECT count(*) FROM kb.qa_reference_units")
    massime = await conn.fetchval("SELECT count(*) FROM kb.massime")
    docs = await conn.fetchval("SELECT count(*) FROM kb.documents")

    print(f"   PDF Manifest entries:     {manifest_count:>6}")
    print(f"   Reference units:          {ref_units:>6}")
    print(f"   Pipeline massime:         {massime:>6}")
    print(f"   Document records:         {docs:>6}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: Document Intelligence Results
    # ═══════════════════════════════════════════════════════════════════
    print("2. DOCUMENT INTELLIGENCE (A/B TEST)")
    print("-" * 70)

    # Run A results
    run_a = await conn.fetch("""
        SELECT doc_type, profile, count(*) as cnt
        FROM kb.doc_intel_ab_results
        WHERE run_name = 'A'
        GROUP BY doc_type, profile
        ORDER BY cnt DESC
    """)

    if run_a:
        total = sum(r["cnt"] for r in run_a)
        print(f"   Documents classified:     {total:>6}")
        print()

        # doc_type distribution
        doc_types = Counter()
        for r in run_a:
            doc_types[r["doc_type"]] += r["cnt"]

        print("   doc_type distribution:")
        for dt, cnt in doc_types.most_common():
            pct = 100 * cnt / total
            print(f"     {dt:30} {cnt:>4} ({pct:>5.1f}%)")
        print()

        # profile distribution
        profiles = Counter()
        for r in run_a:
            profiles[r["profile"]] += r["cnt"]

        print("   profile distribution:")
        for p, cnt in profiles.most_common():
            pct = 100 * cnt / total
            print(f"     {p:30} {cnt:>4} ({pct:>5.1f}%)")
        print()

        # A/B comparison
        comparison = await conn.fetch("""
            SELECT
                a.filename,
                a.doc_type as a_type, b.doc_type as b_type,
                a.profile as a_prof, b.profile as b_prof
            FROM kb.doc_intel_ab_results a
            JOIN kb.doc_intel_ab_results b ON a.manifest_id = b.manifest_id
            WHERE a.run_name = 'A' AND b.run_name = 'B'
        """)

        if comparison:
            type_flips = sum(1 for r in comparison if r["a_type"] != r["b_type"])
            prof_flips = sum(1 for r in comparison if r["a_prof"] != r["b_prof"])
            print(f"   A/B Comparison ({len(comparison)} docs):")
            print(f"     doc_type flip rate:     {type_flips:>4} ({100*type_flips/len(comparison):.1f}%)")
            print(f"     profile flip rate:      {prof_flips:>4} ({100*prof_flips/len(comparison):.1f}%)")
            print()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: Ingestion Stats
    # ═══════════════════════════════════════════════════════════════════
    print("3. INGESTION STATISTICS")
    print("-" * 70)

    batches = await conn.fetch("""
        SELECT batch_name, pipeline, status, started_at, completed_at
        FROM kb.ingest_batches
        ORDER BY id
    """)

    print("   Batches:")
    for b in batches:
        print(f"     {b['batch_name']:25} {b['pipeline'] or 'N/A':15} {b['status']}")
    print()

    # Massime per document
    massime_stats = await conn.fetch("""
        SELECT d.source_path, count(m.id) as cnt
        FROM kb.documents d
        LEFT JOIN kb.massime m ON m.document_id = d.id
        GROUP BY d.id, d.source_path
        ORDER BY cnt DESC
        LIMIT 10
    """)

    if massime_stats:
        print("   Top 10 documents by massime count:")
        for m in massime_stats:
            print(f"     {m['source_path'][:45]:45} {m['cnt']:>5}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: Reference Alignment
    # ═══════════════════════════════════════════════════════════════════
    print("4. REFERENCE ALIGNMENT")
    print("-" * 70)

    alignment = await conn.fetch("""
        SELECT
            coverage_pct, fragmentation_score, fusion_score, avg_overlap,
            total_ref_units, matched_count, unmatched_count
        FROM kb.reference_alignment_summary
    """)

    if alignment:
        total_ref = sum(r["total_ref_units"] for r in alignment)
        total_matched = sum(r["matched_count"] for r in alignment)
        total_unmatched = sum(r["unmatched_count"] for r in alignment)
        avg_coverage = sum(r["coverage_pct"] for r in alignment) / len(alignment)
        avg_frag = sum(r["fragmentation_score"] for r in alignment) / len(alignment)

        print(f"   Documents analyzed:       {len(alignment):>6}")
        print(f"   Total reference units:    {total_ref:>6}")
        print(f"   Total matched:            {total_matched:>6}")
        print(f"   Total unmatched:          {total_unmatched:>6}")
        print(f"   Average coverage:         {avg_coverage:>5.1f}%")
        print(f"   Average fragmentation:    {avg_frag:>5.2f}")
        print()

        # Coverage buckets
        buckets = {"0%": 0, "1-25%": 0, "26-50%": 0, "51-75%": 0, "76-100%": 0}
        for a in alignment:
            cov = a["coverage_pct"]
            if cov == 0:
                buckets["0%"] += 1
            elif cov <= 25:
                buckets["1-25%"] += 1
            elif cov <= 50:
                buckets["26-50%"] += 1
            elif cov <= 75:
                buckets["51-75%"] += 1
            else:
                buckets["76-100%"] += 1

        print("   Coverage distribution:")
        for bucket, cnt in buckets.items():
            pct = 100 * cnt / len(alignment)
            bar = "#" * int(pct / 2)
            print(f"     {bucket:>10}: {cnt:>4} ({pct:>5.1f}%) {bar}")
        print()
    else:
        print("   No alignment data available")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: Key Findings
    # ═══════════════════════════════════════════════════════════════════
    print("5. KEY FINDINGS")
    print("-" * 70)

    findings = []

    # Finding 1: Doc Intel stability
    if comparison and type_flips == 0:
        findings.append("[OK] Document Intelligence: 100% stable (0% doc_type flip)")
    elif comparison:
        findings.append(f"[WARN] Document Intelligence: {100*type_flips/len(comparison):.1f}% doc_type flip")

    # Finding 2: Profile assignments
    if profiles:
        top_profile = profiles.most_common(1)[0][0]
        top_pct = 100 * profiles.most_common(1)[0][1] / total
        findings.append(f"[INFO] Primary profile: {top_profile} ({top_pct:.0f}%)")

    # Finding 3: Coverage issue
    if alignment and avg_coverage < 10:
        findings.append("[WARN] Low reference coverage (< 10%) - extraction method mismatch")
    elif alignment:
        findings.append(f"[INFO] Reference coverage: {avg_coverage:.1f}%")

    # Finding 4: Massime extraction
    if massime > 5000:
        findings.append(f"[OK] {massime} massime extracted successfully")
    else:
        findings.append(f"[WARN] Only {massime} massime extracted")

    for f in findings:
        print(f"   {f}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: Recommendations
    # ═══════════════════════════════════════════════════════════════════
    print("6. RECOMMENDATIONS")
    print("-" * 70)

    recommendations = [
        "1. Use temperature=0.0 for Document Intelligence (deterministic)",
        "2. Apply 'structured_parent_child' profile for most documents",
        "3. Use same extraction method for reference units and pipeline massime",
        "4. Implement TOC detection and skip for 'baseline_toc_filter' profiles",
        "5. Consider Unstructured hi_res for complex legacy documents",
    ]

    for r in recommendations:
        print(f"   {r}")
    print()

    print("=" * 70)
    print("[REPORT COMPLETE]")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
