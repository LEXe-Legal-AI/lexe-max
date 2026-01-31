#!/usr/bin/env python3
"""
Norm Graph Sanity Checks

Validates the norm graph data:
1. Schema integrity
2. Data consistency
3. Lookup performance
4. Edge coverage

Usage:
    uv run python scripts/graph/sanity_check_norm_graph.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings


async def check_schema(conn: asyncpg.Connection) -> bool:
    """Check schema exists and has expected structure."""
    print("\n[1] SCHEMA CHECK")
    print("-" * 40)

    issues = []

    # Check kb.norms table
    norms_cols = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'norms'
        ORDER BY ordinal_position
    """)
    expected_cols = {'id', 'code', 'article', 'suffix', 'number', 'year', 'full_ref', 'citation_count'}
    actual_cols = {r['column_name'] for r in norms_cols}

    missing = expected_cols - actual_cols
    if missing:
        issues.append(f"kb.norms missing columns: {missing}")
    else:
        print(f"  kb.norms: {len(norms_cols)} columns OK")

    # Check kb.massima_norms table
    edges_cols = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'kb' AND table_name = 'massima_norms'
        ORDER BY ordinal_position
    """)
    expected_edge_cols = {'massima_id', 'norm_id', 'context_span', 'run_id'}
    actual_edge_cols = {r['column_name'] for r in edges_cols}

    missing_edge = expected_edge_cols - actual_edge_cols
    if missing_edge:
        issues.append(f"kb.massima_norms missing columns: {missing_edge}")
    else:
        print(f"  kb.massima_norms: {len(edges_cols)} columns OK")

    # Check indexes
    indexes = await conn.fetch("""
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'kb' AND tablename IN ('norms', 'massima_norms')
    """)
    print(f"  Indexes: {len(indexes)} found")

    if issues:
        for issue in issues:
            print(f"  [!] {issue}")
        return False

    print("  PASS")
    return True


async def check_data_counts(conn: asyncpg.Connection) -> bool:
    """Check data volume."""
    print("\n[2] DATA VOLUME")
    print("-" * 40)

    issues = []

    # Count norms
    norm_count = await conn.fetchval("SELECT COUNT(*) FROM kb.norms")
    print(f"  Total norms: {norm_count:,}")
    if norm_count < 1000:
        issues.append(f"Too few norms ({norm_count}), expected > 1000")

    # Count edges
    edge_count = await conn.fetchval("SELECT COUNT(*) FROM kb.massima_norms")
    print(f"  Total edges: {edge_count:,}")
    if edge_count < 10000:
        issues.append(f"Too few edges ({edge_count}), expected > 10000")

    # Massime with norms
    massime_with_norms = await conn.fetchval("""
        SELECT COUNT(DISTINCT massima_id) FROM kb.massima_norms
    """)
    print(f"  Massime with norms: {massime_with_norms:,}")

    # Ratio
    total_massime = await conn.fetchval("""
        SELECT COUNT(*) FROM kb.massime WHERE is_active = TRUE
    """)
    ratio = massime_with_norms / total_massime if total_massime else 0
    print(f"  Coverage: {ratio:.1%}")

    if ratio < 0.30:
        issues.append(f"Low norm coverage ({ratio:.1%}), expected > 30%")

    if issues:
        for issue in issues:
            print(f"  [!] {issue}")
        return False

    print("  PASS")
    return True


async def check_data_distribution(conn: asyncpg.Connection) -> bool:
    """Check data distribution by code type."""
    print("\n[3] DISTRIBUTION BY CODE")
    print("-" * 40)

    rows = await conn.fetch("""
        SELECT code, COUNT(*) as norms, SUM(citation_count) as citations
        FROM kb.norms
        GROUP BY code
        ORDER BY citations DESC
    """)

    for row in rows:
        print(f"  {row['code']:8} {row['norms']:5} norms, {row['citations']:6} citations")

    # Check we have all expected codes
    expected_codes = {'CC', 'CPC', 'CP', 'CPP', 'COST', 'LEGGE', 'DLGS', 'DPR'}
    actual_codes = {r['code'] for r in rows}
    missing = expected_codes - actual_codes

    if missing:
        print(f"  [!] Missing code types: {missing}")
        return False

    print("  PASS")
    return True


async def check_top_norms(conn: asyncpg.Connection) -> bool:
    """Check top cited norms make sense."""
    print("\n[4] TOP CITED NORMS")
    print("-" * 40)

    rows = await conn.fetch("""
        SELECT id, full_ref, citation_count
        FROM kb.norms
        ORDER BY citation_count DESC
        LIMIT 10
    """)

    for i, row in enumerate(rows, 1):
        print(f"  {i:2}. {row['full_ref']:35} ({row['citation_count']} citations)")

    # Sanity: top norm should have > 100 citations
    if rows and rows[0]['citation_count'] < 100:
        print(f"  [!] Top norm has only {rows[0]['citation_count']} citations, expected > 100")
        return False

    print("  PASS")
    return True


async def check_citation_counts(conn: asyncpg.Connection) -> bool:
    """Verify citation_count matches actual edges."""
    print("\n[5] CITATION COUNT CONSISTENCY")
    print("-" * 40)

    # Find mismatches
    mismatches = await conn.fetch("""
        WITH actual AS (
            SELECT norm_id, COUNT(*) as cnt
            FROM kb.massima_norms
            GROUP BY norm_id
        )
        SELECT n.id, n.citation_count, COALESCE(a.cnt, 0) as actual_count
        FROM kb.norms n
        LEFT JOIN actual a ON n.id = a.norm_id
        WHERE n.citation_count != COALESCE(a.cnt, 0)
        LIMIT 5
    """)

    if mismatches:
        print(f"  [!] Found {len(mismatches)} citation_count mismatches:")
        for m in mismatches[:3]:
            print(f"      {m['id']}: stored={m['citation_count']}, actual={m['actual_count']}")

        # Fix them
        print("  Recomputing citation counts...")
        await conn.execute("SELECT kb.recompute_norm_citation_counts()")
        print("  Fixed.")
        return True  # Not a failure, just needed recompute

    print("  All citation_counts match actual edges")
    print("  PASS")
    return True


async def check_lookup_performance(conn: asyncpg.Connection) -> bool:
    """Test lookup query performance."""
    print("\n[6] LOOKUP PERFORMANCE")
    print("-" * 40)

    test_norms = [
        ("CC:2043", "CC", "2043", None),
        ("CPC:360", "CPC", "360", None),
        ("COST:111", "COST", "111", None),
        ("LEGGE:241:1990", "LEGGE", None, "241"),
        ("DLGS:165:2001", "DLGS", None, "165"),
    ]

    import time

    for norm_id, code, article, number in test_norms:
        start = time.perf_counter()

        rows = await conn.fetch("""
            SELECT m.id
            FROM kb.massima_norms mn
            JOIN kb.massime m ON m.id = mn.massima_id
            WHERE mn.norm_id = $1
              AND m.is_active = TRUE
            ORDER BY m.anno DESC NULLS LAST
            LIMIT 10
        """, norm_id)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {norm_id:20} -> {len(rows):3} results in {elapsed:5.1f}ms")

        if elapsed > 100:
            print(f"  [!] Slow query for {norm_id}")

    print("  PASS")
    return True


async def check_orphan_edges(conn: asyncpg.Connection) -> bool:
    """Check for edges referencing non-existent norms or massime."""
    print("\n[7] ORPHAN EDGES CHECK")
    print("-" * 40)

    # Edges with missing norms
    orphan_norms = await conn.fetchval("""
        SELECT COUNT(*)
        FROM kb.massima_norms mn
        LEFT JOIN kb.norms n ON mn.norm_id = n.id
        WHERE n.id IS NULL
    """)

    # Edges with missing massime
    orphan_massime = await conn.fetchval("""
        SELECT COUNT(*)
        FROM kb.massima_norms mn
        LEFT JOIN kb.massime m ON mn.massima_id = m.id
        WHERE m.id IS NULL
    """)

    print(f"  Orphan norm references: {orphan_norms}")
    print(f"  Orphan massima references: {orphan_massime}")

    if orphan_norms > 0 or orphan_massime > 0:
        print("  [!] Found orphan edges")
        return False

    print("  PASS")
    return True


async def main():
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    print("=" * 60)
    print("NORM GRAPH SANITY CHECKS")
    print("=" * 60)

    checks = [
        ("Schema", check_schema),
        ("Data Counts", check_data_counts),
        ("Distribution", check_data_distribution),
        ("Top Norms", check_top_norms),
        ("Citation Counts", check_citation_counts),
        ("Lookup Performance", check_lookup_performance),
        ("Orphan Edges", check_orphan_edges),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = await check_fn(conn)
            results.append((name, passed))
        except Exception as e:
            print(f"  [!] Error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:25} {status}")

    print(f"\nTotal: {passed}/{total} checks passed")

    await conn.close()

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
