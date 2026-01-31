#!/usr/bin/env python3
"""
Sanity Check Category Graph

Validates category graph integrity and distribution.

Usage:
    uv run python scripts/graph/sanity_check_category_graph.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings


async def run_sanity_checks():
    """Run all sanity checks."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    checks_passed = 0
    checks_failed = 0

    def check(name: str, passed: bool, detail: str = ""):
        nonlocal checks_passed, checks_failed
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if detail:
            print(f"       {detail}")
        if passed:
            checks_passed += 1
        else:
            checks_failed += 1

    print("=" * 60)
    print("CATEGORY GRAPH SANITY CHECKS")
    print("=" * 60)

    try:
        # 1. Schema exists
        print("\n--- Schema Integrity ---")
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'kb' AND table_name IN ('categories', 'category_assignments', 'category_runs')
        """)
        table_names = [r["table_name"] for r in tables]
        check("categories table exists", "categories" in table_names)
        check("category_assignments table exists", "category_assignments" in table_names)
        check("category_runs table exists", "category_runs" in table_names)

        # 2. Categories populated
        print("\n--- Category Definitions ---")
        l1_count = await conn.fetchval("SELECT COUNT(*) FROM kb.categories WHERE level = 1")
        l2_count = await conn.fetchval("SELECT COUNT(*) FROM kb.categories WHERE level = 2")
        check("L1 categories populated", l1_count == 8, f"{l1_count} categories (expected 8)")
        check("L2 categories populated", l2_count >= 30, f"{l2_count} categories (expected ~40)")

        # 3. All L2 have valid parent
        orphan_l2 = await conn.fetchval("""
            SELECT COUNT(*) FROM kb.categories c
            WHERE c.level = 2 AND c.parent_id NOT IN (SELECT id FROM kb.categories WHERE level = 1)
        """)
        check("All L2 have valid L1 parent", orphan_l2 == 0, f"{orphan_l2} orphan L2")

        # 4. Assignment data volume
        print("\n--- Assignment Volume ---")
        total_assignments = await conn.fetchval("SELECT COUNT(*) FROM kb.category_assignments")
        l1_assignments = await conn.fetchval("""
            SELECT COUNT(*) FROM kb.category_assignments ca
            JOIN kb.categories c ON c.id = ca.category_id WHERE c.level = 1
        """)
        l2_assignments = await conn.fetchval("""
            SELECT COUNT(*) FROM kb.category_assignments ca
            JOIN kb.categories c ON c.id = ca.category_id WHERE c.level = 2
        """)

        check("Total assignments > 0", total_assignments > 0, f"{total_assignments} total")
        check("L1 assignments > 30K", l1_assignments >= 30000, f"{l1_assignments} L1 assignments")
        print(f"       L2 assignments: {l2_assignments}")

        # 5. Coverage
        print("\n--- Coverage ---")
        total_active = await conn.fetchval("SELECT COUNT(*) FROM kb.massime WHERE is_active = TRUE")
        with_l1 = await conn.fetchval("""
            SELECT COUNT(DISTINCT ca.massima_id)
            FROM kb.category_assignments ca
            JOIN kb.categories c ON c.id = ca.category_id
            WHERE c.level = 1
        """)
        coverage = with_l1 / total_active * 100 if total_active > 0 else 0
        check("L1 coverage >= 80%", coverage >= 80, f"{coverage:.1f}% ({with_l1}/{total_active})")

        # 6. Distribution balance
        print("\n--- Distribution Balance ---")
        rows = await conn.fetch("""
            SELECT c.id, c.name, COUNT(*) as cnt
            FROM kb.category_assignments ca
            JOIN kb.categories c ON c.id = ca.category_id
            WHERE c.level = 1
            GROUP BY c.id, c.name
            ORDER BY cnt DESC
        """)

        if rows:
            max_count = rows[0]["cnt"]
            min_count = rows[-1]["cnt"]
            ratio = max_count / min_count if min_count > 0 else float("inf")

            # Imbalance is expected (CIVILE and PROCESSUALE_CIVILE dominate)
            check("Distribution ratio < 50:1", ratio < 50, f"Max/Min ratio: {ratio:.1f}")

            print("\nL1 Distribution:")
            for row in rows:
                pct = row["cnt"] / l1_assignments * 100 if l1_assignments > 0 else 0
                print(f"  {row['id']:25} {row['cnt']:6} ({pct:5.1f}%)")

        # 7. Confidence distribution
        print("\n--- Confidence Distribution ---")
        conf_stats = await conn.fetchrow("""
            SELECT
                AVG(confidence) as avg_conf,
                MIN(confidence) as min_conf,
                MAX(confidence) as max_conf,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY confidence) as median_conf
            FROM kb.category_assignments
        """)
        if conf_stats:
            print(f"  Avg confidence:    {conf_stats['avg_conf']:.3f}")
            print(f"  Median confidence: {conf_stats['median_conf']:.3f}")
            print(f"  Min confidence:    {conf_stats['min_conf']:.3f}")
            print(f"  Max confidence:    {conf_stats['max_conf']:.3f}")
            check("Avg confidence >= 0.50", conf_stats['avg_conf'] >= 0.50)

        # 8. L2 by parent
        print("\n--- L2 by Parent ---")
        l2_by_parent = await conn.fetch("""
            SELECT c.parent_id, COUNT(*) as cnt
            FROM kb.category_assignments ca
            JOIN kb.categories c ON c.id = ca.category_id
            WHERE c.level = 2
            GROUP BY c.parent_id
            ORDER BY cnt DESC
        """)
        if l2_by_parent:
            for row in l2_by_parent:
                print(f"  {row['parent_id']:25} {row['cnt']:6} L2 assignments")

        # 9. No orphan assignments
        print("\n--- Referential Integrity ---")
        orphan_assignments = await conn.fetchval("""
            SELECT COUNT(*) FROM kb.category_assignments ca
            WHERE NOT EXISTS (SELECT 1 FROM kb.massime m WHERE m.id = ca.massima_id)
        """)
        check("No orphan assignments", orphan_assignments == 0, f"{orphan_assignments} orphan")

        invalid_categories = await conn.fetchval("""
            SELECT COUNT(*) FROM kb.category_assignments ca
            WHERE NOT EXISTS (SELECT 1 FROM kb.categories c WHERE c.id = ca.category_id)
        """)
        check("All categories valid", invalid_categories == 0, f"{invalid_categories} invalid")

        # 10. Run tracking
        print("\n--- Run Tracking ---")
        last_run = await conn.fetchrow("""
            SELECT id, status, completed_at, total_massime, assigned_l1, assigned_l2
            FROM kb.category_runs
            ORDER BY id DESC LIMIT 1
        """)
        if last_run:
            print(f"  Last run: #{last_run['id']} - {last_run['status']}")
            print(f"  Completed: {last_run['completed_at']}")
            print(f"  Total: {last_run['total_massime']}, L1: {last_run['assigned_l1']}, L2: {last_run['assigned_l2']}")
            check("Last run completed", last_run["status"] == "completed")
        else:
            check("Run tracking exists", False, "No runs found")

        # Summary
        print("\n" + "=" * 60)
        print(f"RESULTS: {checks_passed} passed, {checks_failed} failed")
        print("=" * 60)

        return checks_failed == 0

    finally:
        await conn.close()


if __name__ == "__main__":
    success = asyncio.run(run_sanity_checks())
    sys.exit(0 if success else 1)
