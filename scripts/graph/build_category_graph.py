#!/usr/bin/env python3
"""
Build Category Graph

Classifies all massime and stores category assignments.

Usage:
    uv run python scripts/graph/build_category_graph.py
    uv run python scripts/graph/build_category_graph.py --batch-size 500
    uv run python scripts/graph/build_category_graph.py --clear  # rebuild
    uv run python scripts/graph/build_category_graph.py --dry-run  # preview
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.category_classifier import classify_massima


async def build_category_graph(
    batch_size: int = 500,
    clear: bool = False,
    dry_run: bool = False,
):
    """Build category assignments for all massime."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Check categories exist
        cat_count = await conn.fetchval("SELECT COUNT(*) FROM kb.categories")
        if cat_count == 0:
            print("ERROR: No categories found! Run seed_categories.py first.")
            return

        print(f"Found {cat_count} categories")

        if clear and not dry_run:
            print("Clearing existing assignments...")
            await conn.execute("DELETE FROM kb.category_assignments")
            print("Cleared.")

        # Create run record
        run_id = None
        if not dry_run:
            config = json.dumps({"method": "keyword", "batch_size": batch_size})
            run_id = await conn.fetchval("""
                INSERT INTO kb.category_runs (run_type, status, config)
                VALUES ('classification', 'running', $1::jsonb)
                RETURNING id
            """, config)
            print(f"Started run {run_id}")

        # Count total massime
        total = await conn.fetchval("""
            SELECT COUNT(*) FROM kb.massime WHERE is_active = TRUE
        """)
        print(f"Processing {total} active massime...")

        # Stats
        assigned_l1 = 0
        assigned_l2 = 0
        unknown_count = 0
        total_processed = 0

        # Process in batches
        offset = 0
        start_time = time.time()

        while offset < total:
            rows = await conn.fetch("""
                SELECT id, testo
                FROM kb.massime
                WHERE is_active = TRUE
                ORDER BY id
                LIMIT $1 OFFSET $2
            """, batch_size, offset)

            if not rows:
                break

            batch_assignments = []

            for row in rows:
                massima_id = row["id"]
                text = row["testo"] or ""

                matches = classify_massima(text)

                if not matches:
                    unknown_count += 1
                    continue

                has_l1 = False
                for match in matches:
                    if match.level == 1:
                        has_l1 = True
                        assigned_l1 += 1
                    else:
                        assigned_l2 += 1

                    batch_assignments.append((
                        massima_id,
                        match.category_id,
                        match.confidence,
                        match.method,
                        match.evidence_terms,
                        run_id,
                    ))

                if not has_l1:
                    unknown_count += 1

            # Insert batch (no ON CONFLICT since we clear first)
            if batch_assignments and not dry_run:
                await conn.executemany("""
                    INSERT INTO kb.category_assignments
                        (massima_id, category_id, confidence, method, evidence_terms, run_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, batch_assignments)

            total_processed += len(rows)
            offset += batch_size

            # Progress
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"  {total_processed}/{total} ({rate:.0f}/s) - L1: {assigned_l1}, L2: {assigned_l2}")

        # Update run record
        if run_id and not dry_run:
            await conn.execute("""
                UPDATE kb.category_runs SET
                    status = 'completed',
                    completed_at = NOW(),
                    total_massime = $2,
                    assigned_l1 = $3,
                    assigned_l2 = $4,
                    unknown_count = $5
                WHERE id = $1
            """, run_id, total_processed, assigned_l1, assigned_l2, unknown_count)

        # Final stats
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("CATEGORY GRAPH BUILD COMPLETE")
        print("=" * 60)
        print(f"Total processed: {total_processed}")
        print(f"L1 assignments:  {assigned_l1}")
        print(f"L2 assignments:  {assigned_l2}")
        print(f"Unknown:         {unknown_count}")
        print(f"Elapsed:         {elapsed:.1f}s")

        if dry_run:
            print("\n[DRY RUN - no data written]")

        # Distribution by L1
        if not dry_run:
            print("\nDistribution by L1:")
            rows = await conn.fetch("""
                SELECT c.id, c.name, COUNT(*) as cnt
                FROM kb.category_assignments ca
                JOIN kb.categories c ON c.id = ca.category_id
                WHERE c.level = 1
                GROUP BY c.id, c.name
                ORDER BY cnt DESC
            """)
            for row in rows:
                pct = row["cnt"] / assigned_l1 * 100 if assigned_l1 > 0 else 0
                print(f"  {row['id']}: {row['cnt']} ({pct:.1f}%)")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build category graph")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--clear", action="store_true", help="Clear and rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    asyncio.run(build_category_graph(args.batch_size, args.clear, args.dry_run))
