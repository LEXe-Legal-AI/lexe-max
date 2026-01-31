#!/usr/bin/env python3
"""
Seed Category Definitions

Populates kb.categories with L1 and L2 category definitions.
Run once after migration 007.

Usage:
    uv run python scripts/graph/seed_categories.py
    uv run python scripts/graph/seed_categories.py --clear  # clear and reseed
"""

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.categories import ALL_CATEGORIES


async def seed_categories(clear: bool = False):
    """Seed all category definitions."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        if clear:
            print("Clearing existing categories...")
            await conn.execute("DELETE FROM kb.category_assignments")
            await conn.execute("DELETE FROM kb.categories")
            print("Cleared.")

        # Check existing
        existing = await conn.fetchval("SELECT COUNT(*) FROM kb.categories")
        if existing > 0 and not clear:
            print(f"Categories already seeded ({existing} records). Use --clear to reseed.")
            return

        # Insert L1 first (no parent), then L2
        l1_cats = [c for c in ALL_CATEGORIES if c.level == 1]
        l2_cats = [c for c in ALL_CATEGORIES if c.level == 2]

        print(f"Seeding {len(l1_cats)} L1 categories...")
        for cat in l1_cats:
            await conn.execute("""
                INSERT INTO kb.categories (id, name, description, level, parent_id, keywords)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    keywords = EXCLUDED.keywords,
                    updated_at = NOW()
            """, cat.id, cat.name, cat.description, cat.level, cat.parent_id, cat.keywords)

        print(f"Seeding {len(l2_cats)} L2 categories...")
        for cat in l2_cats:
            await conn.execute("""
                INSERT INTO kb.categories (id, name, description, level, parent_id, keywords)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    keywords = EXCLUDED.keywords,
                    updated_at = NOW()
            """, cat.id, cat.name, cat.description, cat.level, cat.parent_id, cat.keywords)

        print(f"\n[OK] Seeded {len(ALL_CATEGORIES)} categories total")
        print(f"   - L1: {len(l1_cats)}")
        print(f"   - L2: {len(l2_cats)}")

        # Verify
        rows = await conn.fetch("""
            SELECT level, COUNT(*) as cnt
            FROM kb.categories
            GROUP BY level
            ORDER BY level
        """)
        print("\nVerification:")
        for row in rows:
            print(f"   Level {row['level']}: {row['cnt']} categories")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed categories")
    parser.add_argument("--clear", action="store_true", help="Clear and reseed")
    args = parser.parse_args()

    asyncio.run(seed_categories(args.clear))
