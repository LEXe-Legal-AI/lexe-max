#!/usr/bin/env python3
"""
Build Norm Graph from Massime

Extracts norm references from all active massime and populates:
- kb.norms: unique norms with citation counts
- kb.massima_norms: edges connecting massime to cited norms

Usage:
    uv run python scripts/graph/build_norm_graph.py
    uv run python scripts/graph/build_norm_graph.py --batch-size 500 --dry-run
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

import asyncpg
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.norm_extractor import NormRef, extract_norms

logger = structlog.get_logger(__name__)


async def fetch_massime_batch(
    conn: asyncpg.Connection,
    offset: int,
    limit: int,
) -> list[dict]:
    """Fetch a batch of active massime."""
    rows = await conn.fetch(
        """
        SELECT id, testo_normalizzato
        FROM kb.massime
        WHERE is_active = TRUE
          AND testo_normalizzato IS NOT NULL
          AND LENGTH(testo_normalizzato) > 50
        ORDER BY id
        OFFSET $1
        LIMIT $2
        """,
        offset,
        limit,
    )
    return [dict(row) for row in rows]


async def upsert_norms(
    conn: asyncpg.Connection,
    norms: list[NormRef],
) -> int:
    """Upsert norms into kb.norms."""
    if not norms:
        return 0

    # Deduplicate by ID
    unique = {n.id: n for n in norms}

    values = [
        (n.id, n.code, n.article, n.suffix, n.number, n.year, n.full_ref)
        for n in unique.values()
    ]

    await conn.executemany(
        """
        INSERT INTO kb.norms (id, code, article, suffix, number, year, full_ref)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE
        SET full_ref = EXCLUDED.full_ref,
            updated_at = NOW()
        """,
        values,
    )

    return len(unique)


async def insert_massima_norms(
    conn: asyncpg.Connection,
    massima_id: str,
    norms: list[NormRef],
    run_id: int,
) -> int:
    """Insert massima-norm edges."""
    if not norms:
        return 0

    # Deduplicate by norm ID
    unique = {n.id: n for n in norms}

    values = [(massima_id, n.id, n.context_span[:500], run_id) for n in unique.values()]

    await conn.executemany(
        """
        INSERT INTO kb.massima_norms (massima_id, norm_id, context_span, run_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT DO NOTHING
        """,
        values,
    )

    return len(unique)


async def create_run(conn: asyncpg.Connection, run_type: str = "norm_graph") -> int:
    """Create a new graph run record."""
    run_id = await conn.fetchval(
        """
        INSERT INTO kb.graph_runs (run_type, status, started_at)
        VALUES ($1, 'running', NOW())
        RETURNING id
        """,
        run_type,
    )
    return run_id


async def complete_run(conn: asyncpg.Connection, run_id: int) -> None:
    """Mark run as completed."""
    await conn.execute(
        """
        UPDATE kb.graph_runs
        SET status = 'completed',
            completed_at = NOW()
        WHERE id = $1
        """,
        run_id,
    )


async def recompute_citation_counts(conn: asyncpg.Connection) -> int:
    """Recompute citation_count for all norms."""
    result = await conn.fetchval("SELECT kb.recompute_norm_citation_counts()")
    return result or 0


async def build_norm_graph(
    conn: asyncpg.Connection,
    batch_size: int = 500,
    dry_run: bool = False,
) -> dict:
    """Main extraction and build pipeline."""

    # Get total count
    total = await conn.fetchval(
        """
        SELECT COUNT(*)
        FROM kb.massime
        WHERE is_active = TRUE
          AND testo_normalizzato IS NOT NULL
          AND LENGTH(testo_normalizzato) > 50
        """
    )

    # Create run record
    run_id = None
    if not dry_run:
        run_id = await create_run(conn, "norm_graph")
        logger.info("Created run", run_id=run_id)

    logger.info("Starting norm graph build", total_massime=total, batch_size=batch_size)

    stats = {
        "total_massime": total,
        "processed": 0,
        "with_norms": 0,
        "total_norms_found": 0,
        "unique_norms": 0,
        "edges_created": 0,
        "code_distribution": Counter(),
    }

    all_norms: list[NormRef] = []
    offset = 0

    while offset < total:
        batch = await fetch_massime_batch(conn, offset, batch_size)
        if not batch:
            break

        batch_norms = []

        # Collect all norms from this batch first
        batch_edges = []  # (massima_id, norms)

        for massima in batch:
            norms = extract_norms(massima["testo_normalizzato"])

            if norms:
                stats["with_norms"] += 1
                stats["total_norms_found"] += len(norms)

                for n in norms:
                    stats["code_distribution"][n.code] += 1
                    batch_norms.append(n)

                batch_edges.append((str(massima["id"]), norms))

            stats["processed"] += 1

        # First upsert all norms (so FK constraints are satisfied)
        if not dry_run and batch_norms:
            await upsert_norms(conn, batch_norms)

        # Then insert edges
        if not dry_run:
            for massima_id, norms in batch_edges:
                await insert_massima_norms(conn, massima_id, norms, run_id)
                stats["edges_created"] += len(norms)

        all_norms.extend(batch_norms)
        offset += batch_size

        logger.info(
            "Batch processed",
            offset=offset,
            total=total,
            batch_norms=len(batch_norms),
            pct=f"{offset / total * 100:.1f}%",
        )

    # Unique norms
    stats["unique_norms"] = len({n.id for n in all_norms})

    # Recompute citation counts
    if not dry_run:
        updated = await recompute_citation_counts(conn)
        logger.info("Citation counts updated", updated=updated)

        # Complete run
        await complete_run(conn, run_id)

    return stats


async def main(args):
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    print("=" * 60)
    print("LEXE KB - Norm Graph Builder")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No database writes ***\n")

    if args.clear:
        print("Clearing existing norm data...")
        await conn.execute("TRUNCATE kb.massima_norms CASCADE")
        await conn.execute("TRUNCATE kb.norms CASCADE")
        print("  Cleared.\n")

    stats = await build_norm_graph(conn, args.batch_size, args.dry_run)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMassime processed:    {stats['processed']:,}")
    print(f"Massime with norms:   {stats['with_norms']:,} ({stats['with_norms'] / stats['processed'] * 100:.1f}%)")
    print(f"Total norms found:    {stats['total_norms_found']:,}")
    print(f"Unique norms:         {stats['unique_norms']:,}")
    print(f"Edges created:        {stats['edges_created']:,}")

    print("\nDistribution by code:")
    for code, count in sorted(stats["code_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {code:8} {count:,}")

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Norm Graph")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")

    args = parser.parse_args()
    asyncio.run(main(args))
