#!/usr/bin/env python3
"""
Build Citation Graph

Constructs the citation graph with dual-write (SQL + AGE).

Process:
1. Creates a graph_run record for versioning
2. Processes massime in batches
3. Extracts citation mentions (Step 1)
4. Resolves mentions to massima_id (Step 2)
5. Deduplicates edges
6. Dual-writes to SQL and AGE
7. Reports metrics

Usage:
    # Dry run (no writes)
    uv run python scripts/graph/build_citation_graph.py

    # With commit (writes to DB)
    uv run python scripts/graph/build_citation_graph.py --commit

    # Skip AGE writes (faster, SQL only)
    uv run python scripts/graph/build_citation_graph.py --commit --skip-age

    # Custom batch size
    uv run python scripts/graph/build_citation_graph.py --commit --batch-size 500

    # Limit massime (for testing)
    uv run python scripts/graph/build_citation_graph.py --commit --limit 1000
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.citation_extractor import (
    extract_mentions,
    resolve_mention,
    dedupe_mentions,
    compute_weight,
    build_evidence,
)
from src.lexe_api.kb.graph.edge_builder import (
    GraphRunMetrics,
    create_graph_run,
    complete_graph_run,
    fail_graph_run,
    insert_edges_dual,
    fetch_massime_batch,
    count_active_massime,
    get_edge_stats,
)


async def build_citation_graph(
    batch_size: int = 1000,
    limit: Optional[int] = None,
    commit: bool = False,
    skip_age: bool = False,
    verbose: bool = True,
) -> GraphRunMetrics:
    """
    Build the citation graph.

    Args:
        batch_size: Number of massime to process per batch
        limit: Maximum massime to process (None = all)
        commit: If True, write to database
        skip_age: If True, skip AGE writes (SQL only)
        verbose: Print progress

    Returns:
        GraphRunMetrics with final statistics
    """
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    # Initialize metrics
    metrics = GraphRunMetrics(
        run_id=0,
        run_type="citation_extraction",
        started_at=datetime.now(),
    )

    run_id = None

    try:
        # Get total count
        total_massime = await count_active_massime(conn)
        if limit:
            total_massime = min(total_massime, limit)

        if verbose:
            print("=" * 60)
            print("LEXE KB - Citation Graph Builder")
            print("=" * 60)
            print(f"\nConfiguration:")
            print(f"  Batch size:    {batch_size}")
            print(f"  Total massime: {total_massime:,}")
            print(f"  Commit:        {commit}")
            print(f"  Skip AGE:      {skip_age}")
            print()

        # Create graph run (if committing)
        if commit:
            run_id = await create_graph_run(
                conn,
                "citation_extraction",
                config={
                    "batch_size": batch_size,
                    "limit": limit,
                    "skip_age": skip_age,
                },
            )
            metrics.run_id = run_id

        # Process in batches
        offset = 0
        start_time = time.time()

        while offset < total_massime:
            batch_start = time.time()
            batch = await fetch_massime_batch(conn, batch_size, offset)

            if not batch:
                break

            # Process batch
            all_resolved = []
            batch_mentions = 0
            batch_resolved = 0

            for massima in batch:
                massima_id = massima["id"]
                testo = massima["testo"]

                # Step 1: Extract mentions
                mentions = extract_mentions(testo)
                batch_mentions += len(mentions)
                metrics.total_mentions_extracted += len(mentions)

                # Step 2: Resolve each mention
                for mention in mentions:
                    target_id, resolver = await resolve_mention(mention, conn, massima_id)

                    if target_id:
                        all_resolved.append((massima_id, mention, target_id, resolver))
                        batch_resolved += 1
                        metrics.total_resolved += 1
                        metrics.by_resolver[resolver] = metrics.by_resolver.get(resolver, 0) + 1
                    else:
                        metrics.total_unresolved += 1

            # Deduplicate
            edges_before = len(all_resolved)
            edges = dedupe_mentions(all_resolved)
            metrics.total_deduped += edges_before - len(edges)

            # Write edges (if committing)
            if commit and edges:
                sql_count, age_count = await insert_edges_dual(
                    conn, edges, run_id, skip_age=skip_age
                )
                metrics.total_edges_created += sql_count

            metrics.total_massime_processed += len(batch)
            offset += batch_size

            # Progress
            if verbose:
                batch_time = time.time() - batch_start
                progress = min(offset, total_massime) / total_massime * 100
                print(
                    f"  [{progress:5.1f}%] Batch {offset//batch_size}: "
                    f"{len(batch)} massime, {batch_mentions} mentions, "
                    f"{batch_resolved} resolved, {len(edges)} edges "
                    f"({batch_time:.1f}s)"
                )

        # Calculate final rates
        total_mentions = metrics.total_resolved + metrics.total_unresolved
        if total_mentions > 0:
            metrics.resolution_rate = metrics.total_resolved / total_mentions
        if metrics.total_resolved > 0:
            metrics.dedup_rate = metrics.total_deduped / (metrics.total_resolved)

        # Complete run
        if commit and run_id:
            metrics.completed_at = datetime.now()
            await complete_graph_run(conn, run_id, metrics)

        # Report
        elapsed = time.time() - start_time
        if verbose:
            print()
            print("=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"\nMassime processed:   {metrics.total_massime_processed:,}")
            print(f"Mentions extracted:  {metrics.total_mentions_extracted:,}")
            print(f"Resolved:            {metrics.total_resolved:,}")
            print(f"Unresolved:          {metrics.total_unresolved:,}")
            print(f"Resolution rate:     {metrics.resolution_rate:.1%}")
            print(f"Deduped:             {metrics.total_deduped:,}")
            print(f"Dedup rate:          {metrics.dedup_rate:.1%}")
            print(f"Edges created:       {metrics.total_edges_created:,}")
            print(f"Elapsed time:        {elapsed:.1f}s")
            print(f"\nBy resolver:")
            for resolver, count in sorted(metrics.by_resolver.items(), key=lambda x: -x[1]):
                print(f"  {resolver}: {count:,}")

            if commit:
                # Get final stats from DB
                stats = await get_edge_stats(conn, run_id)
                print(f"\nDatabase stats (run {run_id}):")
                print(f"  Total edges:       {stats.get('total_edges', 0):,}")
                print(f"  Unique sources:    {stats.get('unique_sources', 0):,}")
                print(f"  Unique targets:    {stats.get('unique_targets', 0):,}")
                print(f"  With subtype:      {stats.get('edges_with_subtype', 0):,}")
                print(f"  Avg weight:        {stats.get('avg_weight', 0):.2f}")
                print(f"  Valid weight (>=0.6): {stats.get('valid_weight_count', 0):,}")
            else:
                print("\n[DRY-RUN] No changes written to database.")

        return metrics

    except Exception as e:
        if run_id:
            await fail_graph_run(conn, run_id, str(e))
        raise

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build citation graph from massime"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Write changes to database (default: dry-run)",
    )
    parser.add_argument(
        "--skip-age",
        action="store_true",
        help="Skip AGE writes, SQL only (faster)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of massime to process (for testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    metrics = asyncio.run(
        build_citation_graph(
            batch_size=args.batch_size,
            limit=args.limit,
            commit=args.commit,
            skip_age=args.skip_age,
            verbose=not args.quiet,
        )
    )

    # Exit code based on success
    sys.exit(0 if metrics.total_edges_created >= 0 else 1)


if __name__ == "__main__":
    main()
