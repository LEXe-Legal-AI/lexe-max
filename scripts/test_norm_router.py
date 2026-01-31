#!/usr/bin/env python3
"""
Test Norm Router Detection and Lookup

Quick validation that norm queries are properly detected and routed.

Usage:
    uv run python scripts/test_norm_router.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.retrieval.router import (
    RouteType,
    LookupResult,
    classify_query,
    route_query,
)


# Test queries for classification
TEST_QUERIES = [
    # Citation queries (should NOT be norm)
    ("Rv. 639966", RouteType.CITATION_RV),
    ("Sez. Un., n. 12345/2020", RouteType.CITATION_SEZ_NUM_ANNO),
    ("n. 12345/2020", RouteType.CITATION_NUM_ANNO),

    # Norm queries (should be NORM)
    ("art. 2043 c.c.", RouteType.NORM),
    ("art. 360 c.p.c.", RouteType.NORM),
    ("art. 640 c.p.", RouteType.NORM),
    ("art. 384 c.p.p.", RouteType.NORM),
    ("art. 111 Cost.", RouteType.NORM),
    ("legge 241/1990", RouteType.NORM),
    ("l. 241/1990", RouteType.NORM),
    ("d.lgs. 50/2016", RouteType.NORM),
    ("d.lgs. 165/2001", RouteType.NORM),
    ("d.p.r. 445/2000", RouteType.NORM),

    # Semantic queries (no special pattern)
    ("responsabilità extracontrattuale", RouteType.SEMANTIC),
    ("danno ingiusto risarcimento", RouteType.SEMANTIC),
]


def test_classification():
    """Test query classification."""
    print("\n" + "=" * 60)
    print("QUERY CLASSIFICATION TEST")
    print("=" * 60)

    passed = 0
    failed = 0

    for query, expected in TEST_QUERIES:
        route_type, citation, norm = classify_query(query)
        status = "OK" if route_type == expected else "FAIL"

        if route_type == expected:
            passed += 1
        else:
            failed += 1

        extra = ""
        if norm:
            extra = f" -> {norm.canonical_id}"
        elif citation:
            extra = f" -> RV:{citation.rv}" if citation.rv else f" -> {citation.numero}/{citation.anno}"

        print(f"  {status} '{query}' -> {route_type.value}{extra}")
        if route_type != expected:
            print(f"      Expected: {expected.value}")

    print(f"\nResults: {passed}/{passed + failed} passed")
    return failed == 0


async def test_norm_lookup():
    """Test norm lookup with database."""
    print("\n" + "=" * 60)
    print("NORM LOOKUP TEST")
    print("=" * 60)

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    # Test queries with known norms
    norm_queries = [
        "art. 2043 c.c.",      # responsabilità extracontrattuale
        "art. 360 c.p.c.",     # ricorso cassazione
        "art. 111 Cost.",      # giusto processo
        "d.lgs. 165/2001",     # pubblico impiego
        "legge 241/1990",      # procedimento amministrativo
    ]

    for query in norm_queries:
        result = await route_query(query, conn, limit=5)

        print(f"\n  Query: '{query}'")
        print(f"    Route: {result.route_type.value}")

        if result.norm:
            print(f"    Norm ID: {result.norm.canonical_id}")

        print(f"    Lookup attempted: {result.lookup_attempted}")
        print(f"    Lookup hit: {result.lookup_hit}")

        if result.lookup_hit:
            print(f"    Result type: {result.lookup_result.value}")
            print(f"    Massime found: {len(result.massima_ids)}")

            # Show first 3 results
            for i, (mid, score) in enumerate(zip(result.massima_ids[:3], result.scores[:3])):
                print(f"      {i+1}. {mid} (score: {score:.3f})")

    await conn.close()
    return True


async def test_stats():
    """Show norm lookup stats."""
    print("\n" + "=" * 60)
    print("NORM GRAPH STATS")
    print("=" * 60)

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    # Top cited norms
    rows = await conn.fetch("""
        SELECT id, full_ref, citation_count
        FROM kb.norms
        ORDER BY citation_count DESC
        LIMIT 10
    """)

    print("\n  Top 10 cited norms:")
    for i, row in enumerate(rows, 1):
        print(f"    {i:2}. {row['full_ref']:30} ({row['citation_count']} citations)")

    # Stats by code
    rows = await conn.fetch("""
        SELECT * FROM kb.norm_stats
    """)

    print("\n  Stats by code:")
    for row in rows:
        print(f"    {row['code']:8} {row['norm_count']:5} norms, {row['total_citations']:6} citations")

    await conn.close()


async def main():
    # Test classification (no DB needed)
    classification_ok = test_classification()

    # Test lookup (needs DB)
    try:
        await test_norm_lookup()
        await test_stats()
    except Exception as e:
        print(f"\n  [!] DB test skipped: {e}")

    print("\n" + "=" * 60)
    if classification_ok:
        print("All classification tests PASSED")
    else:
        print("Some classification tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
