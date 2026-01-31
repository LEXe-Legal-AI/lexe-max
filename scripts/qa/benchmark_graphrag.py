#!/usr/bin/env python3
"""
GraphRAG Benchmark Script

Confronta hybrid search vs hybrid+graph reranking.

Metriche:
- Recall@K
- MRR (Mean Reciprocal Rank)
- Graph hit rate
- Latency

Usage:
    uv run python scripts/qa/benchmark_graphrag.py
    uv run python scripts/qa/benchmark_graphrag.py --top-k 10 --samples 50
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

import asyncpg

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.retrieval.graph_reranker import (
    GraphRAGConfig,
    rerank_with_graph,
    calculate_graph_hit_rate,
    calculate_rank_change,
)


async def get_golden_queries(conn, limit: int = 100) -> list[dict]:
    """Fetch golden queries with expected massima_id."""
    rows = await conn.fetch("""
        SELECT
            gq.id,
            gq.query_text,
            gq.query_type,
            gq.expected_massima_id,
            m.rv,
            m.sezione,
            m.anno
        FROM kb.golden_queries gq
        JOIN kb.massime m ON gq.expected_massima_id = m.id
        WHERE gq.is_active = TRUE
        ORDER BY RANDOM()
        LIMIT $1
    """, limit)

    return [dict(row) for row in rows]


async def get_query_embedding(conn, expected_massima_id: UUID, model: str = "openai/text-embedding-3-small", noise: float = 0.0) -> str | None:
    """Get embedding of expected massima as query proxy (as string).

    Args:
        noise: Add gaussian noise with this std dev to simulate realistic query.
               0.0 = exact embedding, 0.1 = moderate noise, 0.2 = significant noise
    """
    import random

    row = await conn.fetchrow("""
        SELECT embedding::text as emb_str
        FROM kb.embeddings
        WHERE massima_id = $1 AND model_name = $2
    """, expected_massima_id, model)

    if not row:
        return None

    if noise == 0.0:
        return row["emb_str"]

    # Parse embedding, add noise, re-serialize
    emb_str = row["emb_str"]
    values = [float(x) for x in emb_str.strip("[]").split(",")]

    # Add gaussian noise
    noisy = [v + random.gauss(0, noise) for v in values]

    # Normalize (L2 norm)
    norm = sum(x**2 for x in noisy) ** 0.5
    noisy = [x / norm for x in noisy]

    return "[" + ",".join(str(x) for x in noisy) + "]"


async def simple_hybrid_search(conn, query_text: str, embedding_str: str | None, limit: int = 50, mode: str = "hybrid"):
    """Simplified hybrid search using direct SQL.

    Modes:
    - hybrid: Dense + BM25 RRF fusion
    - bm25_only: Only BM25 (text search)
    - dense_only: Only dense (vector search)
    """
    from dataclasses import dataclass

    @dataclass
    class SimpleResult:
        massima_id: UUID
        rrf_score: float
        final_rank: int
        dense_score: float | None = None
        bm25_score: float | None = None

    scores = {}
    dense_lookup = {}
    bm25_lookup = {}
    k = 60  # RRF constant

    # Dense search (if enabled and embedding available)
    if mode in ("hybrid", "dense_only") and embedding_str:
        dense_rows = await conn.fetch("""
            SELECT m.id as massima_id, 1 - (e.embedding <=> $1::vector) as score
            FROM kb.embeddings e
            JOIN kb.massime m ON e.massima_id = m.id
            WHERE e.model_name = 'openai/text-embedding-3-small'
              AND m.is_active = TRUE
            ORDER BY e.embedding <=> $1::vector
            LIMIT $2
        """, embedding_str, limit)

        for i, row in enumerate(dense_rows, 1):
            mid = row["massima_id"]
            scores[mid] = scores.get(mid, 0) + 1.0 / (k + i)
            dense_lookup[mid] = row["score"]

    # BM25 search (if enabled)
    if mode in ("hybrid", "bm25_only"):
        bm25_rows = await conn.fetch("""
            SELECT id as massima_id,
                   ts_rank_cd(tsv_italian, plainto_tsquery('italian', $1)) as score
            FROM kb.massime
            WHERE is_active = TRUE
              AND tsv_italian @@ plainto_tsquery('italian', $1)
            ORDER BY score DESC
            LIMIT $2
        """, query_text, limit)

        for i, row in enumerate(bm25_rows, 1):
            mid = row["massima_id"]
            scores[mid] = scores.get(mid, 0) + 1.0 / (k + i)
            bm25_lookup[mid] = row["score"]

    # Sort and build results
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (mid, rrf_score) in enumerate(sorted_items[:limit], 1):
        results.append(SimpleResult(
            massima_id=mid,
            rrf_score=rrf_score,
            final_rank=rank,
            dense_score=dense_lookup.get(mid),
            bm25_score=bm25_lookup.get(mid),
        ))

    return results


async def run_benchmark(
    conn,
    golden_queries: list[dict],
    top_k: int = 10,
    graph_config: GraphRAGConfig = None,
    search_mode: str = "hybrid",
    noise: float = 0.0,
) -> dict:
    """Run benchmark and return metrics."""

    results = {
        "hybrid": {"hits": 0, "mrr_sum": 0, "latencies": []},
        "hybrid_graph": {"hits": 0, "mrr_sum": 0, "latencies": [], "graph_hits": 0},
        "total": len(golden_queries),
    }

    for i, gq in enumerate(golden_queries):
        query_text = gq["query_text"]
        expected_id = gq["expected_massima_id"]

        # Get embedding of expected massima (as query proxy, optionally with noise)
        query_embedding = await get_query_embedding(conn, expected_id, noise=noise)
        if query_embedding is None:
            continue  # Skip if no embedding

        # --- Baseline Search ---
        t0 = time.perf_counter()
        hybrid_results = await simple_hybrid_search(conn, query_text, query_embedding, limit=50, mode=search_mode)
        t_hybrid = (time.perf_counter() - t0) * 1000

        results["hybrid"]["latencies"].append(t_hybrid)

        # Check hit
        hybrid_ids = [r.massima_id for r in hybrid_results[:top_k]]
        if expected_id in hybrid_ids:
            results["hybrid"]["hits"] += 1
            rank = hybrid_ids.index(expected_id) + 1
            results["hybrid"]["mrr_sum"] += 1.0 / rank

        # --- Hybrid + Graph ---
        if graph_config and hybrid_results:
            t0 = time.perf_counter()
            graph_results = await rerank_with_graph(hybrid_results, conn, graph_config)
            t_graph = (time.perf_counter() - t0) * 1000

            results["hybrid_graph"]["latencies"].append(t_hybrid + t_graph)

            graph_ids = [r.massima_id for r in graph_results[:top_k]]
            if expected_id in graph_ids:
                results["hybrid_graph"]["hits"] += 1
                rank = graph_ids.index(expected_id) + 1
                results["hybrid_graph"]["mrr_sum"] += 1.0 / rank

            # Graph hit rate
            graph_hit_rate = calculate_graph_hit_rate(graph_results, top_k)
            results["hybrid_graph"]["graph_hits"] += graph_hit_rate

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(golden_queries)} queries...")

    # Compute final metrics
    n = results["total"]

    def safe_p95(latencies):
        if not latencies:
            return 0
        idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        return sorted(latencies)[idx]

    results["hybrid"]["recall"] = results["hybrid"]["hits"] / n if n else 0
    results["hybrid"]["mrr"] = results["hybrid"]["mrr_sum"] / n if n else 0
    results["hybrid"]["latency_avg"] = sum(results["hybrid"]["latencies"]) / len(results["hybrid"]["latencies"]) if results["hybrid"]["latencies"] else 0
    results["hybrid"]["latency_p95"] = safe_p95(results["hybrid"]["latencies"])

    if graph_config:
        hg_lats = results["hybrid_graph"]["latencies"]
        results["hybrid_graph"]["recall"] = results["hybrid_graph"]["hits"] / n if n else 0
        results["hybrid_graph"]["mrr"] = results["hybrid_graph"]["mrr_sum"] / n if n else 0
        results["hybrid_graph"]["latency_avg"] = sum(hg_lats) / len(hg_lats) if hg_lats else 0
        results["hybrid_graph"]["latency_p95"] = safe_p95(hg_lats)
        results["hybrid_graph"]["graph_hit_rate"] = results["hybrid_graph"]["graph_hits"] / n if n else 0

    return results


async def main(args):
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    print("=" * 60)
    print("LEXE KB - GraphRAG Benchmark")
    print("=" * 60)

    # Get golden queries
    print(f"\nLoading {args.samples} golden queries...")
    golden_queries = await get_golden_queries(conn, args.samples)
    print(f"  Loaded {len(golden_queries)} queries")

    # Configure GraphRAG
    graph_config = GraphRAGConfig(
        seed_count=args.seeds,
        expansion_depth=args.depth,
        min_edge_weight=args.min_weight,
        graph_boost_factor=args.boost,
    )

    print(f"\nGraphRAG Config:")
    print(f"  Seeds: {graph_config.seed_count}")
    print(f"  Depth: {graph_config.expansion_depth}")
    print(f"  Min weight: {graph_config.min_edge_weight}")
    print(f"  Boost factor: {graph_config.graph_boost_factor}")

    # Run benchmark
    print(f"\nRunning benchmark (top-k={args.top_k}, mode={args.mode}, noise={args.noise})...")
    results = await run_benchmark(conn, golden_queries, args.top_k, graph_config, args.mode, args.noise)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Hybrid':>12} {'Hybrid+Graph':>12} {'Delta':>10}")
    print("-" * 60)

    h = results["hybrid"]
    hg = results["hybrid_graph"]

    recall_delta = (hg["recall"] - h["recall"]) * 100
    mrr_delta = hg["mrr"] - h["mrr"]
    lat_delta = hg["latency_avg"] - h["latency_avg"]

    print(f"{'Recall@' + str(args.top_k):<25} {h['recall']*100:>11.1f}% {hg['recall']*100:>11.1f}% {recall_delta:>+9.1f}%")
    print(f"{'MRR':<25} {h['mrr']:>12.3f} {hg['mrr']:>12.3f} {mrr_delta:>+10.3f}")
    print(f"{'Latency avg (ms)':<25} {h['latency_avg']:>12.1f} {hg['latency_avg']:>12.1f} {lat_delta:>+10.1f}")
    print(f"{'Latency p95 (ms)':<25} {h['latency_p95']:>12.1f} {hg['latency_p95']:>12.1f} {hg['latency_p95']-h['latency_p95']:>+10.1f}")
    print(f"{'Graph hit rate':<25} {'-':>12} {hg['graph_hit_rate']*100:>11.1f}%")

    print("\n" + "-" * 60)

    # Verdict
    if mrr_delta > 0:
        print(f"GraphRAG IMPROVED MRR by {mrr_delta:.3f} (+{mrr_delta/h['mrr']*100:.1f}%)")
    else:
        print(f"GraphRAG did not improve MRR ({mrr_delta:.3f})")

    if recall_delta >= 0:
        print(f"Recall maintained/improved ({recall_delta:+.1f}%)")
    else:
        print(f"WARNING: Recall decreased by {-recall_delta:.1f}%")

    # Save results
    if args.save:
        output_file = Path(f"scripts/qa/graphrag_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "top_k": args.top_k,
                    "samples": args.samples,
                    "seeds": args.seeds,
                    "depth": args.depth,
                    "min_weight": args.min_weight,
                    "boost": args.boost,
                },
                "results": {
                    "hybrid": {k: v for k, v in h.items() if k != "latencies"},
                    "hybrid_graph": {k: v for k, v in hg.items() if k != "latencies"},
                },
            }, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of golden queries to test")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for graph expansion")
    parser.add_argument("--depth", type=int, default=2, help="Graph expansion depth")
    parser.add_argument("--min-weight", type=float, default=0.5, help="Minimum edge weight")
    parser.add_argument("--boost", type=float, default=0.15, help="Graph boost factor")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "bm25_only", "dense_only"],
                       help="Search mode: hybrid (default), bm25_only, or dense_only")
    parser.add_argument("--noise", type=float, default=0.0,
                       help="Add noise to embeddings (0.0=exact, 0.1=moderate, 0.2=significant)")

    args = parser.parse_args()
    asyncio.run(main(args))
