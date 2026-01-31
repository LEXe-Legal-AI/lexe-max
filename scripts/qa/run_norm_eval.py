#!/usr/bin/env python3
"""
Norm Golden Set Evaluation

Evaluates norm lookup and mixed query performance:
- Norm Recall@10: At least 1 result in top-10 cites expected norm
- Norm MRR: Position of first correct result
- Router Accuracy: % of pure_norm queries routed to RouteType.NORM

Targets:
- pure_norm: Recall@10 >= 0.98, MRR >= 0.90, Router Accuracy >= 0.98
- mixed: Norm Hit Rate >= 0.70, Norm MRR >= 0.70

Usage:
    uv run python scripts/qa/run_norm_eval.py
    uv run python scripts/qa/run_norm_eval.py --top-k 10 --log-results
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import asyncpg


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.retrieval.router import (
    RouteType,
    classify_query,
    route_query,
)


@dataclass
class QueryResult:
    """Result for a single query."""
    query_id: str
    query_text: str
    query_class: str
    expected_norm_id: str

    # Routing
    route_type: str
    router_correct: bool  # True if pure_norm -> NORM

    # Results
    top_k_ids: list[str]
    top_k_scores: list[float]
    top_k_cites_norm: list[bool]

    # Metrics
    norm_hit: bool  # Any result cites expected norm
    norm_rank: int | None  # Rank of first hit (1-indexed), None if no hit
    mrr: float  # 1/rank or 0

    latency_ms: float


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    batch_id: int
    run_at: str
    top_k: int

    # Pure norm
    pure_norm_count: int
    pure_norm_recall_at_k: float
    pure_norm_mrr: float
    pure_norm_router_accuracy: float

    # Mixed
    mixed_count: int
    mixed_norm_hit_rate: float
    mixed_norm_mrr: float

    # Overall
    total_queries: int
    avg_latency_ms: float

    # Pass/Fail
    pure_norm_pass: bool
    mixed_pass: bool


async def check_cites_norm(
    conn: asyncpg.Connection,
    massima_id: UUID,
    norm_id: str,
) -> bool:
    """Check if a massima cites a specific norm."""
    row = await conn.fetchrow(
        """
        SELECT 1 FROM kb.massima_norms
        WHERE massima_id = $1 AND norm_id = $2
        LIMIT 1
        """,
        massima_id,
        norm_id,
    )
    return row is not None


async def evaluate_query(
    conn: asyncpg.Connection,
    query_id: str,
    query_text: str,
    query_class: str,
    expected_norm_id: str,
    top_k: int,
) -> QueryResult:
    """Evaluate a single query."""
    start = time.perf_counter()

    # Route query
    route_type, citation, norm = classify_query(query_text)
    router_correct = (query_class == "pure_norm" and route_type == RouteType.NORM)

    # Get results
    route_result = await route_query(query_text, conn, limit=top_k)

    latency_ms = (time.perf_counter() - start) * 1000

    # Check which results cite the expected norm
    top_k_ids = [str(mid) for mid in route_result.massima_ids]
    top_k_scores = route_result.scores
    top_k_cites_norm = []

    for mid in route_result.massima_ids:
        cites = await check_cites_norm(conn, mid, expected_norm_id)
        top_k_cites_norm.append(cites)

    # Calculate metrics
    norm_hit = any(top_k_cites_norm)
    norm_rank = None
    mrr = 0.0

    for i, cites in enumerate(top_k_cites_norm):
        if cites:
            norm_rank = i + 1
            mrr = 1.0 / norm_rank
            break

    return QueryResult(
        query_id=query_id,
        query_text=query_text,
        query_class=query_class,
        expected_norm_id=expected_norm_id,
        route_type=route_type.value,
        router_correct=router_correct,
        top_k_ids=top_k_ids,
        top_k_scores=top_k_scores,
        top_k_cites_norm=top_k_cites_norm,
        norm_hit=norm_hit,
        norm_rank=norm_rank,
        mrr=mrr,
        latency_ms=latency_ms,
    )


async def run_evaluation(
    top_k: int = 10,
    log_results: bool = False,
) -> EvalSummary:
    """Run full evaluation on golden set."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    # Fetch active golden queries
    rows = await conn.fetch("""
        SELECT id, batch_id, query_text, query_class, expected_norm_id
        FROM kb.golden_norm_queries
        WHERE is_active = TRUE
        ORDER BY query_class, id
    """)

    if not rows:
        print("No active golden queries found!")
        await conn.close()
        return None

    batch_id = rows[0]["batch_id"]
    print(f"Evaluating batch_id={batch_id}, {len(rows)} queries, top_k={top_k}")
    print("=" * 60)

    results: list[QueryResult] = []

    for i, row in enumerate(rows):
        result = await evaluate_query(
            conn,
            str(row["id"]),
            row["query_text"],
            row["query_class"],
            row["expected_norm_id"],
            top_k,
        )
        results.append(result)

        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rows)}...")

    await conn.close()

    # Calculate metrics
    pure_norm = [r for r in results if r.query_class == "pure_norm"]
    mixed = [r for r in results if r.query_class == "mixed"]

    # Pure norm metrics
    pure_recall = sum(1 for r in pure_norm if r.norm_hit) / len(pure_norm) if pure_norm else 0
    pure_mrr = sum(r.mrr for r in pure_norm) / len(pure_norm) if pure_norm else 0
    pure_router_acc = sum(1 for r in pure_norm if r.router_correct) / len(pure_norm) if pure_norm else 0

    # Mixed metrics
    mixed_hit_rate = sum(1 for r in mixed if r.norm_hit) / len(mixed) if mixed else 0
    mixed_mrr = sum(r.mrr for r in mixed) / len(mixed) if mixed else 0

    # Overall
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    summary = EvalSummary(
        batch_id=batch_id,
        run_at=datetime.now().isoformat(),
        top_k=top_k,
        pure_norm_count=len(pure_norm),
        pure_norm_recall_at_k=round(pure_recall, 4),
        pure_norm_mrr=round(pure_mrr, 4),
        pure_norm_router_accuracy=round(pure_router_acc, 4),
        mixed_count=len(mixed),
        mixed_norm_hit_rate=round(mixed_hit_rate, 4),
        mixed_norm_mrr=round(mixed_mrr, 4),
        total_queries=len(results),
        avg_latency_ms=round(avg_latency, 2),
        pure_norm_pass=(pure_recall >= 0.98 and pure_mrr >= 0.90 and pure_router_acc >= 0.98),
        mixed_pass=(mixed_hit_rate >= 0.70),
    )

    # Print results
    print("\n" + "=" * 60)
    print("PURE NORM RESULTS")
    print("=" * 60)
    print(f"  Count:           {summary.pure_norm_count}")
    print(f"  Recall@{top_k}:       {summary.pure_norm_recall_at_k:.2%} (target >= 98%)")
    print(f"  MRR:             {summary.pure_norm_mrr:.4f} (target >= 0.90)")
    print(f"  Router Accuracy: {summary.pure_norm_router_accuracy:.2%} (target >= 98%)")
    print(f"  PASS:            {'YES' if summary.pure_norm_pass else 'NO'}")

    print("\n" + "=" * 60)
    print("MIXED RESULTS")
    print("=" * 60)
    print(f"  Count:           {summary.mixed_count}")
    print(f"  Norm Hit Rate:   {summary.mixed_norm_hit_rate:.2%} (target >= 70%)")
    print(f"  Norm MRR:        {summary.mixed_norm_mrr:.4f}")
    print(f"  PASS:            {'YES' if summary.mixed_pass else 'NO'}")

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"  Total Queries:   {summary.total_queries}")
    print(f"  Avg Latency:     {summary.avg_latency_ms:.1f}ms")

    # Log detailed results
    if log_results:
        log_dir = Path("scripts/qa/retrieval_logs/norm_eval")
        log_dir.mkdir(parents=True, exist_ok=True)

        # JSONL with all results
        jsonl_path = log_dir / f"batch_{batch_id}_k{top_k}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False, cls=DecimalEncoder) + "\n")
        print(f"\nDetailed results: {jsonl_path}")

        # Summary JSON
        summary_path = log_dir / f"batch_{batch_id}_k{top_k}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
        print(f"Summary: {summary_path}")

        # Failures report
        failures = [r for r in results if not r.norm_hit]
        if failures:
            print(f"\n{len(failures)} queries with no norm hit:")
            for r in failures[:10]:
                print(f"  [{r.query_class}] {r.query_text[:50]}... -> {r.expected_norm_id}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Norm Eval")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results")
    parser.add_argument("--log-results", action="store_true", help="Save detailed logs")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.top_k, args.log_results))
