#!/usr/bin/env python3
"""
Category Golden Set Evaluation

Evaluates category-based retrieval:
- Runs query through hybrid search
- Checks if top-K results have the expected category assignment
- Calculates accuracy (category hit rate) and MRR

Targets:
- topic_only: Category Hit Rate >= 0.80 (at least one result has expected category)
- topic_semantic: Category Hit Rate >= 0.70

Usage:
    uv run python scripts/qa/run_category_eval.py
    uv run python scripts/qa/run_category_eval.py --top-k 10 --log-results
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
from src.lexe_api.kb.retrieval.router import route_query


@dataclass
class QueryResult:
    """Result for a single query."""
    query_id: str
    query_text: str
    query_class: str
    expected_category_id: str
    expected_level: int

    # Retrieval results
    top_k_ids: list[str]
    top_k_scores: list[float]
    top_k_categories: list[list[str]]  # Categories for each result

    # Metrics
    hit: bool  # At least one result has expected category
    rank: int | None  # Rank of first result with expected category
    mrr: float  # 1/rank or 0

    latency_ms: float


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    batch_id: int
    run_at: str
    top_k: int

    # Topic-only metrics
    topic_only_count: int
    topic_only_hit_rate: float
    topic_only_mrr: float

    # Topic-semantic metrics
    topic_semantic_count: int
    topic_semantic_hit_rate: float
    topic_semantic_mrr: float

    # Overall
    total_queries: int
    avg_latency_ms: float

    # Pass/Fail
    topic_only_pass: bool
    topic_semantic_pass: bool


async def get_massima_categories(conn: asyncpg.Connection, massima_id: UUID) -> list[str]:
    """Get all category IDs assigned to a massima."""
    rows = await conn.fetch("""
        SELECT category_id FROM kb.category_assignments
        WHERE massima_id = $1
    """, massima_id)
    return [r["category_id"] for r in rows]


async def evaluate_query(
    conn: asyncpg.Connection,
    query_id: str,
    query_text: str,
    query_class: str,
    expected_category_id: str,
    expected_level: int,
    top_k: int,
) -> QueryResult:
    """Evaluate a single query by running retrieval and checking categories."""
    start = time.perf_counter()

    # Run query through router (hybrid search)
    route_result = await route_query(query_text, conn, limit=top_k)

    latency_ms = (time.perf_counter() - start) * 1000

    top_k_ids = [str(mid) for mid in route_result.massima_ids]
    top_k_scores = route_result.scores

    # Get categories for each result
    top_k_categories = []
    for mid in route_result.massima_ids:
        cats = await get_massima_categories(conn, mid)
        top_k_categories.append(cats)

    # Check if expected category is in any result
    hit = False
    rank = None
    mrr = 0.0

    for i, cats in enumerate(top_k_categories):
        if expected_category_id in cats:
            hit = True
            rank = i + 1
            mrr = 1.0 / rank
            break

    return QueryResult(
        query_id=query_id,
        query_text=query_text,
        query_class=query_class,
        expected_category_id=expected_category_id,
        expected_level=expected_level,
        top_k_ids=top_k_ids,
        top_k_scores=top_k_scores,
        top_k_categories=top_k_categories,
        hit=hit,
        rank=rank,
        mrr=mrr,
        latency_ms=latency_ms,
    )


async def run_evaluation(top_k: int = 10, log_results: bool = False) -> EvalSummary:
    """Run full evaluation on golden set."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Fetch active golden queries
        rows = await conn.fetch("""
            SELECT id, batch_id, query_text, query_class, expected_category_id, expected_level
            FROM kb.golden_category_queries
            WHERE is_active = TRUE
            ORDER BY query_class, id
        """)

        if not rows:
            print("No active golden category queries found!")
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
                row["expected_category_id"],
                row["expected_level"],
                top_k,
            )
            results.append(result)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(rows)}...")

        # Calculate metrics
        topic_only = [r for r in results if r.query_class == "topic_only"]
        topic_semantic = [r for r in results if r.query_class == "topic_semantic"]

        # Topic-only metrics
        to_hit_rate = sum(1 for r in topic_only if r.hit) / len(topic_only) if topic_only else 0
        to_mrr = sum(r.mrr for r in topic_only) / len(topic_only) if topic_only else 0

        # Topic-semantic metrics
        ts_hit_rate = sum(1 for r in topic_semantic if r.hit) / len(topic_semantic) if topic_semantic else 0
        ts_mrr = sum(r.mrr for r in topic_semantic) / len(topic_semantic) if topic_semantic else 0

        # Overall
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        summary = EvalSummary(
            batch_id=batch_id,
            run_at=datetime.now().isoformat(),
            top_k=top_k,
            topic_only_count=len(topic_only),
            topic_only_hit_rate=round(to_hit_rate, 4),
            topic_only_mrr=round(to_mrr, 4),
            topic_semantic_count=len(topic_semantic),
            topic_semantic_hit_rate=round(ts_hit_rate, 4),
            topic_semantic_mrr=round(ts_mrr, 4),
            total_queries=len(results),
            avg_latency_ms=round(avg_latency, 2),
            topic_only_pass=(to_hit_rate >= 0.80),
            topic_semantic_pass=(ts_hit_rate >= 0.70),
        )

        # Print results
        print("\n" + "=" * 60)
        print("TOPIC-ONLY RESULTS")
        print("=" * 60)
        print(f"  Count:      {summary.topic_only_count}")
        print(f"  Hit Rate:   {summary.topic_only_hit_rate:.2%} (target >= 80%)")
        print(f"  MRR:        {summary.topic_only_mrr:.4f}")
        print(f"  PASS:       {'YES' if summary.topic_only_pass else 'NO'}")

        print("\n" + "=" * 60)
        print("TOPIC-SEMANTIC RESULTS")
        print("=" * 60)
        print(f"  Count:      {summary.topic_semantic_count}")
        print(f"  Hit Rate:   {summary.topic_semantic_hit_rate:.2%} (target >= 70%)")
        print(f"  MRR:        {summary.topic_semantic_mrr:.4f}")
        print(f"  PASS:       {'YES' if summary.topic_semantic_pass else 'NO'}")

        print("\n" + "=" * 60)
        print("OVERALL")
        print("=" * 60)
        print(f"  Total Queries: {summary.total_queries}")
        print(f"  Avg Latency:   {summary.avg_latency_ms:.1f}ms")

        # Log detailed results
        if log_results:
            log_dir = Path("scripts/qa/retrieval_logs/category_eval")
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
                json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
            print(f"Summary: {summary_path}")

            # Failures report
            failures = [r for r in results if not r.hit]
            if failures:
                print(f"\n{len(failures)} queries without category hit:")
                for r in failures[:10]:
                    cats = r.top_k_categories[0] if r.top_k_categories else []
                    first_cat = cats[0] if cats else "NONE"
                    print(f"  [{r.query_class}] {r.query_text[:35]:35} -> expected {r.expected_category_id}, got {first_cat}")

        # Store in DB
        await conn.execute("""
            INSERT INTO kb.category_eval_runs
                (batch_id, top_k, topic_only_count, topic_only_accuracy, topic_only_mrr,
                 topic_semantic_count, topic_semantic_accuracy, topic_semantic_mrr,
                 total_queries, avg_latency_ms, topic_only_pass, topic_semantic_pass)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """, batch_id, top_k, len(topic_only), to_hit_rate, to_mrr,
            len(topic_semantic), ts_hit_rate, ts_mrr,
            len(results), avg_latency, summary.topic_only_pass, summary.topic_semantic_pass)

        return summary

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Category Eval")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results")
    parser.add_argument("--log-results", action="store_true", help="Save detailed logs")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.top_k, args.log_results))
