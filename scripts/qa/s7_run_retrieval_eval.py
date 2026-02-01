"""
QA Protocol - Phase 7: Run Retrieval Evaluation

Runs evaluation queries against 3 methods: R1_hybrid, dense_only, sparse_only.
Computes R@5, R@10, MRR, nDCG@10, noise_rate_at_10.

Uses Mistral Embed via OpenRouter for dense search.

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    export OPENROUTER_API_KEY='sk-or-...'
    uv run python scripts/qa/s7_run_retrieval_eval.py
"""

import asyncio
import math
import os
import time

import asyncpg
import httpx

from qa_config import DB_URL
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_MODEL = "mistralai/mistral-embed-2312"
RRF_K = 60


async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float] | None:
    """Get embedding from OpenRouter."""
    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MISTRAL_MODEL, "input": [text]},
            timeout=30.0,
        )
        if response.status_code != 200:
            return None
        return response.json()["data"][0]["embedding"]
    except Exception:
        return None


def emb_to_pg(embedding: list[float]) -> str:
    """Convert embedding to pgvector format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def dense_search(conn, embedding: list[float], limit: int = 10):
    """Dense-only search via pgvector."""
    rows = await conn.fetch(
        """
        SELECT e.massima_id, 1 - (e.embedding <=> $1::vector) as score
        FROM kb.emb_mistral e
        ORDER BY e.embedding <=> $1::vector
        LIMIT $2
        """,
        emb_to_pg(embedding), limit,
    )
    return [(r["massima_id"], float(r["score"])) for r in rows]


async def sparse_search(conn, query_text: str, limit: int = 10):
    """Sparse-only search via tsvector."""
    rows = await conn.fetch(
        """
        SELECT m.id as massima_id,
               ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', $1)) as score
        FROM kb.massime m
        WHERE m.tsv_italian @@ plainto_tsquery('italian', $1)
           OR m.tsv_simple @@ plainto_tsquery('simple', $1)
        ORDER BY score DESC
        LIMIT $2
        """,
        query_text, limit,
    )
    return [(r["massima_id"], float(r["score"])) for r in rows]


async def hybrid_search(conn, query_text: str, embedding: list[float], limit: int = 10):
    """R1 Hybrid (BM25 + Dense + RRF)."""
    rows = await conn.fetch(
        """
        WITH dense_results AS (
            SELECT e.massima_id,
                   ROW_NUMBER() OVER (ORDER BY e.embedding <=> $1::vector) as rank
            FROM kb.emb_mistral e
            LIMIT 50
        ),
        sparse_results AS (
            SELECT m.id as massima_id,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', $2)) DESC
                   ) as rank
            FROM kb.massime m
            WHERE m.tsv_italian @@ plainto_tsquery('italian', $2)
               OR m.tsv_simple @@ plainto_tsquery('simple', $2)
            LIMIT 50
        ),
        combined AS (
            SELECT COALESCE(d.massima_id, s.massima_id) as massima_id,
                   COALESCE(1.0 / ($3 + d.rank), 0) +
                   COALESCE(1.0 / ($3 + s.rank), 0) as rrf_score
            FROM dense_results d
            FULL OUTER JOIN sparse_results s ON d.massima_id = s.massima_id
        )
        SELECT massima_id, rrf_score as score
        FROM combined
        ORDER BY rrf_score DESC
        LIMIT $4
        """,
        emb_to_pg(embedding), query_text, RRF_K, limit,
    )
    return [(r["massima_id"], float(r["score"])) for r in rows]


def compute_recall(result_ids, ground_truth_ids, k):
    """Recall@K."""
    if not ground_truth_ids:
        return 0.0
    top_k = set(result_ids[:k])
    gt = set(ground_truth_ids)
    return len(top_k & gt) / len(gt)


def compute_mrr(result_ids, ground_truth_ids):
    """Mean Reciprocal Rank."""
    if not ground_truth_ids:
        return 0.0
    gt = set(ground_truth_ids)
    for i, rid in enumerate(result_ids):
        if rid in gt:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(result_ids, ground_truth_ids, k=10):
    """nDCG@K."""
    if not ground_truth_ids:
        return 0.0
    gt = set(ground_truth_ids)
    dcg = 0.0
    for i, rid in enumerate(result_ids[:k]):
        if rid in gt:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / idcg if idcg > 0 else 0.0


async def compute_noise_rate(conn, result_ids, k=10):
    """Noise rate at K: % of top-K labeled toc/citation_list/noise."""
    top_k = result_ids[:k]
    if not top_k:
        return 0.0

    noise_count = 0
    for mid in top_k:
        label = await conn.fetchval(
            """
            SELECT cl.final_label
            FROM kb.chunk_labels cl
            JOIN kb.chunk_features cf ON cf.id = cl.chunk_feature_id
            WHERE cf.massima_id = $1
            LIMIT 1
            """,
            mid,
        )
        if label in ("toc", "citation_list", "noise"):
            noise_count += 1

    return noise_count / len(top_k)


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 7: RUN RETRIEVAL EVAL")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("[ERROR] Set OPENROUTER_API_KEY")
        return

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )
    print(f"[OK] qa_run_id={qa_run_id}, batch_id={batch_id}")

    queries = await conn.fetch(
        """
        SELECT id, query_text, query_type, source_massima_id, ground_truth_ids, keywords
        FROM kb.retrieval_eval_queries
        WHERE qa_run_id = $1
        """,
        qa_run_id,
    )
    print(f"[OK] Found {len(queries)} queries")

    methods = ["R1_hybrid", "dense_only", "sparse_only"]
    method_results = {m: [] for m in methods}

    async with httpx.AsyncClient() as http_client:
        for i, q in enumerate(queries):
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(queries)}] Processing...")

            query_text = q["query_text"]
            gt_ids = q["ground_truth_ids"] or []
            query_id = q["id"]

            # Get embedding
            embedding = await get_embedding(http_client, query_text)
            if not embedding:
                continue

            for method in methods:
                start = time.time()

                if method == "R1_hybrid":
                    results = await hybrid_search(conn, query_text, embedding, 10)
                elif method == "dense_only":
                    results = await dense_search(conn, embedding, 10)
                else:
                    results = await sparse_search(conn, query_text, 10)

                latency = int((time.time() - start) * 1000)
                result_ids = [r[0] for r in results]
                result_scores = [r[1] for r in results]

                r5 = compute_recall(result_ids, gt_ids, 5)
                r10 = compute_recall(result_ids, gt_ids, 10)
                mrr = compute_mrr(result_ids, gt_ids)
                ndcg = compute_ndcg(result_ids, gt_ids, 10)
                noise_rate = await compute_noise_rate(conn, result_ids, 10)

                await conn.execute(
                    """
                    INSERT INTO kb.retrieval_eval_results
                      (qa_run_id, query_id, ingest_batch_id, method,
                       recall_at_5, recall_at_10, mrr, ndcg_at_10, noise_rate_at_10,
                       result_ids, result_scores, latency_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    qa_run_id, query_id, batch_id, method,
                    round(r5, 4), round(r10, 4), round(mrr, 4), round(ndcg, 4),
                    round(noise_rate, 4),
                    [str(r) for r in result_ids],
                    result_scores,
                    latency,
                )

                method_results[method].append({
                    "r5": r5, "r10": r10, "mrr": mrr, "ndcg": ndcg,
                    "noise": noise_rate, "latency": latency,
                    "query_type": q["query_type"],
                })

            # Rate limit
            await asyncio.sleep(0.3)

    # Compute and store summaries
    for method in methods:
        results = method_results[method]
        if not results:
            continue

        # Group by query type
        by_type = {}
        for r in results:
            qt = r["query_type"]
            if qt not in by_type:
                by_type[qt] = []
            by_type[qt].append(r)

        for qt, type_results in by_type.items():
            n = len(type_results)
            await conn.execute(
                """
                INSERT INTO kb.retrieval_eval_summary
                  (qa_run_id, ingest_batch_id, method, query_type, query_count,
                   avg_recall_5, avg_recall_10, avg_mrr, avg_ndcg_10,
                   avg_noise_rate_10, avg_latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                qa_run_id, batch_id, method, qt, n,
                round(sum(r["r5"] for r in type_results) / n, 4),
                round(sum(r["r10"] for r in type_results) / n, 4),
                round(sum(r["mrr"] for r in type_results) / n, 4),
                round(sum(r["ndcg"] for r in type_results) / n, 4),
                round(sum(r["noise"] for r in type_results) / n, 4),
                round(sum(r["latency"] for r in type_results) / n, 1),
            )

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL EVAL COMPLETE")
    print(f"{'=' * 70}")

    for method in methods:
        results = method_results[method]
        if not results:
            continue
        n = len(results)
        print(f"\n{method} ({n} queries):")
        print(f"  R@5:  {sum(r['r5'] for r in results)/n:.3f}")
        print(f"  R@10: {sum(r['r10'] for r in results)/n:.3f}")
        print(f"  MRR:  {sum(r['mrr'] for r in results)/n:.3f}")
        print(f"  nDCG@10: {sum(r['ndcg'] for r in results)/n:.3f}")
        print(f"  Noise@10: {sum(r['noise'] for r in results)/n:.3f}")
        print(f"  Latency: {sum(r['latency'] for r in results)/n:.0f}ms")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
