"""
QA Protocol - Embedding Benchmark

Compares retrieval performance of:
- mistralai/mistral-embed-2312 (existing embeddings)
- google/gemini-embedding-001 (new embeddings)

Both at 1024 dimensions.

Steps:
1. Create emb_gemini table if not exists
2. Embed all massime with Gemini
3. Run retrieval on both sets
4. Compare R@5, R@10, MRR, nDCG

Usage:
    cd /opt/lexe-platform/lexe-max
    export OPENROUTER_API_KEY='sk-or-...'
    uv run python scripts/qa/benchmark_embeddings.py
"""

import asyncio
import os
import time
from uuid import uuid4

import asyncpg
import httpx

from qa_config import (
    DB_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_EMBED_URL,
    EMBED_MISTRAL,
    EMBED_GEMINI,
    EMBED_DIM,
)

# Use env var if config is empty
API_KEY = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")


async def create_gemini_table(conn):
    """Create emb_gemini table if not exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS kb.emb_gemini (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            massima_id UUID NOT NULL REFERENCES kb.massime(id),
            chunk_idx SMALLINT NOT NULL DEFAULT 0,
            embedding vector(1024) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            ingest_batch_id BIGINT REFERENCES kb.ingest_batches(id)
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_emb_gemini_massima
        ON kb.emb_gemini(massima_id)
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_emb_gemini_hnsw
        ON kb.emb_gemini USING hnsw (embedding vector_cosine_ops)
    """)
    print("[OK] emb_gemini table ready")


async def get_embedding(client: httpx.AsyncClient, text: str, model: str) -> list[float] | None:
    """Get embedding from OpenRouter."""
    try:
        # Truncate text if too long (Gemini has 2048 token limit for embedding)
        text = text[:8000]  # ~2000 tokens

        # Build request payload - Mistral has fixed 1024 dims, Gemini accepts dimensions param
        payload = {
            "model": model,
            "input": [text],
        }
        # Only add dimensions for models that support it (Gemini)
        if "gemini" in model.lower():
            payload["dimensions"] = EMBED_DIM

        response = await client.post(
            OPENROUTER_EMBED_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )

        # Check for error in response body (OpenRouter returns 200 with error object)
        data = response.json()
        if "error" in data:
            print(f"  [WARN] API error: {data['error'].get('message', 'Unknown')[:100]}")
            return None

        if response.status_code != 200:
            print(f"  [WARN] API error {response.status_code}: {response.text[:200]}")
            return None

        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def emb_to_pg(embedding: list[float]) -> str:
    """Convert embedding to pgvector format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def embed_massime_gemini(conn, client: httpx.AsyncClient, batch_id: int):
    """Embed all massime with Gemini."""
    # Get massime that don't have Gemini embedding yet
    massime = await conn.fetch("""
        SELECT m.id, m.testo
        FROM kb.massime m
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.emb_gemini e WHERE e.massima_id = m.id
        )
        ORDER BY m.id
    """)

    print(f"[INFO] {len(massime)} massime to embed with Gemini")

    embedded = 0
    errors = 0

    for i, m in enumerate(massime):
        embedding = await get_embedding(client, m["testo"], EMBED_GEMINI)

        if embedding and len(embedding) == EMBED_DIM:
            await conn.execute(
                """
                INSERT INTO kb.emb_gemini (massima_id, embedding, ingest_batch_id)
                VALUES ($1, $2::vector, $3)
                """,
                m["id"],
                emb_to_pg(embedding),
                batch_id,
            )
            embedded += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(massime)} ({embedded} embedded, {errors} errors)")

        # Rate limit: ~10 req/sec
        await asyncio.sleep(0.1)

    print(f"[OK] Gemini embeddings: {embedded} created, {errors} errors")
    return embedded


async def run_retrieval_test(conn, client: httpx.AsyncClient, emb_table: str, embed_model: str):
    """Run retrieval test on a set of queries."""
    # Get evaluation queries
    queries = await conn.fetch("""
        SELECT id, query_text, query_type, source_massima_id, ground_truth_ids
        FROM kb.retrieval_eval_queries
        ORDER BY id
    """)

    if not queries:
        print("[WARN] No queries found")
        return {}

    print(f"[INFO] Testing {emb_table} with {len(queries)} queries")

    results = {
        "hits_at_5": 0,
        "hits_at_10": 0,
        "reciprocal_ranks": [],
        "total": 0,
        "queries_with_expected": 0,
    }

    for i, q in enumerate(queries):
        # Get query embedding
        query_emb = await get_embedding(client, q["query_text"], embed_model)
        if not query_emb:
            continue

        # Search
        search_results = await conn.fetch(
            f"""
            SELECT massima_id, 1 - (embedding <=> $1::vector) as score
            FROM kb.{emb_table}
            ORDER BY embedding <=> $1::vector
            LIMIT 10
            """,
            emb_to_pg(query_emb),
        )

        results["total"] += 1

        # Ground truth: use ground_truth_ids array, fallback to source_massima_id
        expected_ids = q["ground_truth_ids"] or ([q["source_massima_id"]] if q["source_massima_id"] else [])

        if expected_ids:
            results["queries_with_expected"] += 1
            result_ids = [r["massima_id"] for r in search_results]

            # Check if ANY expected result is in top-k
            hit_5 = any(eid in result_ids[:5] for eid in expected_ids)
            hit_10 = any(eid in result_ids[:10] for eid in expected_ids)

            if hit_5:
                results["hits_at_5"] += 1
            if hit_10:
                results["hits_at_10"] += 1

            # Reciprocal rank (best rank among expected)
            best_rank = None
            for eid in expected_ids:
                if eid in result_ids:
                    rank = result_ids.index(eid) + 1
                    if best_rank is None or rank < best_rank:
                        best_rank = rank

            if best_rank:
                results["reciprocal_ranks"].append(1.0 / best_rank)
            else:
                results["reciprocal_ranks"].append(0.0)

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(queries)}")

        await asyncio.sleep(0.1)

    return results


def compute_metrics(results: dict) -> dict:
    """Compute final metrics."""
    if results["queries_with_expected"] == 0:
        return {"recall_5": 0, "recall_10": 0, "mrr": 0}

    return {
        "recall_5": results["hits_at_5"] / results["queries_with_expected"],
        "recall_10": results["hits_at_10"] / results["queries_with_expected"],
        "mrr": sum(results["reciprocal_ranks"]) / len(results["reciprocal_ranks"]) if results["reciprocal_ranks"] else 0,
        "total_queries": results["total"],
        "queries_with_expected": results["queries_with_expected"],
    }


async def main():
    print("=" * 70)
    print("QA PROTOCOL - EMBEDDING BENCHMARK")
    print(f"Models: {EMBED_MISTRAL} vs {EMBED_GEMINI}")
    print(f"Dimension: {EMBED_DIM}")
    print("=" * 70)

    if not API_KEY:
        print("[ERROR] OPENROUTER_API_KEY not set")
        return

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Create Gemini table
    await create_gemini_table(conn)

    # Get batch ID
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )
    print(f"[OK] batch_id={batch_id}")

    # Check current embedding counts
    mistral_count = await conn.fetchval("SELECT count(*) FROM kb.emb_mistral")
    gemini_count = await conn.fetchval("SELECT count(*) FROM kb.emb_gemini")
    print(f"[INFO] Existing embeddings: Mistral={mistral_count}, Gemini={gemini_count}")

    async with httpx.AsyncClient() as client:
        # Embed with Gemini if needed
        if gemini_count < mistral_count:
            print("\n--- STEP 1: Create Gemini Embeddings ---")
            await embed_massime_gemini(conn, client, batch_id)
            gemini_count = await conn.fetchval("SELECT count(*) FROM kb.emb_gemini")

        # Run retrieval tests
        print("\n--- STEP 2: Retrieval Benchmark ---")

        print("\n[Mistral]")
        mistral_results = await run_retrieval_test(conn, client, "emb_mistral", EMBED_MISTRAL)
        mistral_metrics = compute_metrics(mistral_results)

        print("\n[Gemini]")
        gemini_results = await run_retrieval_test(conn, client, "emb_gemini", EMBED_GEMINI)
        gemini_metrics = compute_metrics(gemini_results)

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Mistral':>15} {'Gemini':>15} {'Delta':>15}")
    print("-" * 70)

    for metric in ["recall_5", "recall_10", "mrr"]:
        m_val = mistral_metrics.get(metric, 0)
        g_val = gemini_metrics.get(metric, 0)
        delta = g_val - m_val
        delta_str = f"{delta:+.3f}" if delta != 0 else "="
        print(f"{metric:<20} {m_val:>15.3f} {g_val:>15.3f} {delta_str:>15}")

    print("-" * 70)
    print(f"Total queries: {mistral_metrics.get('total_queries', 0)}")
    print(f"Queries with expected: {mistral_metrics.get('queries_with_expected', 0)}")

    # Save to benchmark table
    await conn.execute("""
        INSERT INTO kb.embedding_benchmarks
            (run_name, model, retrieval_mode, query_set, query_count,
             mrr, recall_10, recall_50, config)
        VALUES
            ($1, $2, 'dense', 'qa_queries', $3, $4, $5, $5, $6::jsonb),
            ($1, $7, 'dense', 'qa_queries', $3, $8, $9, $9, $10::jsonb)
    """,
        f"benchmark_{int(time.time())}",
        EMBED_MISTRAL, mistral_metrics.get("total_queries", 0),
        mistral_metrics.get("mrr", 0), mistral_metrics.get("recall_10", 0),
        '{"dim": 1024}',
        EMBED_GEMINI,
        gemini_metrics.get("mrr", 0), gemini_metrics.get("recall_10", 0),
        '{"dim": 1024}',
    )

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
