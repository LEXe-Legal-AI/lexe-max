"""
KB Massimari - Complete R1/R2 Benchmark
Confronto: R1 Hybrid vs R2-Cohere vs R2-Local

Usage:
    export OPENROUTER_API_KEY='sk-or-...'
    export COHERE_API_KEY='...'
    python scripts/benchmark_r1_r2_complete.py
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import asyncpg
import httpx

# Config
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MISTRAL_MODEL = "mistralai/mistral-embed-2312"
RRF_K = 60

# Local reranker model
LOCAL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Alternative multilingual: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Benchmark queries
BENCHMARK_QUERIES = [
    {"type": "istituto", "query": "responsabilita civile per danni", "keywords": ["responsabil", "dann", "civil"]},
    {"type": "istituto", "query": "nullita del contratto", "keywords": ["null", "contratt"]},
    {"type": "istituto", "query": "fallimento e procedure concorsuali", "keywords": ["falliment", "concorsual"]},
    {"type": "istituto", "query": "reato di truffa elementi costitutivi", "keywords": ["truff", "reat"]},
    {"type": "istituto", "query": "omicidio colposo presupposti", "keywords": ["omicid", "colpos"]},
    {"type": "avversaria", "query": "quando NON sussiste responsabilita", "keywords": ["responsabil", "sussist"]},
    {"type": "avversaria", "query": "esclusione del dolo nel reato", "keywords": ["dol", "reat"]},
    {"type": "avversaria", "query": "rigetto della domanda risarcitoria", "keywords": ["rigett", "risarcitor"]},
    {"type": "avversaria", "query": "assenza di nesso causale", "keywords": ["nesso", "causal"]},
    {"type": "avversaria", "query": "incompetenza territoriale del giudice", "keywords": ["incompetenz", "territorial"]},
    {"type": "citazione", "query": "art. 2043 codice civile", "keywords": ["2043", "civil"]},
    {"type": "citazione", "query": "art. 640 codice penale truffa", "keywords": ["640", "penal"]},
    {"type": "citazione", "query": "art. 575 codice penale omicidio", "keywords": ["575", "penal"]},
    {"type": "citazione", "query": "legge fallimentare art. 67", "keywords": ["67", "fallimentar"]},
]


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


@dataclass
class QueryResult:
    query_type: str
    query: str
    method: str
    keyword_hits: int
    keyword_total: int
    latency_ms: float


def embedding_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def get_mistral_embedding(client: httpx.AsyncClient, text: str) -> list[float] | None:
    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": MISTRAL_MODEL, "input": [text]},
            timeout=30.0,
        )
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
    return None


async def r1_hybrid_search(
    conn: asyncpg.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 30,
) -> list[tuple[UUID, str, float]]:
    """R1 Hybrid search."""
    emb_str = embedding_to_pgvector(query_embedding)

    rows = await conn.fetch("""
        WITH dense_results AS (
            SELECT e.massima_id, ROW_NUMBER() OVER (ORDER BY e.embedding <=> $1::vector) as rank
            FROM kb.emb_mistral e
            ORDER BY e.embedding <=> $1::vector
            LIMIT 50
        ),
        sparse_results AS (
            SELECT m.id as massima_id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', $2)) DESC) as rank
            FROM kb.massime m
            WHERE m.tsv_italian @@ plainto_tsquery('italian', $2)
               OR m.tsv_simple @@ plainto_tsquery('simple', $2)
            LIMIT 50
        ),
        combined AS (
            SELECT COALESCE(d.massima_id, s.massima_id) as massima_id,
                   COALESCE(1.0 / ($3 + d.rank), 0) + COALESCE(1.0 / ($3 + s.rank), 0) as rrf_score
            FROM dense_results d
            FULL OUTER JOIN sparse_results s ON d.massima_id = s.massima_id
        )
        SELECT c.massima_id, m.testo, c.rrf_score
        FROM combined c
        JOIN kb.massime m ON m.id = c.massima_id
        ORDER BY c.rrf_score DESC
        LIMIT $4
    """, emb_str, query_text, RRF_K, limit)

    return [(row["massima_id"], row["testo"], float(row["rrf_score"])) for row in rows]


async def cohere_rerank(
    client: httpx.AsyncClient,
    query: str,
    documents: list[str],
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """Cohere Rerank API."""
    try:
        response = await client.post(
            "https://api.cohere.ai/v1/rerank",
            headers={"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "rerank-v3.5",
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=30.0,
        )
        if response.status_code == 200:
            data = response.json()
            return [(r["index"], r["relevance_score"]) for r in data["results"]]
    except Exception as e:
        print(f"Cohere error: {e}")
    return []


class LocalReranker:
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = LOCAL_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        print(f"Loading local reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        print("Local reranker loaded")

    def rerank(self, query: str, documents: list[str], top_n: int = 20) -> list[tuple[int, float]]:
        """Rerank documents using local cross-encoder."""
        pairs = [[query, doc[:500]] for doc in documents]  # Truncate for speed
        scores = self.model.predict(pairs)

        # Create (index, score) pairs and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_n]


def check_keyword_hits(texts: list[str], keywords: list[str], top_k: int = 5) -> int:
    hits = 0
    for text in texts[:top_k]:
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                hits += 1
                break
    return hits


async def run_benchmark():
    """Run complete R1/R2 benchmark."""
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY")
        return

    print("=" * 70)
    print("KB MASSIMARI - COMPLETE R1/R2 BENCHMARK")
    print("=" * 70)
    print(f"Queries: {len(BENCHMARK_QUERIES)}")
    print("Methods: R1 Hybrid, R2-Cohere, R2-Local")
    print()

    # Initialize local reranker
    local_reranker = None
    try:
        local_reranker = LocalReranker()
    except Exception as e:
        print(f"[WARN] Local reranker failed to load: {e}")

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    all_results: list[QueryResult] = []

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(BENCHMARK_QUERIES, 1):
            print(f"\n[{i}/{len(BENCHMARK_QUERIES)}] {q['type']}: {q['query'][:40]}...")

            # Get embedding
            embedding = await get_mistral_embedding(client, q["query"])
            if not embedding:
                print("  [ERROR] Failed to get embedding")
                continue

            # R1: Hybrid
            start = time.time()
            r1_results = await r1_hybrid_search(conn, q["query"], embedding, limit=30)
            r1_latency = (time.time() - start) * 1000

            if not r1_results:
                print("  [WARN] No R1 results")
                continue

            documents = [text for _, text, _ in r1_results]
            r1_texts = documents[:5]
            r1_hits = check_keyword_hits(r1_texts, q["keywords"])

            all_results.append(QueryResult(
                query_type=q["type"], query=q["query"], method="R1",
                keyword_hits=r1_hits, keyword_total=5, latency_ms=r1_latency
            ))
            print(f"  R1: {r1_hits}/5 hits, {r1_latency:.0f}ms")

            # R2-Cohere
            if COHERE_API_KEY:
                start = time.time()
                cohere_results = await cohere_rerank(client, q["query"], [d[:2000] for d in documents])
                cohere_latency = (time.time() - start) * 1000 + r1_latency

                if cohere_results:
                    cohere_texts = [documents[idx] for idx, _ in cohere_results[:5]]
                    cohere_hits = check_keyword_hits(cohere_texts, q["keywords"])

                    all_results.append(QueryResult(
                        query_type=q["type"], query=q["query"], method="R2-Cohere",
                        keyword_hits=cohere_hits, keyword_total=5, latency_ms=cohere_latency
                    ))
                    print(f"  R2-Cohere: {cohere_hits}/5 hits, {cohere_latency:.0f}ms")

            # R2-Local
            if local_reranker:
                start = time.time()
                local_results = local_reranker.rerank(q["query"], documents)
                local_latency = (time.time() - start) * 1000 + r1_latency

                local_texts = [documents[idx] for idx, _ in local_results[:5]]
                local_hits = check_keyword_hits(local_texts, q["keywords"])

                all_results.append(QueryResult(
                    query_type=q["type"], query=q["query"], method="R2-Local",
                    keyword_hits=local_hits, keyword_total=5, latency_ms=local_latency
                ))
                print(f"  R2-Local: {local_hits}/5 hits, {local_latency:.0f}ms")

            # Rate limit for Cohere trial
            await asyncio.sleep(6)

    await conn.close()

    # Generate report
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    methods = ["R1", "R2-Cohere", "R2-Local"]
    method_stats = {}

    for method in methods:
        results = [r for r in all_results if r.method == method]
        if not results:
            continue

        total_hits = sum(r.keyword_hits for r in results)
        total_possible = sum(r.keyword_total for r in results)
        avg_hits = total_hits / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        method_stats[method] = {"avg_hits": avg_hits, "accuracy": avg_hits/5*100, "latency": avg_latency}

        print(f"\n### {method}")
        print(f"Queries: {len(results)}")
        print(f"Avg hits: {avg_hits:.2f}/5")
        print(f"Accuracy: {avg_hits/5*100:.1f}%")
        print(f"Avg latency: {avg_latency:.0f}ms")

        for qtype in ["istituto", "avversaria", "citazione"]:
            type_results = [r for r in results if r.query_type == qtype]
            if type_results:
                type_hits = sum(r.keyword_hits for r in type_results) / len(type_results)
                print(f"  {qtype}: {type_hits:.2f}/5")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("\n| Method | Avg Hits | Accuracy | Latency | vs R1 |")
    print("|--------|----------|----------|---------|-------|")

    r1_acc = method_stats.get("R1", {}).get("accuracy", 0)
    for method in methods:
        if method in method_stats:
            stats = method_stats[method]
            delta = stats["accuracy"] - r1_acc if method != "R1" else 0
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%" if delta < 0 else "-"
            print(f"| {method} | {stats['avg_hits']:.2f}/5 | {stats['accuracy']:.1f}% | {stats['latency']:.0f}ms | {delta_str} |")

    # Winner
    if method_stats:
        winner = max(method_stats.items(), key=lambda x: (x[1]["accuracy"], -x[1]["latency"]))
        print(f"\n** WINNER: {winner[0]} ({winner[1]['accuracy']:.1f}% accuracy) **")

    # Save results
    output_path = Path("C:/PROJECTS/LEO-ITC/lexe-api/data/r1_r2_complete_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries": len(BENCHMARK_QUERIES),
            "results": [asdict(r) for r in all_results],
            "summary": method_stats,
        }, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)

    print(f"\n[OK] Results saved to {output_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
