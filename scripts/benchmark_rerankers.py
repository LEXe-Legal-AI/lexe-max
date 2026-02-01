"""
KB Massimari - Reranker Benchmark
Confronto top reranker per massima qualita' retrieval.

Reranker testati:
- Cohere Rerank 3.5 (top closed-source)
- Voyage Rerank 2 (veloce, multilingua)
- Jina Reranker v2 (open-source, structured)

Usage:
    export COHERE_API_KEY='...'
    export VOYAGE_API_KEY='...'
    export JINA_API_KEY='...'
    python scripts/benchmark_rerankers.py
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from uuid import UUID

import asyncpg
import httpx

# Config
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Mistral embedding for R1
MISTRAL_MODEL = "mistralai/mistral-embed-2312"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

# RRF config
RRF_K = 60

# Benchmark queries (subset for speed)
BENCHMARK_QUERIES = [
    # ISTITUTO
    {"type": "istituto", "query": "responsabilita civile per danni", "keywords": ["responsabil", "dann", "civil"]},
    {"type": "istituto", "query": "nullita del contratto", "keywords": ["null", "contratt"]},
    {"type": "istituto", "query": "fallimento e procedure concorsuali", "keywords": ["falliment", "concorsual"]},
    {"type": "istituto", "query": "reato di truffa elementi costitutivi", "keywords": ["truff", "reat"]},
    {"type": "istituto", "query": "omicidio colposo presupposti", "keywords": ["omicid", "colpos"]},

    # AVVERSARIA
    {"type": "avversaria", "query": "quando NON sussiste responsabilita", "keywords": ["responsabil", "sussist"]},
    {"type": "avversaria", "query": "esclusione del dolo nel reato", "keywords": ["dol", "reat"]},
    {"type": "avversaria", "query": "rigetto della domanda risarcitoria", "keywords": ["rigett", "risarcitor"]},
    {"type": "avversaria", "query": "assenza di nesso causale", "keywords": ["nesso", "causal"]},
    {"type": "avversaria", "query": "incompetenza territoriale del giudice", "keywords": ["incompetenz", "territorial"]},

    # CITAZIONE
    {"type": "citazione", "query": "art. 2043 codice civile", "keywords": ["2043", "civil"]},
    {"type": "citazione", "query": "art. 640 codice penale truffa", "keywords": ["640", "penal"]},
    {"type": "citazione", "query": "art. 575 codice penale omicidio", "keywords": ["575", "penal"]},
    {"type": "citazione", "query": "legge fallimentare art. 67", "keywords": ["67", "fallimentar"]},
]


@dataclass
class RerankerResult:
    reranker: str
    query_type: str
    query: str
    keyword_hits: int
    keyword_total: int
    latency_ms: float
    top_scores: list[float]


def embedding_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def get_mistral_embedding(client: httpx.AsyncClient, text: str) -> list[float] | None:
    """Get Mistral embedding via OpenRouter."""
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
    """R1 Hybrid search, returns (id, text, score)."""
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


# ============== RERANKER IMPLEMENTATIONS ==============

async def rerank_cohere(
    client: httpx.AsyncClient,
    query: str,
    documents: list[str],
    top_n: int = 20,
) -> tuple[list[tuple[int, float]], float]:
    """Cohere Rerank 3.5 API."""
    start = time.time()

    try:
        response = await client.post(
            "https://api.cohere.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "rerank-v3.5",
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=30.0,
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            print(f"Cohere error: {response.status_code} - {response.text[:200]}")
            return [], latency

        data = response.json()
        results = [(r["index"], r["relevance_score"]) for r in data["results"]]
        return results, latency

    except Exception as e:
        latency = (time.time() - start) * 1000
        print(f"Cohere exception: {e}")
        return [], latency


async def rerank_voyage(
    client: httpx.AsyncClient,
    query: str,
    documents: list[str],
    top_n: int = 20,
) -> tuple[list[tuple[int, float]], float]:
    """Voyage Rerank 2 API."""
    start = time.time()

    try:
        response = await client.post(
            "https://api.voyageai.com/v1/rerank",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "rerank-2",
                "query": query,
                "documents": documents,
                "top_k": top_n,
                "return_documents": False,
            },
            timeout=30.0,
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            print(f"Voyage error: {response.status_code} - {response.text[:200]}")
            return [], latency

        data = response.json()
        results = [(r["index"], r["relevance_score"]) for r in data["data"]]
        return results, latency

    except Exception as e:
        latency = (time.time() - start) * 1000
        print(f"Voyage exception: {e}")
        return [], latency


async def rerank_jina(
    client: httpx.AsyncClient,
    query: str,
    documents: list[str],
    top_n: int = 20,
) -> tuple[list[tuple[int, float]], float]:
    """Jina Reranker v2 API."""
    start = time.time()

    try:
        response = await client.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=30.0,
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            print(f"Jina error: {response.status_code} - {response.text[:200]}")
            return [], latency

        data = response.json()
        results = [(r["index"], r["relevance_score"]) for r in data["results"]]
        return results, latency

    except Exception as e:
        latency = (time.time() - start) * 1000
        print(f"Jina exception: {e}")
        return [], latency


def check_keyword_hits(texts: list[str], keywords: list[str], top_k: int = 5) -> int:
    """Count keyword hits in top-K texts."""
    hits = 0
    for text in texts[:top_k]:
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                hits += 1
                break
    return hits


async def run_benchmark():
    """Run reranker benchmark."""
    print("=" * 70)
    print("KB MASSIMARI - RERANKER BENCHMARK")
    print("=" * 70)

    # Check API keys
    rerankers = []
    if COHERE_API_KEY:
        rerankers.append(("Cohere 3.5", rerank_cohere))
        print("[OK] Cohere API key found")
    else:
        print("[SKIP] COHERE_API_KEY not set")

    if VOYAGE_API_KEY:
        rerankers.append(("Voyage 2", rerank_voyage))
        print("[OK] Voyage API key found")
    else:
        print("[SKIP] VOYAGE_API_KEY not set")

    if JINA_API_KEY:
        rerankers.append(("Jina v2", rerank_jina))
        print("[OK] Jina API key found")
    else:
        print("[SKIP] JINA_API_KEY not set")

    if not rerankers:
        print("\nERROR: No reranker API keys found!")
        print("Set at least one of: COHERE_API_KEY, VOYAGE_API_KEY, JINA_API_KEY")
        return

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY required for embeddings")
        return

    print(f"\nTesting {len(rerankers)} rerankers on {len(BENCHMARK_QUERIES)} queries")
    print()

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    all_results: list[RerankerResult] = []

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(BENCHMARK_QUERIES, 1):
            print(f"\n[{i}/{len(BENCHMARK_QUERIES)}] {q['type']}: {q['query'][:40]}...")

            # Get embedding
            embedding = await get_mistral_embedding(client, q["query"])
            if not embedding:
                print("  [ERROR] Failed to get embedding")
                continue

            # Get R1 candidates (top-30)
            r1_results = await r1_hybrid_search(conn, q["query"], embedding, limit=30)

            if not r1_results:
                print("  [WARN] No R1 results")
                continue

            documents = [text[:2000] for _, text, _ in r1_results]  # Truncate for API
            doc_ids = [mid for mid, _, _ in r1_results]

            # Test each reranker
            for reranker_name, reranker_fn in rerankers:
                reranked, latency = await reranker_fn(client, q["query"], documents)

                if not reranked:
                    print(f"  {reranker_name}: FAILED")
                    continue

                # Get top-5 reranked texts
                top_indices = [idx for idx, _ in reranked[:5]]
                top_texts = [documents[idx] for idx in top_indices]
                top_scores = [score for _, score in reranked[:5]]

                hits = check_keyword_hits(top_texts, q["keywords"])

                result = RerankerResult(
                    reranker=reranker_name,
                    query_type=q["type"],
                    query=q["query"],
                    keyword_hits=hits,
                    keyword_total=5,
                    latency_ms=latency,
                    top_scores=top_scores,
                )
                all_results.append(result)

                print(f"  {reranker_name}: {hits}/5 hits, {latency:.0f}ms, top={top_scores[0]:.3f}")

            # Rate limit - 6 seconds to respect 10 calls/min for trial keys
            await asyncio.sleep(6)

    await conn.close()

    # Generate report
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for reranker_name, _ in rerankers:
        results = [r for r in all_results if r.reranker == reranker_name]
        if not results:
            continue

        total_hits = sum(r.keyword_hits for r in results)
        total_possible = sum(r.keyword_total for r in results)
        avg_hits = total_hits / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_top_score = sum(r.top_scores[0] for r in results) / len(results)

        print(f"\n### {reranker_name}")
        print(f"Queries: {len(results)}")
        print(f"Avg hits: {avg_hits:.2f}/5 ({total_hits}/{total_possible})")
        print(f"Accuracy: {avg_hits/5*100:.1f}%")
        print(f"Avg latency: {avg_latency:.0f}ms")
        print(f"Avg top score: {avg_top_score:.3f}")

        # By type
        for qtype in ["istituto", "avversaria", "citazione"]:
            type_results = [r for r in results if r.query_type == qtype]
            if type_results:
                type_hits = sum(r.keyword_hits for r in type_results) / len(type_results)
                print(f"  {qtype}: {type_hits:.2f}/5")

    # Winner
    print("\n" + "=" * 70)
    print("RANKING")
    print("=" * 70)

    rankings = []
    for reranker_name, _ in rerankers:
        results = [r for r in all_results if r.reranker == reranker_name]
        if results:
            avg_hits = sum(r.keyword_hits for r in results) / len(results)
            avg_latency = sum(r.latency_ms for r in results) / len(results)
            rankings.append((reranker_name, avg_hits, avg_latency))

    rankings.sort(key=lambda x: (-x[1], x[2]))  # Best hits, then lowest latency

    print("\n| Rank | Reranker | Avg Hits | Accuracy | Latency |")
    print("|------|----------|----------|----------|---------|")
    for i, (name, hits, lat) in enumerate(rankings, 1):
        acc = hits / 5 * 100
        print(f"| {i} | {name} | {hits:.2f}/5 | {acc:.1f}% | {lat:.0f}ms |")

    if rankings:
        winner = rankings[0][0]
        print(f"\n** WINNER: {winner} **")

    # Save results
    output_path = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/reranker_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": len(BENCHMARK_QUERIES),
        "rerankers_tested": [name for name, _ in rerankers],
        "results": [asdict(r) for r in all_results],
        "rankings": [{"reranker": name, "avg_hits": hits, "accuracy": hits/5*100, "latency_ms": lat}
                     for name, hits, lat in rankings],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {output_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
