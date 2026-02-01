"""
Test OpenRouter Embedding Models
Confronta i principali modelli su campioni di massime italiane.

Requires: OPENROUTER_API_KEY env var
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import asyncpg
import httpx
import numpy as np

# Config
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Modelli da testare - Round 2
MODELS = [
    {
        "id": "mistralai/codestral-embed-2505",
        "name": "Codestral Embed",
        "dim": 1024,  # Native dim
        "priority": 1,
    },
    {
        "id": "google/gemini-embedding-001",
        "name": "Gemini Embed",
        "dim": 768,  # Ridotto per pgvector
        "priority": 2,
    },
]

# Query di test (istituto, avversaria, citazione)
TEST_QUERIES = [
    {
        "type": "istituto",
        "query": "responsabilita medica per colpa grave",
        "expected_keywords": ["medic", "colpa", "responsabil"],
    },
    {
        "type": "avversaria",
        "query": "quando NON sussiste il reato di truffa",
        "expected_keywords": ["truffa", "sussist", "reat"],
    },
    {
        "type": "citazione",
        "query": "art. 640 codice penale",
        "expected_keywords": ["640", "penal", "truffa"],
    },
]


@dataclass
class EmbeddingResult:
    model_id: str
    model_name: str
    dim: int
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class RetrievalResult:
    model_name: str
    query_type: str
    query: str
    top_5_ids: list[str]
    top_5_scores: list[float]
    keyword_hits: int
    keyword_total: int


async def get_embedding_openrouter(
    client: httpx.AsyncClient,
    model_id: str,
    texts: list[str],
) -> tuple[list[list[float]] | None, float, str | None]:
    """Get embeddings from OpenRouter API."""
    start = time.time()

    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "input": texts,
            },
            timeout=60.0,
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            return None, latency, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings, latency, None

    except Exception as e:
        latency = (time.time() - start) * 1000
        return None, latency, str(e)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


async def load_sample_massime(conn: asyncpg.Connection, limit: int = 50) -> list[dict]:
    """Load sample massime from each document type."""
    rows = await conn.fetch("""
        WITH ranked AS (
            SELECT m.id, m.testo, d.anno, d.tipo,
                   ROW_NUMBER() OVER (PARTITION BY d.tipo ORDER BY RANDOM()) as rn
            FROM kb.massime m
            JOIN kb.documents d ON d.id = m.document_id
        )
        SELECT id, testo, anno, tipo
        FROM ranked
        WHERE rn <= $1
        ORDER BY tipo, anno
    """, limit // 3 + 1)

    return [dict(row) for row in rows]


async def test_model(
    client: httpx.AsyncClient,
    model: dict,
    massime: list[dict],
    queries: list[dict],
) -> tuple[EmbeddingResult, list[RetrievalResult]]:
    """Test a single embedding model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model['name']} ({model['id']})")
    print(f"{'='*60}")

    # 1. Embed all massime
    texts = [m["testo"][:2000] for m in massime]  # Truncate for API limits

    embeddings, latency, error = await get_embedding_openrouter(
        client, model["id"], texts
    )

    if error:
        print(f"  ERROR: {error}")
        return EmbeddingResult(
            model_id=model["id"],
            model_name=model["name"],
            dim=model["dim"],
            latency_ms=latency,
            success=False,
            error=error,
        ), []

    print(f"  Embedded {len(texts)} massime in {latency:.0f}ms")
    print(f"  Dimension: {len(embeddings[0])}")

    # 2. Test retrieval for each query
    retrieval_results = []

    for q in queries:
        # Embed query
        query_emb, q_latency, q_error = await get_embedding_openrouter(
            client, model["id"], [q["query"]]
        )

        if q_error:
            print(f"  Query '{q['type']}' failed: {q_error}")
            continue

        # Calculate similarities
        similarities = [
            (i, cosine_similarity(query_emb[0], emb))
            for i, emb in enumerate(embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top 5
        top_5 = similarities[:5]
        top_5_ids = [str(massime[i]["id"])[:8] for i, _ in top_5]
        top_5_scores = [score for _, score in top_5]

        # Check keyword hits in top 5 results
        keyword_hits = 0
        for i, _ in top_5:
            testo_lower = massime[i]["testo"].lower()
            for kw in q["expected_keywords"]:
                if kw.lower() in testo_lower:
                    keyword_hits += 1
                    break

        result = RetrievalResult(
            model_name=model["name"],
            query_type=q["type"],
            query=q["query"],
            top_5_ids=top_5_ids,
            top_5_scores=top_5_scores,
            keyword_hits=keyword_hits,
            keyword_total=5,
        )
        retrieval_results.append(result)

        print(f"  [{q['type']}] '{q['query'][:30]}...'")
        print(f"    Top score: {top_5_scores[0]:.4f}, Keyword hits: {keyword_hits}/5")

    return EmbeddingResult(
        model_id=model["id"],
        model_name=model["name"],
        dim=len(embeddings[0]),
        latency_ms=latency,
        success=True,
    ), retrieval_results


async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  export OPENROUTER_API_KEY='sk-or-...'")
        return

    print("=" * 70)
    print("OPENROUTER EMBEDDING MODELS - BENCHMARK")
    print("=" * 70)

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Load sample massime
    massime = await load_sample_massime(conn, limit=30)
    print(f"[OK] Loaded {len(massime)} sample massime")

    # Show distribution
    by_tipo = {}
    for m in massime:
        by_tipo[m["tipo"]] = by_tipo.get(m["tipo"], 0) + 1
    print(f"    Distribution: {by_tipo}")

    await conn.close()

    # Test each model
    all_embedding_results = []
    all_retrieval_results = []

    async with httpx.AsyncClient() as client:
        for model in MODELS:
            emb_result, retr_results = await test_model(
                client, model, massime, TEST_QUERIES
            )
            all_embedding_results.append(emb_result)
            all_retrieval_results.extend(retr_results)

            # Small delay between models
            await asyncio.sleep(1)

    # Generate report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    # Embedding performance
    print("\n### Embedding Performance\n")
    print("| Model | Dim | Latency (ms) | Status |")
    print("|-------|-----|--------------|--------|")
    for r in all_embedding_results:
        status = "OK" if r.success else f"FAIL: {r.error[:30]}"
        print(f"| {r.model_name} | {r.dim} | {r.latency_ms:.0f} | {status} |")

    # Retrieval quality
    print("\n### Retrieval Quality (Keyword Hits in Top 5)\n")
    print("| Model | Istituto | Avversaria | Citazione | Avg |")
    print("|-------|----------|------------|-----------|-----|")

    by_model = {}
    for r in all_retrieval_results:
        if r.model_name not in by_model:
            by_model[r.model_name] = {}
        by_model[r.model_name][r.query_type] = r.keyword_hits

    for model_name, scores in by_model.items():
        ist = scores.get("istituto", 0)
        avv = scores.get("avversaria", 0)
        cit = scores.get("citazione", 0)
        avg = (ist + avv + cit) / 3
        print(f"| {model_name} | {ist}/5 | {avv}/5 | {cit}/5 | {avg:.1f}/5 |")

    # Top scores
    print("\n### Top Similarity Scores\n")
    print("| Model | Query Type | Top Score |")
    print("|-------|------------|-----------|")
    for r in all_retrieval_results:
        if r.top_5_scores:
            print(f"| {r.model_name} | {r.query_type} | {r.top_5_scores[0]:.4f} |")

    # Save results
    output_path = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/openrouter_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(massime),
        "embedding_results": [asdict(r) for r in all_embedding_results],
        "retrieval_results": [asdict(r) for r in all_retrieval_results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {output_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
