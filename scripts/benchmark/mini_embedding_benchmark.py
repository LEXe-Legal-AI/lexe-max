#!/usr/bin/env python
"""
Mini-Benchmark Embeddings per Altalex Normativa

Testa diversi modelli di embedding su articoli GDPR per determinare
il miglior provider per la pipeline di ingestion.

Modelli testati:
- text-embedding-3-small via OpenRouter (1536 dims)
- multilingual-e5-large-instruct via sentence-transformers (1024 dims)
- bge-m3 via sentence-transformers (1024 dims)
- Italian-Legal-BERT via sentence-transformers (768 dims)

Metriche:
- Recall@10 su query manuali
- Latency per embedding
- Throughput (embeddings/sec)

Usage:
    OPENROUTER_API_KEY=sk-or-... python scripts/benchmark/mini_embedding_benchmark.py
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Risultato benchmark per un modello."""
    model_name: str
    dims: int
    provider: str

    # Performance
    total_embeddings: int = 0
    total_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0

    # Retrieval quality
    recall_at_10: float = 0.0
    mrr: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)


@dataclass
class TestQuery:
    """Query di test con ground truth."""
    query: str
    expected_articles: list[str]  # List of articolo_num expected in top 10
    query_type: str  # 'direct', 'semantic', 'keyword'


# Test queries per GDPR
GDPR_TEST_QUERIES = [
    TestQuery(
        query="diritto all'oblio cancellazione dati",
        expected_articles=["17"],
        query_type="semantic"
    ),
    TestQuery(
        query="consenso al trattamento dati personali",
        expected_articles=["7", "8"],
        query_type="semantic"
    ),
    TestQuery(
        query="portabilità dei dati",
        expected_articles=["20"],
        query_type="semantic"
    ),
    TestQuery(
        query="data protection officer DPO",
        expected_articles=["37", "38", "39"],
        query_type="keyword"
    ),
    TestQuery(
        query="trasferimento dati verso paesi terzi",
        expected_articles=["44", "45", "46"],
        query_type="semantic"
    ),
    TestQuery(
        query="sanzioni amministrative violazioni",
        expected_articles=["83", "84"],
        query_type="semantic"
    ),
    TestQuery(
        query="responsabile del trattamento",
        expected_articles=["28", "29"],
        query_type="keyword"
    ),
    TestQuery(
        query="informativa privacy interessato",
        expected_articles=["13", "14"],
        query_type="semantic"
    ),
    TestQuery(
        query="valutazione impatto protezione dati DPIA",
        expected_articles=["35", "36"],
        query_type="keyword"
    ),
    TestQuery(
        query="diritto di rettifica",
        expected_articles=["16"],
        query_type="direct"
    ),
]


class EmbeddingProvider:
    """Base class per embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


class OpenRouterProvider(EmbeddingProvider):
    """OpenRouter API per embeddings."""

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                }
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index and extract embeddings
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]


class SentenceTransformersProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        # Run in thread pool since it's CPU/GPU bound
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def run_benchmark(
    provider: EmbeddingProvider,
    model_name: str,
    dims: int,
    provider_name: str,
    articles: list[dict],
    test_queries: list[TestQuery],
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Run benchmark for a single embedding provider.

    Args:
        provider: Embedding provider
        model_name: Model identifier
        dims: Expected dimensions
        provider_name: Provider name (for logging)
        articles: List of articles with 'articolo_num' and 'testo'
        test_queries: List of test queries
        batch_size: Batch size for embedding
    """
    result = BenchmarkResult(
        model_name=model_name,
        dims=dims,
        provider=provider_name,
    )

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} ({provider_name})")
    print(f"{'='*60}")

    # 1. Embed all articles
    print(f"\nEmbedding {len(articles)} articles...")
    article_texts = [a["testo"][:2000] for a in articles]  # Truncate long texts

    start = time.time()
    try:
        article_embeddings = await provider.embed(article_texts)
        result.total_time_ms = (time.time() - start) * 1000
        result.total_embeddings = len(article_embeddings)
        result.avg_latency_ms = result.total_time_ms / result.total_embeddings
        result.throughput_per_sec = result.total_embeddings / (result.total_time_ms / 1000)

        print(f"  Embeddings: {result.total_embeddings}")
        print(f"  Total time: {result.total_time_ms:.1f}ms")
        print(f"  Avg latency: {result.avg_latency_ms:.1f}ms/embedding")
        print(f"  Throughput: {result.throughput_per_sec:.1f}/sec")

        # Verify dims
        actual_dims = len(article_embeddings[0])
        if actual_dims != dims:
            print(f"  WARNING: Expected {dims} dims, got {actual_dims}")

    except Exception as e:
        result.errors.append(f"Embedding failed: {e}")
        print(f"  ERROR: {e}")
        return result

    # 2. Run retrieval benchmark
    print(f"\nRunning retrieval benchmark ({len(test_queries)} queries)...")

    total_recall = 0.0
    total_mrr = 0.0

    for q in test_queries:
        try:
            # Embed query
            query_emb = await provider.embed_single(q.query)

            # Calculate similarities
            similarities = []
            for i, (art, emb) in enumerate(zip(articles, article_embeddings)):
                sim = cosine_similarity(query_emb, emb)
                similarities.append((art["articolo_num"], sim))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_10 = [x[0] for x in similarities[:10]]

            # Calculate Recall@10
            hits = sum(1 for exp in q.expected_articles if exp in top_10)
            recall = hits / len(q.expected_articles)
            total_recall += recall

            # Calculate MRR (position of first hit)
            mrr = 0.0
            for i, art_num in enumerate(top_10):
                if art_num in q.expected_articles:
                    mrr = 1.0 / (i + 1)
                    break
            total_mrr += mrr

            # Print results
            hit_marker = "[OK]" if recall > 0 else "[NO]"
            print(f"  {hit_marker} '{q.query[:40]}...' -> Recall: {recall:.0%}, MRR: {mrr:.2f}")
            print(f"      Expected: {q.expected_articles}, Got top-3: {top_10[:3]}")

        except Exception as e:
            result.errors.append(f"Query failed '{q.query[:30]}': {e}")
            print(f"  [ERR] Query error: {e}")

    result.recall_at_10 = total_recall / len(test_queries)
    result.mrr = total_mrr / len(test_queries)

    print(f"\n  Overall Recall@10: {result.recall_at_10:.1%}")
    print(f"  Overall MRR: {result.mrr:.3f}")

    return result


async def main():
    """Run mini-benchmark."""

    # Check for API key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    # Load GDPR articles using marker_chunker
    print("Loading GDPR articles...")

    from lexe_api.kb.ingestion.marker_chunker import MarkerChunker

    gdpr_json = Path(__file__).parent.parent.parent.parent / "altalex-md" / "chunks-test" / "gdpr-24-01-2019 pdf" / "gdpr-24-01-2019 pdf.json"

    if not gdpr_json.exists():
        print(f"ERROR: GDPR JSON not found at {gdpr_json}")
        print("Please run marker first to generate the JSON.")
        sys.exit(1)

    chunker = MarkerChunker()
    articles = chunker.process_file(gdpr_json, "GDPR")

    # Convert to dict format
    articles_data = [
        {
            "articolo_num": art.articolo_num,
            "rubrica": art.rubrica,
            "testo": art.testo or art.rubrica or "",  # Fallback to rubrica if testo empty
        }
        for art in articles
        if art.testo or art.rubrica  # Skip completely empty articles
    ]

    print(f"Loaded {len(articles_data)} articles with content")

    # Define providers to test
    providers = []

    # OpenRouter (if key available)
    if openrouter_key:
        providers.append({
            "name": "openai/text-embedding-3-small",
            "provider": OpenRouterProvider(openrouter_key, "openai/text-embedding-3-small"),
            "dims": 1536,
            "provider_name": "OpenRouter",
        })
    else:
        print("\nWARNING: OPENROUTER_API_KEY not set, skipping OpenRouter tests")

    # Local sentence-transformers
    try:
        import sentence_transformers

        providers.extend([
            {
                "name": "multilingual-e5-large-instruct",
                "provider": SentenceTransformersProvider("intfloat/multilingual-e5-large-instruct"),
                "dims": 1024,
                "provider_name": "sentence-transformers",
            },
            # BGE-M3 - skip for now (large model)
            # {
            #     "name": "bge-m3",
            #     "provider": SentenceTransformersProvider("BAAI/bge-m3"),
            #     "dims": 1024,
            #     "provider_name": "sentence-transformers",
            # },
        ])
    except ImportError:
        print("\nWARNING: sentence-transformers not installed, skipping local tests")
        print("Install with: pip install sentence-transformers")

    if not providers:
        print("\nERROR: No providers available. Set OPENROUTER_API_KEY or install sentence-transformers.")
        sys.exit(1)

    # Run benchmarks
    results = []
    for p in providers:
        try:
            result = await run_benchmark(
                provider=p["provider"],
                model_name=p["name"],
                dims=p["dims"],
                provider_name=p["provider_name"],
                articles=articles_data,
                test_queries=GDPR_TEST_QUERIES,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR benchmarking {p['name']}: {e}")

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"\n{'Model':<40} {'Recall@10':<12} {'MRR':<8} {'Latency':<12} {'Throughput'}")
    print("-"*90)

    for r in sorted(results, key=lambda x: x.recall_at_10, reverse=True):
        print(f"{r.model_name:<40} {r.recall_at_10:>10.1%} {r.mrr:>8.3f} {r.avg_latency_ms:>10.1f}ms {r.throughput_per_sec:>8.1f}/s")

    # Recommendation
    if results:
        best = max(results, key=lambda x: x.recall_at_10)
        print(f"\n>> RECOMMENDED: {best.model_name}")
        print(f"  Recall@10: {best.recall_at_10:.1%}, MRR: {best.mrr:.3f}")
        print(f"  Latency: {best.avg_latency_ms:.1f}ms, Throughput: {best.throughput_per_sec:.1f}/s")

    # Save results
    results_file = Path(__file__).parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([
            {
                "model": r.model_name,
                "provider": r.provider,
                "dims": r.dims,
                "recall_at_10": r.recall_at_10,
                "mrr": r.mrr,
                "avg_latency_ms": r.avg_latency_ms,
                "throughput_per_sec": r.throughput_per_sec,
                "errors": r.errors,
            }
            for r in results
        ], f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
