#!/usr/bin/env python3
"""
Test Cross-Encoder Reranker for KB Normativa.

This script tests the reranking layer that takes top-20 results from hybrid search
and re-scores them using a cross-encoder model.

Usage:
    # Basic test (uses local model)
    python scripts/test_reranker.py

    # With staging database
    OPENROUTER_API_KEY=sk-or-... python scripts/test_reranker.py --staging

    # Test specific model
    python scripts/test_reranker.py --model bge-m3
    python scripts/test_reranker.py --model mmarco

    # Custom query
    python scripts/test_reranker.py --query "responsabilita civile art. 2043"

Requirements:
    - sentence-transformers>=5.2.2
    - psycopg2 or asyncpg
    - OPENROUTER_API_KEY (for embeddings, only with --staging)
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

STAGING_DB = {
    "host": "localhost",
    "port": 5436,
    "user": "lexe_max",
    "password": "lexe_max_dev_password",
    "dbname": "lexe_max",
}

LOCAL_DB = {
    "host": "localhost",
    "port": 5434,
    "user": "lexe_kb",
    "password": "lexe_kb_dev_password",
    "dbname": "lexe_kb",
}

TEST_QUERIES = [
    {
        "query": "risarcimento danno responsabilita civile",
        "keywords": ["risarcimento", "danno", "responsabilita", "2043"],
        "type": "civil",
    },
    {
        "query": "nullita del contratto cause",
        "keywords": ["nullita", "contratto", "1418"],
        "type": "civil",
    },
    {
        "query": "obbligazioni solidali adempimento",
        "keywords": ["solidali", "obbligazione", "1292"],
        "type": "civil",
    },
    {
        "query": "prescrizione diritti termine",
        "keywords": ["prescrizione", "termine", "2934"],
        "type": "civil",
    },
    {
        "query": "proprieta possesso usucapione",
        "keywords": ["proprieta", "possesso", "usucapione", "1158"],
        "type": "civil",
    },
]


@dataclass
class TestResult:
    """Result of a single reranking test."""

    query: str
    query_type: str
    hybrid_top5_keywords: int
    reranked_top5_keywords: int
    rank_changes: int
    latency_ms: float
    top_rerank_score: float
    improvement: int  # positive = better


# =============================================================================
# Embedding utilities
# =============================================================================


def get_embedding_openrouter(text: str, api_key: str) -> list[float]:
    """Get embedding via OpenRouter API."""
    import requests

    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "openai/text-embedding-3-small", "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# =============================================================================
# Database utilities
# =============================================================================


def hybrid_search_sync(conn, query: str, query_embedding: list[float], limit: int = 20):
    """
    Run hybrid search (dense + sparse + RRF) and return chunks.

    Returns list of dicts with: chunk_id, work_code, articolo, chunk_no,
                                text, rrf_score, dense_score, sparse_score, rank
    """
    cur = conn.cursor()

    RRF_K = 60

    cur.execute(
        """
WITH query_params AS (
    SELECT
        %s::vector(1536) as qemb,
        plainto_tsquery('italian', %s) as qtsv
),
-- Dense search (top 50)
dense AS (
    SELECT c.id as chunk_id,
           ROW_NUMBER() OVER (ORDER BY e.embedding <=> q.qemb) as rank_dense,
           1 - (e.embedding <=> q.qemb) as score_dense
    FROM kb.normativa_chunk c
    JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
    CROSS JOIN query_params q
    ORDER BY e.embedding <=> q.qemb
    LIMIT 50
),
-- Sparse search (top 50)
sparse AS (
    SELECT f.chunk_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC) as rank_sparse,
           ts_rank_cd(f.tsv_it, q.qtsv) as score_sparse
    FROM kb.normativa_chunk_fts f
    CROSS JOIN query_params q
    WHERE f.tsv_it @@ q.qtsv
    ORDER BY ts_rank_cd(f.tsv_it, q.qtsv) DESC
    LIMIT 50
),
-- RRF Fusion
rrf AS (
    SELECT
        COALESCE(d.chunk_id, s.chunk_id) as chunk_id,
        COALESCE(1.0 / (%s + d.rank_dense), 0) + COALESCE(1.0 / (%s + s.rank_sparse), 0) as rrf_score,
        d.score_dense,
        s.score_sparse,
        ROW_NUMBER() OVER (
            ORDER BY COALESCE(1.0 / (%s + d.rank_dense), 0) + COALESCE(1.0 / (%s + s.rank_sparse), 0) DESC
        ) as rrf_rank
    FROM dense d
    FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
)
SELECT r.chunk_id, w.code as work_code, n.articolo, c.chunk_no,
       c.text, r.rrf_score, r.score_dense, r.score_sparse, r.rrf_rank
FROM rrf r
JOIN kb.normativa_chunk c ON c.id = r.chunk_id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
ORDER BY r.rrf_score DESC
LIMIT %s;
    """,
        (query_embedding, query, RRF_K, RRF_K, RRF_K, RRF_K, limit),
    )

    rows = cur.fetchall()
    cur.close()

    return [
        {
            "chunk_id": row[0],
            "work_code": row[1],
            "articolo": row[2],
            "chunk_no": row[3],
            "text": row[4],
            "rrf_score": float(row[5]) if row[5] else 0,
            "dense_score": float(row[6]) if row[6] else None,
            "sparse_score": float(row[7]) if row[7] else None,
            "rank": row[8],
        }
        for row in rows
    ]


def count_keyword_hits(texts: list[str], keywords: list[str]) -> int:
    """Count how many texts contain at least one keyword."""
    hits = 0
    for text in texts:
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                hits += 1
                break
    return hits


# =============================================================================
# Test functions
# =============================================================================


async def test_reranker_basic():
    """Test reranker with synthetic data (no database)."""
    from lexe_api.kb.retrieval.reranker import (
        CrossEncoderReranker,
        RerankerModel,
    )

    print("=" * 70)
    print("TEST 1: Basic CrossEncoder Reranker (no database)")
    print("=" * 70)

    # Test data - Italian legal text
    query = "risarcimento danno responsabilita civile"
    documents = [
        "Art. 2043 c.c. - Risarcimento per fatto illecito. Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno.",
        "Art. 2044 c.c. - Legittima difesa. Non e responsabile chi cagiona il danno per legittima difesa di se o di altri.",
        "Art. 1218 c.c. - Responsabilita del debitore. Il debitore che non esegue esattamente la prestazione dovuta e tenuto al risarcimento del danno.",
        "Art. 2054 c.c. - Circolazione di veicoli. Il conducente di un veicolo senza guida di rotaie e obbligato a risarcire il danno prodotto.",
        "Art. 844 c.c. - Immissioni. Il proprietario di un fondo non puo impedire le immissioni di fumo o di calore.",
        "Art. 2059 c.c. - Danni non patrimoniali. Il danno non patrimoniale deve essere risarcito solo nei casi determinati dalla legge.",
        "Art. 1321 c.c. - Nozione. Il contratto e l'accordo di due o piu parti per costituire, regolare o estinguere un rapporto giuridico.",
        "Art. 2056 c.c. - Valutazione dei danni. Il risarcimento dovuto al danneggiato si deve determinare secondo le disposizioni degli artt. 1223, 1226 e 1227.",
    ]

    # Test different models
    models_to_test = [
        (RerankerModel.BGE_M3, "BGE-M3 (multilingual)"),
        # Uncomment to test other models:
        # (RerankerModel.MMARCO_MULTILINGUAL, "mMARCO (multilingual)"),
    ]

    for model, model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")

        start = time.time()
        reranker = CrossEncoderReranker(model=model)

        # Test predict method directly
        pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(pairs)
        latency = (time.time() - start) * 1000

        print(f"Model loaded and scored in {latency:.0f}ms")
        print("\nResults (sorted by rerank score):")
        print("-" * 70)

        # Sort by score
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        for i, (doc, score) in enumerate(scored, 1):
            preview = doc[:80].replace("\n", " ")
            print(f"{i}. [{score:.4f}] {preview}...")

        # Check if art. 2043 (most relevant) is at top
        art_2043_rank = next(
            (i for i, (doc, _) in enumerate(scored, 1) if "2043" in doc), -1
        )
        print(f"\nArt. 2043 (expected top) ranked: #{art_2043_rank}")

    print("\n[OK] Basic test passed")
    return True


async def test_reranker_with_database(db_config: dict, api_key: str | None = None):
    """Test reranker with real database and hybrid search."""
    import psycopg2

    from lexe_api.kb.retrieval.reranker import (
        CrossEncoderReranker,
        RerankerModel,
        rerank_normativa_chunks,
    )

    print("\n" + "=" * 70)
    print("TEST 2: Reranker with Hybrid Search (database)")
    print("=" * 70)

    if not api_key:
        print("[SKIP] OPENROUTER_API_KEY not set, skipping database test")
        return True

    # Connect to database
    try:
        conn = psycopg2.connect(**db_config)
        print(f"[OK] Connected to {db_config['dbname']}@{db_config['host']}:{db_config['port']}")
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return False

    # Initialize reranker
    reranker = CrossEncoderReranker(model=RerankerModel.BGE_M3, batch_size=16)

    results: list[TestResult] = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        keywords = test_case["keywords"]

        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query}")

        # Get embedding
        try:
            embedding = get_embedding_openrouter(query, api_key)
        except Exception as e:
            print(f"  [ERROR] Embedding failed: {e}")
            continue

        # Hybrid search (top 20)
        chunks = hybrid_search_sync(conn, query, embedding, limit=20)

        if not chunks:
            print("  [WARN] No results from hybrid search")
            continue

        print(f"  Hybrid search returned {len(chunks)} chunks")

        # Count keyword hits in top-5 before reranking
        hybrid_texts = [c["text"] for c in chunks[:5]]
        hybrid_hits = count_keyword_hits(hybrid_texts, keywords)

        # Rerank
        start = time.time()
        reranked = await rerank_normativa_chunks(
            query=query,
            chunks=chunks,
            reranker=reranker,
            top_k=10,
            rrf_weight=0.3,
        )
        latency = (time.time() - start) * 1000

        # Count keyword hits in top-5 after reranking
        reranked_texts = [r.text_preview for r in reranked[:5]]
        reranked_hits = count_keyword_hits(reranked_texts, keywords)

        # Count rank changes
        rank_changes = sum(1 for r in reranked if r.new_rank != r.original_rank)

        result = TestResult(
            query=query,
            query_type=test_case["type"],
            hybrid_top5_keywords=hybrid_hits,
            reranked_top5_keywords=reranked_hits,
            rank_changes=rank_changes,
            latency_ms=latency,
            top_rerank_score=reranked[0].rerank_score if reranked else 0,
            improvement=reranked_hits - hybrid_hits,
        )
        results.append(result)

        # Print results
        print(f"  Reranking: {latency:.0f}ms, {rank_changes} rank changes")
        print(f"  Keywords in top-5: {hybrid_hits} -> {reranked_hits} ({'+' if result.improvement >= 0 else ''}{result.improvement})")

        print(f"\n  Top 5 after reranking:")
        for r in reranked[:5]:
            preview = r.text_preview[:60]
            change = "=" if r.new_rank == r.original_rank else f"{r.original_rank}->{r.new_rank}"
            print(f"    {r.new_rank}. [{r.rerank_score:.3f}] {r.work_code} {r.articolo} ({change}) {preview}...")

    conn.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_improvement = sum(r.improvement for r in results) / len(results)
        total_hybrid_hits = sum(r.hybrid_top5_keywords for r in results)
        total_reranked_hits = sum(r.reranked_top5_keywords for r in results)

        print(f"Queries tested: {len(results)}")
        print(f"Avg latency: {avg_latency:.0f}ms")
        print(f"Avg improvement: {avg_improvement:+.2f} keywords in top-5")
        print(f"Total keyword hits: {total_hybrid_hits} -> {total_reranked_hits} ({'+' if total_reranked_hits >= total_hybrid_hits else ''}{total_reranked_hits - total_hybrid_hits})")

    print("\n[OK] Database test completed")
    return True


async def test_model_comparison():
    """Compare different reranker models."""
    from lexe_api.kb.retrieval.reranker import CrossEncoderReranker, RerankerModel

    print("\n" + "=" * 70)
    print("TEST 3: Model Comparison")
    print("=" * 70)

    query = "risarcimento danno per fatto illecito"
    documents = [
        "Art. 2043 - Risarcimento per fatto illecito. Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno.",
        "Art. 2059 - Danni non patrimoniali. Il danno non patrimoniale deve essere risarcito solo nei casi determinati dalla legge.",
        "Art. 1218 - Responsabilita del debitore. Il debitore che non esegue esattamente la prestazione dovuta e tenuto al risarcimento.",
        "Art. 844 - Immissioni. Il proprietario di un fondo non puo impedire le immissioni di fumo o calore.",
        "Art. 1321 - Nozione di contratto. Il contratto e l'accordo tra due o piu parti.",
    ]

    models = [
        ("BGE-M3", RerankerModel.BGE_M3),
        # Add more models to compare:
        # ("mMARCO", RerankerModel.MMARCO_MULTILINGUAL),
    ]

    print(f"\nQuery: {query}")
    print(f"Documents: {len(documents)}")

    for model_name, model_enum in models:
        print(f"\n--- {model_name} ---")

        try:
            start = time.time()
            reranker = CrossEncoderReranker(model=model_enum)
            pairs = [(query, doc) for doc in documents]
            scores = reranker.predict(pairs)
            latency = (time.time() - start) * 1000

            # Rank documents
            ranked = sorted(zip(range(len(documents)), scores), key=lambda x: x[1], reverse=True)

            print(f"Latency: {latency:.0f}ms")
            print("Ranking: ", end="")
            for idx, score in ranked:
                print(f"Doc{idx+1}({score:.2f}) ", end="")
            print()

        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n[OK] Model comparison completed")
    return True


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test Cross-Encoder Reranker")
    parser.add_argument("--staging", action="store_true", help="Use staging database")
    parser.add_argument("--model", choices=["bge-m3", "mmarco", "gte"], default="bge-m3", help="Model to test")
    parser.add_argument("--query", type=str, help="Custom query to test")
    parser.add_argument("--skip-basic", action="store_true", help="Skip basic test")
    parser.add_argument("--skip-db", action="store_true", help="Skip database test")
    args = parser.parse_args()

    print("=" * 70)
    print("LEXE Cross-Encoder Reranker Test")
    print("=" * 70)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    db_config = STAGING_DB if args.staging else LOCAL_DB

    print(f"\nDatabase: {db_config['dbname']}@{db_config['host']}:{db_config['port']}")
    print(f"OpenRouter API: {'configured' if api_key else 'not set'}")

    async def run_tests():
        success = True

        # Test 1: Basic reranker
        if not args.skip_basic:
            try:
                await test_reranker_basic()
            except Exception as e:
                print(f"\n[ERROR] Basic test failed: {e}")
                success = False

        # Test 2: With database
        if not args.skip_db and api_key:
            try:
                await test_reranker_with_database(db_config, api_key)
            except Exception as e:
                print(f"\n[ERROR] Database test failed: {e}")
                import traceback
                traceback.print_exc()
                success = False

        # Test 3: Model comparison
        if not args.skip_basic:
            try:
                await test_model_comparison()
            except Exception as e:
                print(f"\n[ERROR] Model comparison failed: {e}")
                success = False

        return success

    success = asyncio.run(run_tests())

    print("\n" + "=" * 70)
    if success:
        print("[DONE] All tests passed")
    else:
        print("[FAILED] Some tests failed")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
