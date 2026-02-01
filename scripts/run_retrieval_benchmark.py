"""
KB Massimari - Retrieval Benchmark R1/R2

R1: Hybrid (BM25 + Dense HNSW + RRF)
R2: Hybrid + Rerank (bge-reranker-v2-m3)

Usage:
    export OPENROUTER_API_KEY='sk-or-...'
    python scripts/run_retrieval_benchmark.py

    # Solo R1
    python scripts/run_retrieval_benchmark.py --r1-only

    # Verbose
    python scripts/run_retrieval_benchmark.py --verbose
"""
import argparse
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


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

# Config
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MISTRAL_MODEL = "mistralai/mistral-embed-2312"

# RRF config
RRF_K = 60  # Standard RRF constant

# Cohere Rerank config
COHERE_RERANK_MODEL = "rerank-v3.5"

# Benchmark queries - 30 total (balanced)
BENCHMARK_QUERIES = [
    # ISTITUTO - 12 queries (concetti giuridici generali)
    {"type": "istituto", "query": "responsabilita civile per danni", "keywords": ["responsabil", "dann", "civil"]},
    {"type": "istituto", "query": "risarcimento del danno patrimoniale", "keywords": ["risarciment", "dann", "patrimonial"]},
    {"type": "istituto", "query": "nullita del contratto", "keywords": ["null", "contratt"]},
    {"type": "istituto", "query": "prescrizione del diritto", "keywords": ["prescrizion", "diritt"]},
    {"type": "istituto", "query": "successione ereditaria legittima", "keywords": ["succession", "ereditar", "legittim"]},
    {"type": "istituto", "query": "fallimento e procedure concorsuali", "keywords": ["falliment", "concorsual", "procedur"]},
    {"type": "istituto", "query": "licenziamento per giusta causa", "keywords": ["licenziament", "giusta", "causa"]},
    {"type": "istituto", "query": "reato di truffa elementi costitutivi", "keywords": ["truff", "reat", "element"]},
    {"type": "istituto", "query": "omicidio colposo presupposti", "keywords": ["omicid", "colpos"]},
    {"type": "istituto", "query": "concorso di persone nel reato", "keywords": ["concors", "person", "reat"]},
    {"type": "istituto", "query": "misure cautelari personali", "keywords": ["misur", "cautelar", "personal"]},
    {"type": "istituto", "query": "appello nel processo civile", "keywords": ["appell", "process", "civil"]},

    # AVVERSARIA - 12 queries (negazioni, casi limite, formulazioni ingannevoli)
    {"type": "avversaria", "query": "quando NON sussiste responsabilita", "keywords": ["responsabil", "sussist"]},
    {"type": "avversaria", "query": "esclusione del dolo nel reato", "keywords": ["dol", "reat", "esclus"]},
    {"type": "avversaria", "query": "inammissibilita del ricorso cassazione", "keywords": ["inammissibil", "ricors", "cassazion"]},
    {"type": "avversaria", "query": "rigetto della domanda risarcitoria", "keywords": ["rigett", "risarcitor"]},
    {"type": "avversaria", "query": "mancanza di legittimazione attiva", "keywords": ["legittimaz", "attiv", "mancanz"]},
    {"type": "avversaria", "query": "improcedibilita della querela", "keywords": ["improcedibil", "querel"]},
    {"type": "avversaria", "query": "assenza di nesso causale", "keywords": ["nesso", "causal", "assenz"]},
    {"type": "avversaria", "query": "insussistenza del fatto contestato", "keywords": ["insussistenz", "fatt", "contestat"]},
    {"type": "avversaria", "query": "non punibilita per particolare tenuita", "keywords": ["punibil", "tenu"]},
    {"type": "avversaria", "query": "difetto di motivazione sentenza", "keywords": ["difett", "motivazion", "sentenz"]},
    {"type": "avversaria", "query": "prescrizione maturata durante processo", "keywords": ["prescrizion", "maturat", "process"]},
    {"type": "avversaria", "query": "incompetenza territoriale del giudice", "keywords": ["incompetenz", "territorial", "giudic"]},

    # CITAZIONE - 6 queries (riferimenti normativi specifici)
    {"type": "citazione", "query": "art. 2043 codice civile", "keywords": ["2043", "civil"]},
    {"type": "citazione", "query": "art. 640 codice penale truffa", "keywords": ["640", "penal", "truff"]},
    {"type": "citazione", "query": "art. 575 codice penale omicidio", "keywords": ["575", "penal", "omicid"]},
    {"type": "citazione", "query": "art. 1218 responsabilita contrattuale", "keywords": ["1218", "contrattual"]},
    {"type": "citazione", "query": "art. 337 cpp resistenza pubblico ufficiale", "keywords": ["337", "resistenz", "pubblic"]},
    {"type": "citazione", "query": "legge fallimentare art. 67", "keywords": ["67", "fallimentar"]},
]


@dataclass
class QueryResult:
    query_type: str
    query: str
    method: str  # R1 or R2
    top_k: list[str]  # massima_id list
    scores: list[float]
    keyword_hits: int
    keyword_total: int
    latency_ms: float


@dataclass
class BenchmarkSummary:
    method: str
    total_queries: int
    avg_keyword_hits: float
    avg_latency_ms: float
    by_type: dict  # type -> avg_hits


async def get_embedding_openrouter(
    client: httpx.AsyncClient,
    text: str,
) -> tuple[list[float] | None, float, str | None]:
    """Get embedding from OpenRouter API."""
    start = time.time()

    try:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "input": [text],
            },
            timeout=30.0,
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            return None, latency, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()
        embedding = data["data"][0]["embedding"]
        return embedding, latency, None

    except Exception as e:
        latency = (time.time() - start) * 1000
        return None, latency, str(e)


def embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def r1_hybrid_search(
    conn: asyncpg.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 20,
) -> tuple[list[tuple[UUID, float]], float]:
    """
    R1: Hybrid search (BM25 + Dense HNSW + RRF fusion)
    Returns: list of (massima_id, rrf_score), latency_ms
    """
    start = time.time()

    emb_str = embedding_to_pgvector(query_embedding)

    rows = await conn.fetch("""
        WITH dense_results AS (
            SELECT
                e.massima_id,
                1 - (e.embedding <=> $1::vector) as similarity,
                ROW_NUMBER() OVER (ORDER BY e.embedding <=> $1::vector) as rank
            FROM kb.emb_mistral e
            ORDER BY e.embedding <=> $1::vector
            LIMIT 50
        ),
        sparse_results AS (
            SELECT
                m.id as massima_id,
                ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', $2)) as score,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', $2)) DESC
                ) as rank
            FROM kb.massime m
            WHERE m.tsv_italian @@ plainto_tsquery('italian', $2)
               OR m.tsv_simple @@ plainto_tsquery('simple', $2)
            LIMIT 50
        ),
        combined AS (
            SELECT
                COALESCE(d.massima_id, s.massima_id) as massima_id,
                COALESCE(1.0 / ($3 + d.rank), 0) +
                COALESCE(1.0 / ($3 + s.rank), 0) as rrf_score,
                d.rank as dense_rank,
                s.rank as sparse_rank
            FROM dense_results d
            FULL OUTER JOIN sparse_results s ON d.massima_id = s.massima_id
        )
        SELECT
            c.massima_id,
            c.rrf_score
        FROM combined c
        ORDER BY c.rrf_score DESC
        LIMIT $4
    """, emb_str, query_text, RRF_K, limit)

    latency = (time.time() - start) * 1000
    results = [(row["massima_id"], row["rrf_score"]) for row in rows]
    return results, latency


async def cohere_rerank(
    http_client: httpx.AsyncClient,
    query: str,
    documents: list[str],
    top_n: int = 20,
) -> tuple[list[tuple[int, float]], float]:
    """Cohere Rerank 3.5 API call."""
    start = time.time()

    try:
        response = await http_client.post(
            "https://api.cohere.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": COHERE_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=30.0,
        )

        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            return [], latency

        data = response.json()
        results = [(r["index"], r["relevance_score"]) for r in data["results"]]
        return results, latency

    except Exception:
        latency = (time.time() - start) * 1000
        return [], latency


async def r2_hybrid_rerank(
    conn: asyncpg.Connection,
    http_client: httpx.AsyncClient,
    query_text: str,
    query_embedding: list[float],
    limit: int = 20,
) -> tuple[list[tuple[UUID, float]], float]:
    """
    R2: Hybrid + Rerank with Cohere rerank-v3.5
    First gets R1 top-30, then reranks with Cohere cross-encoder.
    """
    start = time.time()

    # Get R1 top-30 candidates
    r1_results, r1_latency = await r1_hybrid_search(
        conn, query_text, query_embedding, limit=30
    )

    if not r1_results:
        return [], r1_latency

    # Get massima texts for reranking
    massima_ids = [str(mid) for mid, _ in r1_results]

    rows = await conn.fetch("""
        SELECT id, testo FROM kb.massime
        WHERE id = ANY($1::uuid[])
    """, massima_ids)

    id_to_text = {row["id"]: row["testo"] for row in rows}

    # Prepare documents in R1 order
    r1_order = [(mid, id_to_text.get(mid, "")[:2000]) for mid, _ in r1_results]
    documents = [text for _, text in r1_order]

    # Call Cohere Rerank
    rerank_results, rerank_latency = await cohere_rerank(
        http_client, query_text, documents, top_n=limit
    )

    if not rerank_results:
        # Fallback to R1 order if Cohere fails
        latency = (time.time() - start) * 1000
        return r1_results[:limit], latency

    # Map back to massima IDs
    reranked = []
    for idx, score in rerank_results:
        mid = r1_results[idx][0]
        reranked.append((mid, score))

    latency = (time.time() - start) * 1000
    return reranked[:limit], latency


def check_keyword_hits(
    massima_texts: dict[UUID, str],
    result_ids: list[UUID],
    keywords: list[str],
    top_k: int = 5,
) -> int:
    """Count how many of top-K results contain at least one keyword."""
    hits = 0
    for mid in result_ids[:top_k]:
        text = massima_texts.get(mid, "").lower()
        for kw in keywords:
            if kw.lower() in text:
                hits += 1
                break
    return hits


async def run_benchmark(r1_only: bool = False, verbose: bool = False):
    """Run the full retrieval benchmark."""
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return

    if not r1_only and not COHERE_API_KEY:
        print("ERROR: Set COHERE_API_KEY for R2 reranking")
        print("  Or use --r1-only to skip R2")
        return

    print("=" * 70)
    print("KB MASSIMARI - RETRIEVAL BENCHMARK R1/R2")
    print("=" * 70)
    print(f"Queries: {len(BENCHMARK_QUERIES)}")
    print(f"Methods: R1 Hybrid" + ("" if r1_only else " + R2 Hybrid+Cohere"))
    if not r1_only:
        print(f"Reranker: Cohere {COHERE_RERANK_MODEL}")
    print()

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Preload all massima texts for keyword checking
    rows = await conn.fetch("SELECT id, testo FROM kb.massime")
    massima_texts = {row["id"]: row["testo"] for row in rows}
    print(f"[OK] Loaded {len(massima_texts)} massime texts")
    print()

    all_results: list[QueryResult] = []

    async with httpx.AsyncClient() as http_client:
        for i, q in enumerate(BENCHMARK_QUERIES, 1):
            print(f"[{i}/{len(BENCHMARK_QUERIES)}] {q['type']}: {q['query'][:40]}...")

            # Get query embedding
            embedding, emb_latency, error = await get_embedding_openrouter(
                http_client, q["query"]
            )

            if error:
                print(f"  ERROR getting embedding: {error}")
                continue

            # R1: Hybrid
            r1_results, r1_latency = await r1_hybrid_search(
                conn, q["query"], embedding, limit=20
            )

            r1_ids = [mid for mid, _ in r1_results]
            r1_scores = [score for _, score in r1_results]
            r1_hits = check_keyword_hits(massima_texts, r1_ids, q["keywords"])

            all_results.append(QueryResult(
                query_type=q["type"],
                query=q["query"],
                method="R1",
                top_k=[str(mid)[:8] for mid in r1_ids[:5]],
                scores=r1_scores[:5],
                keyword_hits=r1_hits,
                keyword_total=5,
                latency_ms=r1_latency + emb_latency,
            ))

            if verbose:
                print(f"  R1: {r1_hits}/5 hits, {r1_latency:.0f}ms")

            # R2: Hybrid + Rerank
            if not r1_only:
                r2_results, r2_latency = await r2_hybrid_rerank(
                    conn, http_client, q["query"], embedding, limit=20
                )

                r2_ids = [mid for mid, _ in r2_results]
                r2_scores = [score for _, score in r2_results]
                r2_hits = check_keyword_hits(massima_texts, r2_ids, q["keywords"])

                all_results.append(QueryResult(
                    query_type=q["type"],
                    query=q["query"],
                    method="R2",
                    top_k=[str(mid)[:8] for mid in r2_ids[:5]],
                    scores=r2_scores[:5],
                    keyword_hits=r2_hits,
                    keyword_total=5,
                    latency_ms=r2_latency + emb_latency,
                ))

                if verbose:
                    print(f"  R2: {r2_hits}/5 hits, {r2_latency:.0f}ms")

            # Rate limit
            await asyncio.sleep(0.3)

    await conn.close()

    # Generate summary
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for method in ["R1", "R2"] if not r1_only else ["R1"]:
        method_results = [r for r in all_results if r.method == method]

        if not method_results:
            continue

        total_hits = sum(r.keyword_hits for r in method_results)
        total_possible = sum(r.keyword_total for r in method_results)
        avg_hits = total_hits / len(method_results)
        avg_latency = sum(r.latency_ms for r in method_results) / len(method_results)

        print(f"\n### {method}: {'Hybrid' if method == 'R1' else 'Hybrid + Rerank'}")
        print(f"Total queries: {len(method_results)}")
        print(f"Avg keyword hits: {avg_hits:.2f}/5 ({total_hits}/{total_possible})")
        print(f"Avg latency: {avg_latency:.0f}ms")

        # By type
        print("\nBy query type:")
        for qtype in ["istituto", "avversaria", "citazione"]:
            type_results = [r for r in method_results if r.query_type == qtype]
            if type_results:
                type_hits = sum(r.keyword_hits for r in type_results) / len(type_results)
                type_latency = sum(r.latency_ms for r in type_results) / len(type_results)
                print(f"  {qtype}: {type_hits:.2f}/5 hits, {type_latency:.0f}ms")

    # Comparison R1 vs R2
    if not r1_only:
        print("\n### R1 vs R2 Comparison")
        r1_results = [r for r in all_results if r.method == "R1"]
        r2_results = [r for r in all_results if r.method == "R2"]

        r1_avg = sum(r.keyword_hits for r in r1_results) / len(r1_results)
        r2_avg = sum(r.keyword_hits for r in r2_results) / len(r2_results)

        improvement = (r2_avg - r1_avg) / r1_avg * 100 if r1_avg > 0 else 0

        print(f"R1 avg hits: {r1_avg:.2f}/5")
        print(f"R2 avg hits: {r2_avg:.2f}/5")
        print(f"Improvement: {improvement:+.1f}%")

        r1_lat = sum(r.latency_ms for r in r1_results) / len(r1_results)
        r2_lat = sum(r.latency_ms for r in r2_results) / len(r2_results)
        lat_overhead = (r2_lat - r1_lat) / r1_lat * 100

        print(f"R1 avg latency: {r1_lat:.0f}ms")
        print(f"R2 avg latency: {r2_lat:.0f}ms")
        print(f"Latency overhead: {lat_overhead:+.1f}%")

    # Save results
    output_path = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/retrieval_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "rrf_k": RRF_K,
            "embedding_model": MISTRAL_MODEL,
            "total_queries": len(BENCHMARK_QUERIES),
        },
        "results": [asdict(r) for r in all_results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)

    print(f"\n[OK] Results saved to {output_path}")
    print("\n[DONE]")


def main():
    parser = argparse.ArgumentParser(description="Run KB retrieval benchmark")
    parser.add_argument("--r1-only", action="store_true", help="Only run R1 Hybrid")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    asyncio.run(run_benchmark(r1_only=args.r1_only, verbose=args.verbose))


if __name__ == "__main__":
    main()
