"""
Run Retrieval Evaluation on Golden Set

Executes golden set queries, measures metrics, logs results.

Search Modes:
- dense: Pure vector similarity (cosine)
- hybrid: Router + Dense + Sparse + RRF fusion

Router Logic:
1. Citation queries (Rv./Sez./n.) → Direct DB lookup
2. Semantic queries → Hybrid (dense + sparse with RRF)

Metrics:
- Recall@K: % of queries where expected doc is in top-K
- MRR (Mean Reciprocal Rank): 1/rank of first correct result
- nDCG@K: Normalized Discounted Cumulative Gain
- Latency p50, p95, p99

Logging:
- JSONL: Per-query results with top-K, mode, scores
- CSV: Summary aggregated metrics
- DB: retrieval_logs table

Usage:
    uv run python scripts/qa/run_retrieval_eval.py --top-k 10 --mode hybrid
    uv run python scripts/qa/run_retrieval_eval.py --top-k 10 --mode hybrid --log-results
    uv run python scripts/qa/run_retrieval_eval.py --top-k 10 --mode hybrid --log-results --log-dir retrieval_logs --tag hybrid_v1
"""

import argparse
import asyncio
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import asyncpg
import httpx

# ============================================================
# Configuration
# ============================================================

DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_ID = "openai/text-embedding-3-small"
EXPECTED_DIM = 1536

# RRF fusion constant
RRF_K = 60


@dataclass
class ResultItem:
    """Single result item with source tracking."""
    massima_id: str
    score: float
    rank: int
    source: str  # dense | sparse | lookup | rrf


@dataclass
class EvalResult:
    """Per-query evaluation result with full detail."""
    query_id: str
    query_type: str
    query_text: str
    expected_id: str
    result_ids: list[str]
    result_scores: list[float]
    hit_at_k: int | None  # Position (1-indexed) or None if miss
    latency_ms: int
    mode: str = "dense"  # citation_lookup | hybrid_rrf | dense | sparse_only
    debug: dict = field(default_factory=dict)  # router_hit, dense_k, sparse_k, etc.

    def to_jsonl_dict(self, top_k: int) -> dict:
        """Convert to JSONL-friendly dict."""
        # Calculate per-query metrics
        hit = self.hit_at_k is not None
        rr = 1.0 / self.hit_at_k if hit else 0.0
        ndcg = 1.0 / math.log2(self.hit_at_k + 1) if hit else 0.0

        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "query_text": self.query_text[:300],  # Truncate for storage
            "mode": self.mode,
            "ground_truth_id": self.expected_id,
            "top_k": top_k,
            "results": [
                {"massima_id": mid, "score": round(float(score), 6), "rank": i + 1}
                for i, (mid, score) in enumerate(zip(self.result_ids, self.result_scores))
            ],
            "metrics": {
                "hit": hit,
                "rank": self.hit_at_k,
                "recall_at_k": 1 if hit else 0,
                "rr": round(rr, 4),
                "ndcg_at_k": round(ndcg, 4),
            },
            "latency_ms": self.latency_ms,
            "debug": self.debug,
        }


@dataclass
class EvalMetrics:
    total_queries: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # By type
    self_recall: float = 0.0
    self_mrr: float = 0.0
    citation_recall: float = 0.0
    citation_mrr: float = 0.0

    # Errors
    empty_results: int = 0
    api_errors: int = 0


# ============================================================
# Citation Parsing
# ============================================================

# Patterns for citation detection
RV_PATTERN = re.compile(r'[Rr]v\.?\s*(\d{5,7}(?:-\d+)?)', re.IGNORECASE)
SEZ_NUM_ANNO_PATTERN = re.compile(
    r'[Ss]ez\.?\s*([A-Za-z0-9]+).*?[Nn]\.?\s*(\d+)(?:/(\d{4}))?',
    re.IGNORECASE | re.DOTALL
)


def parse_citation(query: str) -> dict | None:
    """
    Parse citation from query text.
    Returns dict with parsed fields or None if not a citation query.
    """
    result = {}

    # Try RV pattern first (most specific)
    rv_match = RV_PATTERN.search(query)
    if rv_match:
        result["rv"] = rv_match.group(1)

    # Try Sez + n. + anno pattern
    sez_match = SEZ_NUM_ANNO_PATTERN.search(query)
    if sez_match:
        result["sezione"] = sez_match.group(1).upper()
        result["numero"] = sez_match.group(2)
        if sez_match.group(3):
            result["anno"] = int(sez_match.group(3))

    return result if result else None


# ============================================================
# Citation Lookup (Router)
# ============================================================

async def citation_lookup(
    conn: asyncpg.Connection,
    citation: dict,
    top_k: int,
) -> list[dict]:
    """
    Direct database lookup for citation queries.
    Tries RV first, then Sez+Num+Anno.
    """
    results = []

    # Strategy 1: RV lookup (most precise)
    if "rv" in citation:
        rows = await conn.fetch("""
            SELECT m.id as massima_id, m.document_id, m.anno, m.ingest_batch_id,
                   1.0 as score
            FROM kb.massime m
            WHERE m.is_active = TRUE
            AND m.rv = $1
            LIMIT $2
        """, citation["rv"], top_k)
        results = [dict(row) for row in rows]

    # Strategy 2: Sez + Num + Anno (if RV not found)
    if not results and "sezione" in citation and "numero" in citation:
        if "anno" in citation:
            rows = await conn.fetch("""
                SELECT m.id as massima_id, m.document_id, m.anno, m.ingest_batch_id,
                       0.95 as score
                FROM kb.massime m
                WHERE m.is_active = TRUE
                AND UPPER(m.sezione) = $1
                AND m.numero = $2
                AND m.anno = $3
                LIMIT $4
            """, citation["sezione"], citation["numero"], citation["anno"], top_k)
        else:
            rows = await conn.fetch("""
                SELECT m.id as massima_id, m.document_id, m.anno, m.ingest_batch_id,
                       0.90 as score
                FROM kb.massime m
                WHERE m.is_active = TRUE
                AND UPPER(m.sezione) = $1
                AND m.numero = $2
                LIMIT $3
            """, citation["sezione"], citation["numero"], top_k)
        results = [dict(row) for row in rows]

    return results


# ============================================================
# Dense Search (Vector)
# ============================================================

async def dense_search(
    conn: asyncpg.Connection,
    query_embedding: list[float],
    limit: int,
) -> list[tuple[str, float, int]]:
    """
    Pure vector similarity search.
    Returns list of (massima_id, score, rank).
    """
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    rows = await conn.fetch("""
        SELECT
            e.massima_id,
            1 - (e.embedding <=> $1::vector) as score
        FROM kb.embeddings e
        JOIN kb.massime m ON m.id = e.massima_id
        WHERE m.is_active = TRUE
        ORDER BY e.embedding <=> $1::vector
        LIMIT $2
    """, vec_str, limit)

    return [(str(row["massima_id"]), row["score"], i + 1) for i, row in enumerate(rows)]


# ============================================================
# Sparse Search (BM25 via tsvector)
# ============================================================

async def sparse_search(
    conn: asyncpg.Connection,
    query_text: str,
    limit: int,
) -> list[tuple[str, float, int]]:
    """
    Full-text search using tsvector (BM25-style ranking).
    Returns list of (massima_id, score, rank).
    """
    # Build tsquery from query text
    # Remove special characters, keep only words
    clean_query = re.sub(r'[^\w\s]', ' ', query_text)
    words = [w.strip() for w in clean_query.split() if len(w.strip()) > 2]

    if not words:
        return []

    # Use OR for flexibility (| in tsquery)
    tsquery = " | ".join(words[:10])  # Limit to 10 terms

    rows = await conn.fetch("""
        SELECT
            m.id as massima_id,
            ts_rank_cd(m.tsv_italian, query) as score
        FROM kb.massime m, plainto_tsquery('italian', $1) query
        WHERE m.is_active = TRUE
        AND m.tsv_italian @@ query
        ORDER BY ts_rank_cd(m.tsv_italian, query) DESC
        LIMIT $2
    """, tsquery, limit)

    return [(str(row["massima_id"]), row["score"], i + 1) for i, row in enumerate(rows)]


# ============================================================
# RRF Fusion
# ============================================================

def rrf_fusion(
    dense_results: list[tuple[str, float, int]],
    sparse_results: list[tuple[str, float, int]],
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion of dense and sparse results.

    RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)

    Returns sorted list of (massima_id, rrf_score).
    """
    scores = {}

    # Process dense results
    for massima_id, _, rank in dense_results:
        scores[massima_id] = scores.get(massima_id, 0) + 1.0 / (rrf_k + rank)

    # Process sparse results
    for massima_id, _, rank in sparse_results:
        scores[massima_id] = scores.get(massima_id, 0) + 1.0 / (rrf_k + rank)

    # Sort by RRF score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:top_k]


# ============================================================
# Hybrid Search (Main Function)
# ============================================================

async def search_hybrid(
    conn: asyncpg.Connection,
    query_text: str,
    query_embedding: list[float] | None,
    top_k: int,
    dense_k: int = 50,
    sparse_k: int = 50,
) -> tuple[list[dict], int, str, dict]:
    """
    Hybrid search with router logic.

    Returns: (results, latency_ms, retrieval_mode, debug_info)
    """
    start = time.time()
    debug = {"dense_k": dense_k, "sparse_k": sparse_k, "rrf_k": RRF_K}

    # Step 1: Try citation lookup (router)
    citation = parse_citation(query_text)
    if citation:
        debug["router_hit"] = True
        debug["citation_parsed"] = citation
        results = await citation_lookup(conn, citation, top_k)
        if results:
            latency_ms = int((time.time() - start) * 1000)
            debug["lookup_results"] = len(results)
            return results, latency_ms, "citation_lookup", debug

    debug["router_hit"] = False

    # Step 2: Hybrid search (dense + sparse + RRF)
    if query_embedding is None:
        # Fallback to sparse only
        sparse_results = await sparse_search(conn, query_text, top_k)
        results = [
            {"massima_id": mid, "score": score}
            for mid, score, _ in sparse_results[:top_k]
        ]
        latency_ms = int((time.time() - start) * 1000)
        debug["sparse_results"] = len(sparse_results)
        return results, latency_ms, "sparse_only", debug

    # Get both dense and sparse results
    dense_results = await dense_search(conn, query_embedding, dense_k)
    sparse_results = await sparse_search(conn, query_text, sparse_k)

    debug["dense_results"] = len(dense_results)
    debug["sparse_results"] = len(sparse_results)

    # RRF fusion
    fused = rrf_fusion(dense_results, sparse_results, top_k)

    results = [
        {"massima_id": mid, "score": score}
        for mid, score in fused
    ]

    latency_ms = int((time.time() - start) * 1000)
    return results, latency_ms, "hybrid_rrf", debug


# ============================================================
# Embedding API
# ============================================================

async def get_query_embedding(
    client: httpx.AsyncClient,
    query: str,
) -> tuple[list[float] | None, str | None]:
    """Get embedding for a single query."""

    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY not set"

    try:
        response = await client.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_ID,
                "input": [query],
            },
            timeout=30.0,
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        data = response.json()
        embedding = data["data"][0]["embedding"]
        return embedding, None

    except Exception as e:
        return None, str(e)


# ============================================================
# Dense-Only Search (Legacy)
# ============================================================

async def search_dense_only(
    conn: asyncpg.Connection,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[dict], int]:
    """Search for similar massime using cosine similarity only."""

    start = time.time()

    # Build vector string
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # Cosine similarity search (1 - cosine distance)
    rows = await conn.fetch("""
        SELECT
            e.massima_id,
            m.document_id,
            m.anno,
            m.ingest_batch_id,
            1 - (e.embedding <=> $1::vector) as score
        FROM kb.embeddings e
        JOIN kb.massime m ON m.id = e.massima_id
        WHERE m.is_active = TRUE
        ORDER BY e.embedding <=> $1::vector
        LIMIT $2
    """, vec_str, top_k)

    latency_ms = int((time.time() - start) * 1000)

    results = [dict(row) for row in rows]
    return results, latency_ms


# ============================================================
# Metrics Calculation
# ============================================================

def calculate_mrr(results: list[EvalResult]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not results:
        return 0.0

    reciprocal_ranks = []
    for r in results:
        if r.hit_at_k:
            reciprocal_ranks.append(1.0 / r.hit_at_k)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_recall_at_k(results: list[EvalResult]) -> float:
    """Calculate Recall@K (% of queries with hit in top-K)."""
    if not results:
        return 0.0

    hits = sum(1 for r in results if r.hit_at_k is not None)
    return hits / len(results)


def calculate_ndcg_at_k(results: list[EvalResult], k: int) -> float:
    """Calculate nDCG@K (simplified: 1 doc is relevant per query)."""
    if not results:
        return 0.0

    ndcg_sum = 0.0
    for r in results:
        if r.hit_at_k:
            # DCG = 1 / log2(rank + 1)
            dcg = 1.0 / math.log2(r.hit_at_k + 1)
            # Ideal DCG = 1 / log2(2) = 1.0 (if hit at position 1)
            idcg = 1.0
            ndcg_sum += dcg / idcg

    return ndcg_sum / len(results)


def calculate_latency_percentiles(latencies: list[int]) -> tuple[float, float, float]:
    """Calculate p50, p95, p99 latencies."""
    if not latencies:
        return 0.0, 0.0, 0.0

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    p50 = sorted_lat[int(n * 0.50)]
    p95 = sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1]
    p99 = sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1]

    return float(p50), float(p95), float(p99)


# ============================================================
# File Logging
# ============================================================

def setup_log_files(log_dir: str, tag: str) -> tuple[Path, Path]:
    """Create log directory and return file paths."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_file = log_path / f"eval_{timestamp}_{tag}.jsonl"
    csv_file = log_path / f"summary_{timestamp}_{tag}.csv"

    return jsonl_file, csv_file


def write_csv_summary(
    csv_file: Path,
    metrics: "EvalMetrics",
    mode_counts: dict,
    search_mode: str,
    top_k: int,
    tag: str,
):
    """Write summary CSV with all metrics."""
    rows = [
        # Overall
        {"category": "overall", "metric": "total_queries", "value": metrics.total_queries},
        {"category": "overall", "metric": f"recall_at_{top_k}", "value": round(metrics.recall_at_k, 4)},
        {"category": "overall", "metric": "mrr", "value": round(metrics.mrr, 4)},
        {"category": "overall", "metric": f"ndcg_at_{top_k}", "value": round(metrics.ndcg_at_k, 4)},
        {"category": "overall", "metric": "latency_p50_ms", "value": metrics.latency_p50},
        {"category": "overall", "metric": "latency_p95_ms", "value": metrics.latency_p95},
        {"category": "overall", "metric": "latency_p99_ms", "value": metrics.latency_p99},
        # By type
        {"category": "self", "metric": f"recall_at_{top_k}", "value": round(metrics.self_recall, 4)},
        {"category": "self", "metric": "mrr", "value": round(metrics.self_mrr, 4)},
        {"category": "citation", "metric": f"recall_at_{top_k}", "value": round(metrics.citation_recall, 4)},
        {"category": "citation", "metric": "mrr", "value": round(metrics.citation_mrr, 4)},
        # Errors
        {"category": "errors", "metric": "api_errors", "value": metrics.api_errors},
        {"category": "errors", "metric": "empty_results", "value": metrics.empty_results},
    ]

    # Add mode distribution
    for mode, count in mode_counts.items():
        if count > 0:
            rows.append({"category": "mode_distribution", "metric": mode, "value": count})

    # Metadata
    rows.append({"category": "meta", "metric": "search_mode", "value": search_mode})
    rows.append({"category": "meta", "metric": "top_k", "value": top_k})
    rows.append({"category": "meta", "metric": "tag", "value": tag})
    rows.append({"category": "meta", "metric": "timestamp", "value": datetime.now().isoformat()})

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main Evaluation
# ============================================================

async def run_evaluation(
    top_k: int,
    log_results: bool,
    search_mode: str = "dense",
    log_dir: str | None = None,
    tag: str = "default",
    verbose: bool = False,
):
    """Run retrieval evaluation on golden set."""

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Top-K:         {top_k}")
    print(f"Search mode:   {search_mode}")
    print(f"Log results:   {log_results}")
    if log_dir:
        print(f"Log dir:       {log_dir}")
        print(f"Tag:           {tag}")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("\n[ERROR] OPENROUTER_API_KEY not set!")
        return

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Fetch golden queries
    rows = await conn.fetch("""
        SELECT id, query_text, query_type, expected_massima_id
        FROM kb.golden_queries
        WHERE is_active = TRUE
        ORDER BY query_type, id
    """)

    if not rows:
        print("\n[ERROR] No golden queries found! Run generate_golden_set.py first.")
        await conn.close()
        return

    print(f"[OK] Loaded {len(rows)} golden queries")

    # Count by type
    by_type = {}
    for row in rows:
        by_type[row["query_type"]] = by_type.get(row["query_type"], 0) + 1
    print(f"    Distribution: {by_type}")

    # Setup file logging if requested
    jsonl_file = None
    jsonl_f = None
    if log_results and log_dir:
        jsonl_file, csv_file = setup_log_files(log_dir, tag)
        jsonl_f = open(jsonl_file, "w", encoding="utf-8")
        print(f"[OK] JSONL log: {jsonl_file}")

    # Run evaluation
    results: list[EvalResult] = []
    latencies: list[int] = []
    metrics = EvalMetrics()
    mode_counts = {"citation_lookup": 0, "hybrid_rrf": 0, "sparse_only": 0, "dense": 0}

    async with httpx.AsyncClient() as client:
        for i, row in enumerate(rows):
            if (i + 1) % 50 == 0:
                print(f"  Processing query {i + 1}/{len(rows)}...")

            query_text = row["query_text"]

            # Get query embedding (needed for both modes)
            embedding, error = await get_query_embedding(client, query_text)

            if error:
                metrics.api_errors += 1
                continue

            # Search based on mode
            debug_info = {}
            if search_mode == "hybrid":
                search_results, latency_ms, actual_mode, debug_info = await search_hybrid(
                    conn, query_text, embedding, top_k
                )
                mode_counts[actual_mode] = mode_counts.get(actual_mode, 0) + 1
            else:
                search_results, latency_ms = await search_dense_only(conn, embedding, top_k)
                actual_mode = "dense"
                mode_counts["dense"] += 1

            latencies.append(latency_ms)

            if not search_results:
                metrics.empty_results += 1

            # Check hit
            result_ids = [str(r["massima_id"]) for r in search_results]
            result_scores = [r["score"] for r in search_results]
            expected_id = str(row["expected_massima_id"])

            hit_at_k = None
            if expected_id in result_ids:
                hit_at_k = result_ids.index(expected_id) + 1  # 1-indexed

            result = EvalResult(
                query_id=str(row["id"]),
                query_type=row["query_type"],
                query_text=query_text,
                expected_id=expected_id,
                result_ids=result_ids,
                result_scores=result_scores,
                hit_at_k=hit_at_k,
                latency_ms=latency_ms,
                mode=actual_mode,
                debug=debug_info if verbose else {},
            )
            results.append(result)

            # Write JSONL per-query
            if jsonl_f:
                jsonl_row = result.to_jsonl_dict(top_k)
                jsonl_f.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")

            # Rate limiting
            await asyncio.sleep(0.1)

    # Close JSONL file
    if jsonl_f:
        jsonl_f.close()
        print(f"[OK] Wrote {len(results)} queries to JSONL")

    # Print mode distribution
    print(f"\n### Search Mode Distribution")
    for mode, count in mode_counts.items():
        if count > 0:
            print(f"  {mode}: {count}")

    # Calculate metrics
    metrics.total_queries = len(results)
    metrics.recall_at_k = calculate_recall_at_k(results)
    metrics.mrr = calculate_mrr(results)
    metrics.ndcg_at_k = calculate_ndcg_at_k(results, top_k)
    metrics.latency_p50, metrics.latency_p95, metrics.latency_p99 = calculate_latency_percentiles(latencies)

    # By type
    self_results = [r for r in results if r.query_type == "self"]
    citation_results = [r for r in results if r.query_type == "citation"]

    metrics.self_recall = calculate_recall_at_k(self_results)
    metrics.self_mrr = calculate_mrr(self_results)
    metrics.citation_recall = calculate_recall_at_k(citation_results)
    metrics.citation_mrr = calculate_mrr(citation_results)

    # Write CSV summary if log_dir specified
    if log_results and log_dir:
        write_csv_summary(csv_file, metrics, mode_counts, search_mode, top_k, tag)
        print(f"[OK] Summary CSV: {csv_file}")

    # Log results to DB if requested
    if log_results:
        print("\nLogging results to retrieval_logs...")
        for r in results:
            await conn.execute("""
                INSERT INTO kb.retrieval_logs
                (query_text, retrieval_mode, top_k, latency_ms, result_count,
                 result_ids, result_scores, is_golden_set, golden_set_type,
                 expected_id, hit_at_k)
                VALUES ($1, $2, $3, $4, $5, $6::uuid[], $7::float[],
                        TRUE, $8, $9::uuid, $10)
            """,
                r.query_text[:500],
                r.mode,  # Use actual mode per-query, not global search_mode
                top_k,
                r.latency_ms,
                len(r.result_ids),
                r.result_ids,
                r.result_scores,
                r.query_type,
                r.expected_id,
                r.hit_at_k,
            )
        print(f"[OK] Logged {len(results)} results to DB")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n### Overall Metrics (K={top_k})")
    print(f"| Metric | Value | Target |")
    print(f"|--------|-------|--------|")
    print(f"| Recall@{top_k} | {metrics.recall_at_k:.1%} | >= 75% |")
    print(f"| MRR | {metrics.mrr:.3f} | >= 0.55 |")
    print(f"| nDCG@{top_k} | {metrics.ndcg_at_k:.3f} | - |")
    print(f"| Latency p50 | {metrics.latency_p50:.0f}ms | - |")
    print(f"| Latency p95 | {metrics.latency_p95:.0f}ms | <= 500ms |")

    print(f"\n### By Query Type")
    print(f"| Type | Recall@{top_k} | MRR | Count |")
    print(f"|------|---------|-----|-------|")
    print(f"| Self | {metrics.self_recall:.1%} | {metrics.self_mrr:.3f} | {len(self_results)} |")
    print(f"| Citation | {metrics.citation_recall:.1%} | {metrics.citation_mrr:.3f} | {len(citation_results)} |")

    print(f"\n### Errors")
    print(f"| Type | Count |")
    print(f"|------|-------|")
    print(f"| API errors | {metrics.api_errors} |")
    print(f"| Empty results | {metrics.empty_results} |")

    # GO/NO-GO
    print("\n### GO/NO-GO")
    go = True

    if metrics.self_recall < 0.75:
        print(f"[FAIL] Self Recall@{top_k} = {metrics.self_recall:.1%} < 75%")
        go = False
    else:
        print(f"[PASS] Self Recall@{top_k} = {metrics.self_recall:.1%} >= 75%")

    if metrics.self_mrr < 0.55:
        print(f"[WARN] Self MRR = {metrics.self_mrr:.3f} < 0.55")
    else:
        print(f"[PASS] Self MRR = {metrics.self_mrr:.3f} >= 0.55")

    if metrics.citation_recall < 0.65:
        print(f"[WARN] Citation Recall@{top_k} = {metrics.citation_recall:.1%} < 65%")
    else:
        print(f"[PASS] Citation Recall@{top_k} = {metrics.citation_recall:.1%} >= 65%")

    if metrics.latency_p95 > 500:
        print(f"[WARN] Latency p95 = {metrics.latency_p95:.0f}ms > 500ms")
    else:
        print(f"[PASS] Latency p95 = {metrics.latency_p95:.0f}ms <= 500ms")

    print("\n" + ("=" * 70))
    print(f"VERDICT: {'GO' if go else 'NO-GO'}")
    print("=" * 70)

    await conn.close()
    print("\n[DONE]")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense",
                        help="Search mode: dense (vector only) or hybrid (router + RRF)")
    parser.add_argument("--log-results", action="store_true",
                        help="Log results to DB and files")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for JSONL and CSV logs (default: no file logging)")
    parser.add_argument("--tag", type=str, default="default",
                        help="Tag for log files (e.g., hybrid_v1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Include debug info in JSONL (router_hit, stage, etc.)")

    args = parser.parse_args()
    asyncio.run(run_evaluation(
        top_k=args.top_k,
        log_results=args.log_results,
        search_mode=args.mode,
        log_dir=args.log_dir,
        tag=args.tag,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
