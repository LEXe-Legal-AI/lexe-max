#!/usr/bin/env python3
"""
Hybrid search on lexe-max staging: Dense + Sparse + RRF fusion

Usage:
    OPENROUTER_API_KEY=sk-or-... python3 hybrid_search_staging.py "query text"
"""
import os
import sys
import psycopg2
import requests

QUERY = sys.argv[1] if len(sys.argv) > 1 else "risarcimento danno responsabilità civile"
TOP_K = 10
RRF_K = 60

DB_CONFIG = {
    "host": "localhost",
    "port": 5436,
    "user": "lexe_max",
    "password": "lexe_max_dev_password",
    "dbname": "lexe_max",
}

def get_embedding(text):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "openai/text-embedding-3-small", "input": [text]},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def main():
    print(f"Query: {QUERY}")
    print("=" * 80)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("Generating query embedding...")
    query_emb = get_embedding(QUERY)

    # Hybrid search with RRF
    cur.execute("""
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
        COALESCE(1.0 / (%s + d.rank_dense), 0) as rrf_dense,
        COALESCE(1.0 / (%s + s.rank_sparse), 0) as rrf_sparse,
        COALESCE(1.0 / (%s + d.rank_dense), 0) + COALESCE(1.0 / (%s + s.rank_sparse), 0) as rrf_score,
        d.score_dense,
        s.score_sparse
    FROM dense d
    FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
)
SELECT w.code, n.articolo, c.chunk_no,
       ROUND(r.rrf_score::numeric, 4) as rrf,
       ROUND(r.score_dense::numeric, 3) as dense,
       ROUND(COALESCE(r.score_sparse, 0)::numeric, 3) as sparse,
       LEFT(c.text, 150) as preview
FROM rrf r
JOIN kb.normativa_chunk c ON c.id = r.chunk_id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
ORDER BY r.rrf_score DESC
LIMIT %s;
    """, (query_emb, QUERY, RRF_K, RRF_K, RRF_K, RRF_K, TOP_K))

    results = cur.fetchall()

    print(f"\nTop {TOP_K} Hybrid Results (RRF k={RRF_K}):\n")
    print(f"{'Code':<5} {'Art':<10} {'#':<2} {'RRF':<7} {'Dense':<6} {'Sparse':<6} Preview")
    print("-" * 100)

    for row in results:
        code, art, chunk, rrf, dense, sparse, preview = row
        preview = preview.replace("\n", " ")[:70] if preview else ""
        dense_str = f"{dense:.3f}" if dense else "-"
        sparse_str = f"{sparse:.3f}" if sparse else "-"
        print(f"{code:<5} {art:<10} {chunk:<2} {rrf:<7.4f} {dense_str:<6} {sparse_str:<6} {preview}")

    conn.close()


if __name__ == "__main__":
    main()
