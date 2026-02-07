#!/usr/bin/env python3
"""Hybrid search test with source indicator"""
import os
import sys
import psycopg2
import requests

QUERY = sys.argv[1] if len(sys.argv) > 1 else "risarcimento danno responsabilità civile"
RRF_K = 60

def get_emb(text):
    key = os.environ.get("OPENROUTER_API_KEY", "")
    r = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "openai/text-embedding-3-small", "input": [text]},
        timeout=30
    )
    return r.json()["data"][0]["embedding"]

conn = psycopg2.connect(
    host="localhost", port=5436, user="lexe_max",
    password="lexe_max_dev_password", dbname="lexe_max"
)
cur = conn.cursor()

print(f"Query: {QUERY}")
print("Generating embedding...")
qemb = get_emb(QUERY)

cur.execute("""
WITH dense AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY e.embedding <=> %s::vector(1536)) as rk,
           1 - (e.embedding <=> %s::vector(1536)) as sc
    FROM kb.normativa_chunk c
    JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
    ORDER BY e.embedding <=> %s::vector(1536)
    LIMIT 50
),
sparse AS (
    SELECT f.chunk_id as id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(f.tsv_it, plainto_tsquery('italian', %s)) DESC) as rk,
           ts_rank_cd(f.tsv_it, plainto_tsquery('italian', %s)) as sc
    FROM kb.normativa_chunk_fts f
    WHERE f.tsv_it @@ plainto_tsquery('italian', %s)
    ORDER BY ts_rank_cd(f.tsv_it, plainto_tsquery('italian', %s)) DESC
    LIMIT 50
),
combined AS (
    SELECT COALESCE(d.id, s.id) as id,
           COALESCE(1.0 / (%s + d.rk), 0) + COALESCE(1.0 / (%s + s.rk), 0) as rrf,
           d.sc as dense_sc,
           s.sc as sparse_sc,
           CASE
               WHEN d.id IS NOT NULL AND s.id IS NOT NULL THEN 'BOTH'
               WHEN d.id IS NOT NULL THEN 'DENSE'
               ELSE 'SPARSE'
           END as source
    FROM dense d
    FULL OUTER JOIN sparse s ON d.id = s.id
)
SELECT w.code, n.articolo, c.chunk_no, r.source,
       ROUND(r.rrf::numeric, 4) as rrf,
       ROUND(COALESCE(r.dense_sc, 0)::numeric, 3) as dense,
       ROUND(COALESCE(r.sparse_sc, 0)::numeric, 4) as sparse,
       LEFT(c.text, 100) as preview
FROM combined r
JOIN kb.normativa_chunk c ON c.id = r.id
JOIN kb.normativa n ON n.id = c.normativa_id
JOIN kb.work w ON w.id = c.work_id
ORDER BY r.rrf DESC
LIMIT 15
""", (qemb, qemb, qemb, QUERY, QUERY, QUERY, QUERY, RRF_K, RRF_K))

results = cur.fetchall()

print("=" * 115)
print(f"{'Code':<5} {'Art':<10} {'#':<2} {'Source':<6} {'RRF':<7} {'Dense':<6} {'Sparse':<7} Preview")
print("-" * 115)

for r in results:
    code, art, chunk, source, rrf, dense, sparse, preview = r
    preview = preview.replace("\n", " ")[:55] if preview else ""
    print(f"{code:<5} {art:<10} {chunk:<2} {source:<6} {rrf:<7} {dense:<6} {sparse:<7} {preview}")

conn.close()
