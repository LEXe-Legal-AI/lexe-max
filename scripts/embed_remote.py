#!/usr/bin/env python3
"""Generate embeddings for normativa chunks on staging."""
import os
import time
import psycopg2
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 50

def get_embeddings(texts):
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": MODEL, "input": texts},
        timeout=120
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

conn = psycopg2.connect(
    host="localhost", port=5435,
    user="lexe", password="lexe_stage_cc07b664a88cb8e6",
    dbname="lexe_kb"
)
cur = conn.cursor()

# Get chunks without embeddings
cur.execute("SELECT id, text FROM kb.normativa_chunk WHERE embedding IS NULL ORDER BY id")
chunks = cur.fetchall()
print(f"Chunks to embed: {len(chunks)}")

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set")
    exit(1)

total = 0
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    ids = [c[0] for c in batch]
    texts = [c[1] for c in batch]

    try:
        embeddings = get_embeddings(texts)
        for chunk_id, emb in zip(ids, embeddings):
            cur.execute(
                "UPDATE kb.normativa_chunk SET embedding = %s WHERE id = %s",
                (emb, chunk_id)
            )
        conn.commit()
        total += len(batch)
        print(f"  Embedded {total}/{len(chunks)} ({100*total/len(chunks):.1f}%)")
    except Exception as e:
        print(f"  ERROR: {e}")
        time.sleep(5)
        continue

cur.execute("SELECT COUNT(*) FROM kb.normativa_chunk WHERE embedding IS NOT NULL")
final = cur.fetchone()[0]
print(f"DONE! Chunks with embeddings: {final}")
conn.close()
