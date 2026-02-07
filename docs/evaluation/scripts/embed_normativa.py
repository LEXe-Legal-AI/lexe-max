#!/usr/bin/env python3
"""
Generate embeddings for normativa chunks on STAGING.

TARGET: lexe-max container (porta 5436) - KB Legal database
NOT: lexe-postgres (porta 5435) - Sistema

Usage:
    python scripts/embed_lexe_max_staging.py --estimate
    OPENROUTER_API_KEY=sk-or-... python scripts/embed_lexe_max_staging.py
"""
import os
import sys
import time
import argparse
import psycopg2

# ============================================================
# CONFIGURAZIONE ESPLICITA - LEXE-MAX STAGING
# Container: lexe-max (NON lexe-postgres!)
# Porta: 5436 (NON 5435!)
# Database: lexe_max
# Schema: kb
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5436,
    "user": "lexe_max",
    "password": "lexe_max_dev_password",
    "dbname": "lexe_max",
}

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "openai/text-embedding-3-small"
CHANNEL = "testo"
DIMS = 1536
BATCH_SIZE = 50


def get_connection():
    print(f"Connecting to lexe-max staging...")
    print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['dbname']}")
    return psycopg2.connect(**DB_CONFIG)


def get_embeddings(texts):
    import requests
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": MODEL, "input": texts},
        timeout=120
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--codice", help="Only process specific codice")
    args = parser.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    # Get chunks without embeddings
    query = """
        SELECT c.id, c.text, w.code
        FROM kb.normativa_chunk c
        JOIN kb.work w ON w.id = c.work_id
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.normativa_chunk_embeddings e
            WHERE e.chunk_id = c.id AND e.model = %s AND e.channel = %s
        )
    """
    params = [MODEL, CHANNEL]

    if args.codice:
        query += " AND w.code = %s"
        params.append(args.codice.upper())

    query += " ORDER BY w.code, c.id"
    cur.execute(query, params)
    chunks = cur.fetchall()

    # Stats per codice
    stats = {}
    for _, _, code in chunks:
        stats[code] = stats.get(code, 0) + 1

    print(f"\nChunks senza embeddings:")
    for code in sorted(stats.keys()):
        print(f"  {code}: {stats[code]}")
    print(f"  TOTALE: {len(chunks)}")

    total_tokens = sum(len(c[1]) // 4 for c in chunks)
    cost = (total_tokens / 1000) * 0.00002
    print(f"\nStima costo: ${cost:.4f} ({total_tokens:,} tokens)")

    if args.estimate:
        print("\n[--estimate] Nessuna modifica")
        conn.close()
        return

    if not OPENROUTER_API_KEY:
        print("\nERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if not chunks:
        print("\nNessun chunk da processare!")
        conn.close()
        return

    print(f"\nGenerando embeddings...")
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        ids = [c[0] for c in batch]
        texts = [c[1] for c in batch]

        try:
            embeddings = get_embeddings(texts)
            for chunk_id, emb in zip(ids, embeddings):
                cur.execute("""
                    INSERT INTO kb.normativa_chunk_embeddings
                    (chunk_id, model, channel, dims, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id, model, channel, dims)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                """, (chunk_id, MODEL, CHANNEL, DIMS, emb))
            conn.commit()
            total += len(batch)
            print(f"  {total}/{len(chunks)} ({100*total/len(chunks):.1f}%)")
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()
            time.sleep(5)

    print(f"\nDONE! Embeddings generati: {total}")
    conn.close()


if __name__ == "__main__":
    main()
