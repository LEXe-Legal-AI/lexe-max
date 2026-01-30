"""
KB Massimari - Test Retrieval on Staging
Tests R1 (full-text) and R2-Local (Mistral vector search).

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export PATH=$HOME/.local/bin:$PATH
    export OPENROUTER_API_KEY='sk-or-v1-...'
    uv run python scripts/test_retrieval_staging.py
"""
import asyncio
import os
import time

import asyncpg
import httpx

# Config
DB_URL = "postgresql://leo:stage_postgres_2026_secure@localhost:5432/leo"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_MODEL = "mistralai/mistral-embed-2312"

# Test queries
TEST_QUERIES = [
    "responsabilità medica danno alla salute",
    "risarcimento danni contratto",
    "prescrizione crediti lavoro",
    "licenziamento giusta causa",
    "proprietà immobiliare usucapione",
]


async def get_query_embedding(client: httpx.AsyncClient, text: str) -> list[float] | None:
    """Get embedding for query text."""
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
            data = response.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"  [ERROR] Embedding: {e}")
    return None


def embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def test_r1_fulltext(conn: asyncpg.Connection, query: str) -> list[dict]:
    """R1: Full-text search using tsvector."""
    # Italian stemming search
    results = await conn.fetch("""
        SELECT m.id, m.anno, m.tipo,
               LEFT(m.testo, 200) as snippet,
               ts_rank(m.tsv_italian, plainto_tsquery('italian', $1)) as score
        FROM kb.massime m
        WHERE m.tsv_italian @@ plainto_tsquery('italian', $1)
        ORDER BY score DESC
        LIMIT 5
    """, query)
    return [dict(r) for r in results]


async def test_r2_vector(conn: asyncpg.Connection, client: httpx.AsyncClient, query: str) -> list[dict]:
    """R2-Local: Vector search using Mistral embeddings."""
    # Get query embedding
    embedding = await get_query_embedding(client, query)
    if not embedding:
        return []

    emb_str = embedding_to_pgvector(embedding)

    # Cosine similarity search with HNSW
    results = await conn.fetch("""
        SELECT m.id, m.anno, m.tipo,
               LEFT(m.testo, 200) as snippet,
               1 - (e.embedding <=> $1::vector) as similarity
        FROM kb.emb_mistral e
        JOIN kb.massime m ON m.id = e.massima_id
        ORDER BY e.embedding <=> $1::vector
        LIMIT 5
    """, emb_str)
    return [dict(r) for r in results]


async def test_hybrid(conn: asyncpg.Connection, client: httpx.AsyncClient, query: str) -> list[dict]:
    """Hybrid: Combine R1 + R2 with RRF (Reciprocal Rank Fusion)."""
    embedding = await get_query_embedding(client, query)
    if not embedding:
        return []

    emb_str = embedding_to_pgvector(embedding)

    # RRF formula: score = sum(1 / (k + rank)) where k=60
    results = await conn.fetch("""
        WITH r1_results AS (
            SELECT m.id, m.anno, m.tipo, m.testo,
                   ROW_NUMBER() OVER (ORDER BY ts_rank(m.tsv_italian, plainto_tsquery('italian', $1)) DESC) as rank
            FROM kb.massime m
            WHERE m.tsv_italian @@ plainto_tsquery('italian', $1)
            LIMIT 20
        ),
        r2_results AS (
            SELECT m.id, m.anno, m.tipo, m.testo,
                   ROW_NUMBER() OVER (ORDER BY e.embedding <=> $2::vector) as rank
            FROM kb.emb_mistral e
            JOIN kb.massime m ON m.id = e.massima_id
            LIMIT 20
        ),
        combined AS (
            SELECT COALESCE(r1.id, r2.id) as id,
                   COALESCE(r1.anno, r2.anno) as anno,
                   COALESCE(r1.tipo, r2.tipo) as tipo,
                   COALESCE(r1.testo, r2.testo) as testo,
                   COALESCE(1.0 / (60 + r1.rank), 0) + COALESCE(1.0 / (60 + r2.rank), 0) as rrf_score
            FROM r1_results r1
            FULL OUTER JOIN r2_results r2 ON r1.id = r2.id
        )
        SELECT id, anno, tipo, LEFT(testo, 200) as snippet, rrf_score
        FROM combined
        ORDER BY rrf_score DESC
        LIMIT 5
    """, query, emb_str)
    return [dict(r) for r in results]


async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return

    print("=" * 70)
    print("KB MASSIMARI - RETRIEVAL TEST (STAGING)")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)

    # Get stats
    massime_count = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")
    emb_count = await conn.fetchval("SELECT COUNT(*) FROM kb.emb_mistral")
    print(f"[OK] Connected - {massime_count} massime, {emb_count} embeddings")

    async with httpx.AsyncClient() as client:
        for query in TEST_QUERIES:
            print(f"\n{'=' * 70}")
            print(f"QUERY: {query}")
            print("=" * 70)

            # R1: Full-text
            print("\n[R1] Full-Text Search (Italian stemming):")
            start = time.time()
            r1_results = await test_r1_fulltext(conn, query)
            r1_time = (time.time() - start) * 1000
            if r1_results:
                for i, r in enumerate(r1_results, 1):
                    print(f"  {i}. [{r['anno']} {r['tipo']}] score={r['score']:.4f}")
                    print(f"     {r['snippet'][:80]}...")
            else:
                print("  (no results)")
            print(f"  Time: {r1_time:.0f}ms")

            # R2: Vector
            print("\n[R2-Local] Mistral Vector Search (cosine):")
            start = time.time()
            r2_results = await test_r2_vector(conn, client, query)
            r2_time = (time.time() - start) * 1000
            if r2_results:
                for i, r in enumerate(r2_results, 1):
                    print(f"  {i}. [{r['anno']} {r['tipo']}] sim={r['similarity']:.4f}")
                    print(f"     {r['snippet'][:80]}...")
            else:
                print("  (no results)")
            print(f"  Time: {r2_time:.0f}ms (includes embedding generation)")

            # Hybrid
            print("\n[HYBRID] RRF (k=60):")
            start = time.time()
            hybrid_results = await test_hybrid(conn, client, query)
            hybrid_time = (time.time() - start) * 1000
            if hybrid_results:
                for i, r in enumerate(hybrid_results, 1):
                    print(f"  {i}. [{r['anno']} {r['tipo']}] rrf={r['rrf_score']:.4f}")
                    print(f"     {r['snippet'][:80]}...")
            else:
                print("  (no results)")
            print(f"  Time: {hybrid_time:.0f}ms")

    await conn.close()

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
