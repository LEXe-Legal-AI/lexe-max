"""
Test Retrieval Script - KB Massimari
Testa il sistema di ricerca ibrido: FTS + trgm.
"""
import asyncio
from datetime import date

import asyncpg

DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"


async def test_fts_search(conn: asyncpg.Connection, query: str, limit: int = 5):
    """Test FTS (full-text search) con ranking."""
    print(f"\n{'='*60}")
    print(f"FTS Search: '{query}'")
    print("="*60)

    rows = await conn.fetch("""
        SELECT m.sezione, m.numero, m.materia, d.anno,
               ts_rank_cd(m.tsv_italian, plainto_tsquery('italian', $1)) as score,
               LEFT(m.testo, 150) as excerpt
        FROM kb.massime m
        JOIN kb.documents d ON m.document_id = d.id
        WHERE m.tsv_italian @@ plainto_tsquery('italian', $1)
           OR m.tsv_simple @@ plainto_tsquery('simple', $1)
        ORDER BY score DESC
        LIMIT $2
    """, query, limit)

    if rows:
        for r in rows:
            print(f"  [{r['score']:.3f}] Sez. {r['sezione']}, n. {r['numero']} ({r['anno']}) - {r['materia']}")
            print(f"          {r['excerpt'][:100]}...")
    else:
        print("  (nessun risultato)")

    return len(rows)


async def test_bm25_function(conn: asyncpg.Connection, query: str, limit: int = 5):
    """Test funzione BM25 custom."""
    print(f"\n{'='*60}")
    print(f"BM25 Function: '{query}'")
    print("="*60)

    rows = await conn.fetch("""
        SELECT r.massima_id, r.score,
               m.sezione, m.numero, m.materia,
               LEFT(m.testo, 120) as excerpt
        FROM kb.bm25_search($1, $2) r
        JOIN kb.massime m ON m.id = r.massima_id
        ORDER BY r.score DESC
    """, query, limit)

    if rows:
        for r in rows:
            print(f"  [{r['score']:.3f}] Sez. {r['sezione']}, n. {r['numero']} - {r['materia']}")
            print(f"          {r['excerpt'][:100]}...")
    else:
        print("  (nessun risultato)")

    return len(rows)


async def test_combined_queries(conn: asyncpg.Connection):
    """Test con query giuridiche tipiche."""
    queries = [
        "colpa omissiva",
        "datore lavoro responsabilità",
        "sezioni unite",
        "infortunio",
        "cautelare",
        "reato",
    ]

    print("\n" + "="*60)
    print("COMBINED QUERY TEST")
    print("="*60)

    for q in queries:
        count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM kb.massime m
            WHERE m.tsv_italian @@ plainto_tsquery('italian', $1)
        """, q)
        print(f"  '{q}': {count} risultati")


async def get_db_stats(conn: asyncpg.Connection):
    """Statistiche database."""
    print("\n" + "="*60)
    print("DATABASE STATS")
    print("="*60)

    docs = await conn.fetchval("SELECT COUNT(*) FROM kb.documents")
    massime = await conn.fetchval("SELECT COUNT(*) FROM kb.massime")
    by_materia = await conn.fetch("""
        SELECT materia, COUNT(*) as cnt
        FROM kb.massime
        GROUP BY materia
    """)
    by_sezione = await conn.fetch("""
        SELECT sezione, COUNT(*) as cnt
        FROM kb.massime
        GROUP BY sezione
        ORDER BY cnt DESC
        LIMIT 5
    """)

    print(f"  Documents: {docs}")
    print(f"  Massime: {massime}")
    print(f"  By materia: {[(r['materia'], r['cnt']) for r in by_materia]}")
    print(f"  Top sezioni: {[(r['sezione'], r['cnt']) for r in by_sezione]}")


async def main():
    print("="*60)
    print("KB Massimari - Test Retrieval")
    print("="*60)

    conn = await asyncpg.connect(DB_URL)

    try:
        await get_db_stats(conn)
        await test_combined_queries(conn)

        # Test specifici
        await test_fts_search(conn, "colpa")
        await test_fts_search(conn, "infortunio lavoro")
        await test_bm25_function(conn, "sezioni unite")
        await test_bm25_function(conn, "datore")

    finally:
        await conn.close()

    print("\n" + "="*60)
    print("Test completato!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
