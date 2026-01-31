"""
Generate Golden Set for Retrieval Evaluation

Two types of queries with deterministic ground truth:

1. SELF-RETRIEVAL: First 12-18 words of massima (without citations)
   Ground truth = the source massima itself
   Tests: semantic similarity, embedding quality

2. CITATION-RETRIEVAL: Citation anchor (Sez. n. anno Rv.)
   Ground truth = massima containing that citation
   Tests: structured search, citation parsing

Usage:
    uv run python scripts/qa/generate_golden_set.py --count 200 --dry-run
    uv run python scripts/qa/generate_golden_set.py --count 200 --commit
"""

import argparse
import asyncio
import re
import random
from dataclasses import dataclass

import asyncpg

# ============================================================
# Configuration
# ============================================================

DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# Self-retrieval config
SELF_MIN_WORDS = 12
SELF_MAX_WORDS = 18
SELF_MIN_TEXT_LENGTH = 200  # Skip very short massime

# Citation patterns to remove from self-queries
CITATION_PATTERNS = [
    r'\bSez\.?\s*\d*\s*[-–]?\s*\d*\s*,?\s*n\.?\s*\d+(?:/\d+)?',
    r'\bRv\.?\s*\d+(?:-\d+)?',
    r'\bCass\.?\s*(?:civ|pen)?\.?',
    r'\bart\.?\s*\d+',
    r'\bc\.?\s*p\.?\s*c\.?',
    r'\bc\.?\s*c\.?',
]


@dataclass
class GoldenQuery:
    query_text: str
    query_type: str  # 'self' | 'citation'
    expected_massima_id: str
    expected_doc_id: str
    source_preview: str
    citation_anchor: str | None = None


# ============================================================
# Query Generation
# ============================================================

def clean_for_self_query(text: str) -> str:
    """Remove citations and legal references for self-retrieval query."""
    cleaned = text
    for pattern in CITATION_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def extract_self_query(text: str) -> str | None:
    """Extract first 12-18 words for self-retrieval query."""
    cleaned = clean_for_self_query(text)
    words = cleaned.split()

    if len(words) < SELF_MIN_WORDS:
        return None

    # Random length between 12-18
    query_len = random.randint(SELF_MIN_WORDS, min(SELF_MAX_WORDS, len(words)))
    query = ' '.join(words[:query_len])

    # Skip if too short after cleaning
    if len(query) < 50:
        return None

    return query


def extract_citation_query(text: str, sezione: str, numero: str, anno: int, rv: str) -> str | None:
    """Build citation query from structured fields."""
    parts = []

    if sezione:
        # Normalize section format
        sez = sezione.upper().strip()
        if sez in ('U', 'UN', 'UNITE'):
            parts.append("Sez. U")
        elif sez.isdigit():
            parts.append(f"Sez. {sez}")
        elif sez in ('L', 'T'):
            parts.append(f"Sez. {sez}")
        else:
            parts.append(f"Sez. {sez}")

    if numero and anno:
        parts.append(f"n. {numero}/{anno}")
    elif numero:
        parts.append(f"n. {numero}")

    if rv:
        parts.append(f"Rv. {rv}")

    if len(parts) < 2:
        return None

    return ", ".join(parts)


async def generate_self_queries(
    conn: asyncpg.Connection,
    count: int,
) -> list[GoldenQuery]:
    """Generate self-retrieval queries from random massime."""

    # Get random active massime with sufficient length
    rows = await conn.fetch("""
        SELECT m.id, m.testo, m.document_id
        FROM kb.massime m
        WHERE m.is_active = TRUE
        AND LENGTH(m.testo) >= $1
        ORDER BY RANDOM()
        LIMIT $2
    """, SELF_MIN_TEXT_LENGTH, count * 2)  # Fetch more, some will be filtered

    queries = []
    for row in rows:
        if len(queries) >= count:
            break

        query_text = extract_self_query(row["testo"])
        if not query_text:
            continue

        queries.append(GoldenQuery(
            query_text=query_text,
            query_type="self",
            expected_massima_id=str(row["id"]),
            expected_doc_id=str(row["document_id"]),
            source_preview=row["testo"][:100],
        ))

    return queries


async def generate_citation_queries(
    conn: asyncpg.Connection,
    count: int,
) -> list[GoldenQuery]:
    """Generate citation-retrieval queries from massime with complete citations."""

    # Get massime with citation_complete = TRUE
    rows = await conn.fetch("""
        SELECT m.id, m.testo, m.document_id,
               m.sezione, m.numero, m.anno, m.rv
        FROM kb.massime m
        WHERE m.is_active = TRUE
        AND (m.sezione IS NOT NULL OR m.rv IS NOT NULL)
        AND (m.numero IS NOT NULL OR m.rv IS NOT NULL)
        ORDER BY RANDOM()
        LIMIT $1
    """, count * 2)

    queries = []
    for row in rows:
        if len(queries) >= count:
            break

        citation = extract_citation_query(
            row["testo"],
            row["sezione"],
            row["numero"],
            row["anno"],
            row["rv"],
        )

        if not citation:
            continue

        queries.append(GoldenQuery(
            query_text=citation,
            query_type="citation",
            expected_massima_id=str(row["id"]),
            expected_doc_id=str(row["document_id"]),
            source_preview=row["testo"][:100],
            citation_anchor=citation,
        ))

    return queries


# ============================================================
# Database Operations
# ============================================================

async def clear_old_golden_set(conn: asyncpg.Connection, batch: int):
    """Deactivate old golden set queries."""
    await conn.execute("""
        UPDATE kb.golden_queries
        SET is_active = FALSE
        WHERE generation_batch < $1
    """, batch)


async def insert_golden_queries(
    conn: asyncpg.Connection,
    queries: list[GoldenQuery],
    batch: int,
) -> int:
    """Insert golden set queries."""

    insert_data = [
        (
            q.query_text,
            q.query_type,
            q.expected_massima_id,
            q.expected_doc_id,
            q.source_preview,
            q.citation_anchor,
            batch,
        )
        for q in queries
    ]

    await conn.executemany("""
        INSERT INTO kb.golden_queries
        (query_text, query_type, expected_massima_id, expected_doc_id,
         source_text_preview, citation_anchor, generation_batch)
        VALUES ($1, $2, $3::uuid, $4::uuid, $5, $6, $7)
    """, insert_data)

    return len(insert_data)


async def get_current_batch(conn: asyncpg.Connection) -> int:
    """Get next batch number."""
    row = await conn.fetchrow("""
        SELECT COALESCE(MAX(generation_batch), 0) + 1 as next_batch
        FROM kb.golden_queries
    """)
    return row["next_batch"]


# ============================================================
# Main
# ============================================================

async def generate_golden_set(count: int, dry_run: bool):
    """Generate golden set queries."""

    print("=" * 70)
    print("GOLDEN SET GENERATION")
    print("=" * 70)
    print(f"Target count:  {count} total ({count//2} self + {count//2} citation)")
    print(f"Mode:          {'DRY RUN' if dry_run else 'COMMIT'}")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Get next batch number
    batch = await get_current_batch(conn)
    print(f"[OK] Batch number: {batch}")

    # Generate queries
    print("\nGenerating self-retrieval queries...")
    self_queries = await generate_self_queries(conn, count // 2)
    print(f"  Generated: {len(self_queries)}")

    print("\nGenerating citation-retrieval queries...")
    citation_queries = await generate_citation_queries(conn, count // 2)
    print(f"  Generated: {len(citation_queries)}")

    all_queries = self_queries + citation_queries
    random.shuffle(all_queries)

    # Show samples
    print("\n--- Sample SELF queries ---")
    for q in self_queries[:3]:
        print(f"  Q: {q.query_text[:60]}...")
        print(f"     -> {q.expected_massima_id[:8]}")

    print("\n--- Sample CITATION queries ---")
    for q in citation_queries[:3]:
        print(f"  Q: {q.query_text}")
        print(f"     -> {q.expected_massima_id[:8]}")

    # Insert
    if not dry_run:
        print(f"\nInserting {len(all_queries)} queries (batch {batch})...")
        inserted = await insert_golden_queries(conn, all_queries, batch)
        print(f"[OK] Inserted {inserted} golden queries")

        # Deactivate old
        await clear_old_golden_set(conn, batch)
        print("[OK] Deactivated old batches")
    else:
        print(f"\n[DRY RUN] Would insert {len(all_queries)} queries")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Self queries:     {len(self_queries)}")
    print(f"Citation queries: {len(citation_queries)}")
    print(f"Total:            {len(all_queries)}")

    await conn.close()
    print("\n[DONE]")


def main():
    parser = argparse.ArgumentParser(description="Generate golden set for retrieval eval")
    parser.add_argument("--count", type=int, default=200, help="Total queries to generate")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert to database")
    parser.add_argument("--commit", action="store_true", help="Insert to database")

    args = parser.parse_args()

    if not args.dry_run and not args.commit:
        print("ERROR: Must specify --dry-run or --commit")
        return

    asyncio.run(generate_golden_set(args.count, args.dry_run))


if __name__ == "__main__":
    main()
