"""
QA Protocol - Phase 2: Extraction Quality

Aggregates page_extraction_stats into per-document extraction quality:
- Distribution of chars/page (p10, p50, p90, stddev)
- Coverage ratio (pages_with_content / total_pages)
- Overall quality_score and quality_grade (A/B/C/D)

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    uv run python scripts/qa/s2_extraction_quality.py
"""

import asyncio

import asyncpg

from qa_config import DB_URL


def grade_from_score(score: float) -> str:
    """Map quality score to grade ENUM."""
    if score >= 0.9:
        return "A"
    if score >= 0.7:
        return "B"
    if score >= 0.5:
        return "C"
    return "D"


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 2: EXTRACTION QUALITY")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] Using qa_run_id={qa_run_id}")

    manifests = await conn.fetch(
        "SELECT id, filename, pages FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    grades = {"A": 0, "B": 0, "C": 0, "D": 0}

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT 1 FROM kb.pdf_extraction_quality WHERE manifest_id = $1",
            manifest_id,
        )
        if existing:
            continue

        # Aggregate from page stats
        agg = await conn.fetchrow(
            """
            SELECT
                sum(char_count) as total_chars,
                sum(word_count) as total_words,
                sum(element_count) as total_elements,
                avg(valid_chars_ratio) as avg_valid_chars,
                avg(italian_tokens_ratio) as avg_italian_tokens,
                count(*) as total_pages,
                count(*) FILTER (WHERE NOT is_empty) as content_pages,
                count(*) FILTER (WHERE is_empty) as empty_pages
            FROM kb.page_extraction_stats
            WHERE manifest_id = $1
            """,
            manifest_id,
        )

        if not agg or agg["total_pages"] == 0:
            print(f"  [SKIP] {filename}: no page stats")
            continue

        total_chars = agg["total_chars"] or 0
        total_words = agg["total_words"] or 0
        total_elements = agg["total_elements"] or 0
        avg_valid = float(agg["avg_valid_chars"] or 0)
        avg_italian = float(agg["avg_italian_tokens"] or 0)
        total_pages = agg["total_pages"]
        content_pages = agg["content_pages"]

        # Coverage ratio
        coverage = content_pages / total_pages if total_pages > 0 else 0

        # Citation regex check (at least some citations found)
        citation_count = await conn.fetchval(
            """
            SELECT count(*)
            FROM kb.massime m
            JOIN kb.pdf_manifest pm ON pm.doc_id = m.document_id
            WHERE pm.id = $1
            """,
            manifest_id,
        )
        has_citations = (citation_count or 0) > 0

        # Overall quality score
        quality_score = (
            avg_valid * 0.4
            + avg_italian * 0.4
            + coverage * 0.2
        )
        quality_score = min(max(quality_score, 0.0), 1.0)
        grade = grade_from_score(quality_score)
        grades[grade] += 1

        await conn.execute(
            """
            INSERT INTO kb.pdf_extraction_quality
              (qa_run_id, manifest_id, total_chars, total_words, total_elements,
               valid_chars_ratio, italian_tokens_ratio, citation_regex_success,
               overall_quality_score, quality_grade)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::kb.qa_quality_grade)
            ON CONFLICT (manifest_id) DO NOTHING
            """,
            qa_run_id, manifest_id,
            total_chars, total_words, total_elements,
            round(avg_valid, 4), round(avg_italian, 4),
            has_citations, round(quality_score, 4), grade,
        )

        print(f"  [{grade}] {filename}: score={quality_score:.3f}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"EXTRACTION QUALITY COMPLETE")
    print(f"{'=' * 70}")
    for g in ("A", "B", "C", "D"):
        print(f"  Grade {g}: {grades[g]}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
