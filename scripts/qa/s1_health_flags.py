"""
QA Protocol - Phase 1: Health Flags

Analyzes page_extraction_stats + year_resolution to flag health issues.
Depends on s1_page_extraction_stats.py and s1_year_resolution.py.

Flags:
- empty_page_sequence (severity 3): 3+ consecutive empty pages
- low_ocr_quality (severity 4): valid_chars_ratio < 0.7 on >20% pages
- year_conflict (severity 3): has_conflict=true in year_resolution
- high_noise_pages (severity 2): >5% pages with non_alnum_ratio > 0.3
- low_extraction (severity 3): <10 chars average per page

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    uv run python scripts/qa/s1_health_flags.py
"""

import asyncio
import json

import asyncpg
from asyncpg import Range

from qa_config import DB_URL


async def check_empty_sequences(conn, qa_run_id: int, manifest_id: int) -> list[dict]:
    """Find sequences of 3+ consecutive empty pages."""
    pages = await conn.fetch(
        """
        SELECT page_number, is_empty
        FROM kb.page_extraction_stats
        WHERE manifest_id = $1
        ORDER BY page_number
        """,
        manifest_id,
    )
    flags = []
    streak = []
    for p in pages:
        if p["is_empty"]:
            streak.append(p["page_number"])
        else:
            if len(streak) >= 3:
                flags.append({
                    "flag_type": "empty_page_sequence",
                    "severity": 3,
                    "page_start": streak[0],
                    "page_end": streak[-1],
                    "details": {"consecutive_empty": len(streak)},
                })
            streak = []
    # Check trailing streak
    if len(streak) >= 3:
        flags.append({
            "flag_type": "empty_page_sequence",
            "severity": 3,
            "page_start": streak[0],
            "page_end": streak[-1],
            "details": {"consecutive_empty": len(streak)},
        })
    return flags


async def check_ocr_quality(conn, qa_run_id: int, manifest_id: int) -> list[dict]:
    """Flag if valid_chars_ratio < 0.7 on >20% of non-empty pages."""
    stats = await conn.fetch(
        """
        SELECT page_number, valid_chars_ratio, is_empty
        FROM kb.page_extraction_stats
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    non_empty = [s for s in stats if not s["is_empty"]]
    if not non_empty:
        return []

    low_ocr = [s for s in non_empty if (s["valid_chars_ratio"] or 0) < 0.7]
    pct = len(low_ocr) / len(non_empty)

    if pct > 0.2:
        low_pages = sorted(s["page_number"] for s in low_ocr)
        return [{
            "flag_type": "low_ocr_quality",
            "severity": 4,
            "page_start": low_pages[0] if low_pages else None,
            "page_end": low_pages[-1] if low_pages else None,
            "details": {
                "pct_low_ocr": round(pct, 3),
                "low_ocr_pages": len(low_ocr),
                "total_pages": len(non_empty),
            },
        }]
    return []


async def check_year_conflict(conn, qa_run_id: int, manifest_id: int) -> list[dict]:
    """Flag year conflict."""
    row = await conn.fetchrow(
        """
        SELECT has_conflict, conflict_details
        FROM kb.pdf_year_resolution
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    if row and row["has_conflict"]:
        return [{
            "flag_type": "year_conflict",
            "severity": 3,
            "page_start": None,
            "page_end": None,
            "details": {"conflict_details": row["conflict_details"]},
        }]
    return []


async def check_noise_pages(conn, qa_run_id: int, manifest_id: int) -> list[dict]:
    """Flag if >5% pages have high non_alnum_ratio."""
    stats = await conn.fetch(
        """
        SELECT page_number, non_alnum_ratio, is_empty
        FROM kb.page_extraction_stats
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    non_empty = [s for s in stats if not s["is_empty"]]
    if not non_empty:
        return []

    noisy = [s for s in non_empty if (s["non_alnum_ratio"] or 0) > 0.3]
    pct = len(noisy) / len(non_empty)

    if pct > 0.05:
        return [{
            "flag_type": "high_noise_pages",
            "severity": 2,
            "page_start": None,
            "page_end": None,
            "details": {
                "pct_noisy": round(pct, 3),
                "noisy_pages": len(noisy),
                "total_pages": len(non_empty),
            },
        }]
    return []


async def check_low_extraction(conn, qa_run_id: int, manifest_id: int) -> list[dict]:
    """Flag if average chars per page is very low (<10)."""
    row = await conn.fetchrow(
        """
        SELECT avg(char_count) as avg_chars, count(*) as total
        FROM kb.page_extraction_stats
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    if row and row["avg_chars"] is not None:
        avg_chars = float(row["avg_chars"])
        if avg_chars < 10 and row["total"] > 5:
            return [{
                "flag_type": "low_extraction",
                "severity": 3,
                "page_start": None,
                "page_end": None,
                "details": {
                    "avg_chars_per_page": round(avg_chars, 1),
                    "total_pages": row["total"],
                },
            }]
    return []


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 1: HEALTH FLAGS")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] Using qa_run_id={qa_run_id}")

    manifests = await conn.fetch(
        "SELECT id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    checks = [
        check_empty_sequences,
        check_ocr_quality,
        check_year_conflict,
        check_noise_pages,
        check_low_extraction,
    ]

    total_flags = 0
    by_type = {}

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT count(*) FROM kb.pdf_health_flags WHERE manifest_id = $1 AND qa_run_id = $2",
            manifest_id, qa_run_id,
        )
        if existing > 0:
            continue

        doc_flags = []
        for check_fn in checks:
            try:
                flags = await check_fn(conn, qa_run_id, manifest_id)
                doc_flags.extend(flags)
            except Exception as e:
                print(f"  [WARN] {filename}: {check_fn.__name__} failed: {e}")

        for flag in doc_flags:
            page_range = None
            if flag["page_start"] is not None and flag["page_end"] is not None:
                page_range = Range(flag["page_start"], flag["page_end"] + 1)

            await conn.execute(
                """
                INSERT INTO kb.pdf_health_flags
                  (qa_run_id, manifest_id, flag_type, severity, page_range, details)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                qa_run_id, manifest_id,
                flag["flag_type"],
                flag["severity"],
                page_range,
                json.dumps(flag["details"]),
            )

            by_type[flag["flag_type"]] = by_type.get(flag["flag_type"], 0) + 1
            total_flags += 1

        if doc_flags:
            types = [f["flag_type"] for f in doc_flags]
            print(f"  [FLAG] {filename}: {', '.join(types)}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"HEALTH FLAGS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total flags: {total_flags}")
    for ft, cnt in sorted(by_type.items()):
        print(f"  {ft}: {cnt}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
