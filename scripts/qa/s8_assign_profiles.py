"""
QA Protocol - Phase 8: Assign Ingestion Profiles

Deterministic profile assignment per PDF based on:
- Extraction quality grade
- Year conflicts
- TOC/citation chunk ratios
- OCR quality

Profiles (priority order):
1. ocr_needed: quality_score < 0.6 OR valid_chars_ratio < 0.7
2. legacy_layout_2010_2013: anno 2010-2013, valid_chars < 0.85
3. toc_heavy: >10% chunks flagged toc
4. citation_dense: >20% chunks citation_list_score > 0.5
5. clean_standard: everything else

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s8_assign_profiles.py
"""

import asyncio
import json

import asyncpg

from qa_config import DB_URL


async def get_features(conn, manifest_id: int, qa_run_id: int) -> dict:
    """Gather all features needed for profile assignment."""
    features = {}

    # Extraction quality
    eq = await conn.fetchrow(
        """
        SELECT overall_quality_score, quality_grade, valid_chars_ratio
        FROM kb.pdf_extraction_quality
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    if eq:
        features["quality_score"] = float(eq["overall_quality_score"] or 0)
        features["quality_grade"] = eq["quality_grade"]
        features["valid_chars_ratio"] = float(eq["valid_chars_ratio"] or 0)
    else:
        features["quality_score"] = 0.0
        features["quality_grade"] = "D"
        features["valid_chars_ratio"] = 0.0

    # Year
    yr = await conn.fetchrow(
        """
        SELECT anno_resolved, has_conflict
        FROM kb.pdf_year_resolution
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    features["anno"] = yr["anno_resolved"] if yr else None
    features["year_conflict"] = yr["has_conflict"] if yr else False

    # Chunk analysis: TOC and citation ratios
    chunk_stats = await conn.fetchrow(
        """
        SELECT
            count(*) as total_chunks,
            count(*) FILTER (WHERE toc_infiltration_score > 0.6) as toc_chunks,
            count(*) FILTER (WHERE citation_list_score > 0.5) as citation_chunks
        FROM kb.chunk_features
        WHERE manifest_id = $1 AND qa_run_id = $2
        """,
        manifest_id, qa_run_id,
    )
    total_chunks = chunk_stats["total_chunks"] if chunk_stats else 0
    features["total_chunks"] = total_chunks
    features["toc_ratio"] = (
        chunk_stats["toc_chunks"] / total_chunks if total_chunks > 0 else 0
    )
    features["citation_ratio"] = (
        chunk_stats["citation_chunks"] / total_chunks if total_chunks > 0 else 0
    )

    # Health flags count
    flags = await conn.fetchval(
        "SELECT count(*) FROM kb.pdf_health_flags WHERE manifest_id = $1",
        manifest_id,
    )
    features["health_flag_count"] = flags or 0

    # OCR candidate pages ratio
    ocr_stats = await conn.fetchrow(
        """
        SELECT
            count(*) as total,
            count(*) FILTER (WHERE is_ocr_candidate) as ocr_pages
        FROM kb.page_extraction_stats
        WHERE manifest_id = $1
        """,
        manifest_id,
    )
    if ocr_stats and ocr_stats["total"] > 0:
        features["ocr_page_ratio"] = ocr_stats["ocr_pages"] / ocr_stats["total"]
    else:
        features["ocr_page_ratio"] = 0.0

    return features


def assign_profile(features: dict) -> tuple[str, float]:
    """Assign ingestion profile based on features. Returns (profile, confidence)."""
    # Priority 1: OCR needed
    if features["quality_score"] < 0.6 or features["valid_chars_ratio"] < 0.7:
        return "ocr_needed", 0.9

    # Priority 2: Legacy layout (2010-2013)
    anno = features.get("anno")
    if anno and 2010 <= anno <= 2013 and features["valid_chars_ratio"] < 0.85:
        return "legacy_layout_2010_2013", 0.85

    # Priority 3: TOC heavy
    if features["toc_ratio"] > 0.10:
        return "toc_heavy", 0.85

    # Priority 4: Citation dense
    if features["citation_ratio"] > 0.20:
        return "citation_dense", 0.80

    # Default: clean standard
    return "clean_standard", 0.95


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 8: ASSIGN PROFILES")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] qa_run_id={qa_run_id}")

    manifests = await conn.fetch(
        "SELECT id, filename FROM kb.pdf_manifest WHERE qa_run_id = $1",
        qa_run_id,
    )
    print(f"[OK] Found {len(manifests)} manifest entries")

    profile_counts = {}

    for m in manifests:
        manifest_id = m["id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            "SELECT 1 FROM kb.qa_ingestion_profiles WHERE manifest_id = $1",
            manifest_id,
        )
        if existing:
            continue

        features = await get_features(conn, manifest_id, qa_run_id)
        profile, confidence = assign_profile(features)

        await conn.execute(
            """
            INSERT INTO kb.qa_ingestion_profiles
              (qa_run_id, manifest_id, profile, confidence, features)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (manifest_id) DO NOTHING
            """,
            qa_run_id, manifest_id, profile, confidence,
            json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()}),
        )

        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        print(f"  [{profile}] {filename} (conf={confidence:.2f})")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"PROFILE ASSIGNMENT COMPLETE")
    print(f"{'=' * 70}")
    for p, cnt in sorted(profile_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {cnt}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
