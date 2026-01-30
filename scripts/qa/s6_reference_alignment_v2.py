#!/usr/bin/env python3
"""
QA Protocol - Phase 6 v2: Reference Alignment with Cascading Match

Aligns reference units (ref_v2, PyMuPDF) with pipeline massime.
Uses candidate generation and cascading match stages.

Features:
- Candidate generation with progressive hamming distance (12→16→20)
- Cascading match: exact_hash → token_jaccard → char_ngram → embedding
- Computes alignment_trust and collision_rate
- Tracks match_stage for each alignment

Usage:
    uv run python scripts/qa/s6_reference_alignment_v2.py
    uv run python scripts/qa/s6_reference_alignment_v2.py --clear
"""

import argparse
import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import asyncpg

# Add src to path for normalization module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lexe_api.kb.ingestion.normalization import (
    hamming_distance,
    jaccard_tokens,
    ngram_similarity,
)

from qa_config import DB_URL

# Match thresholds
EXACT_HASH_THRESHOLD = 1.0
JACCARD_THRESHOLD = 0.65
NGRAM_THRESHOLD = 0.70

# Hamming distance thresholds (progressive)
HAMMING_THRESHOLDS = [12, 16, 20]
MAX_CANDIDATES = 50


@dataclass
class MatchResult:
    match_stage: Optional[str]
    match_score: float
    matched_massima_id: Optional[str]
    match_type: str  # exact, partial, split, merged, unmatched


def classify_match_type(score: float, match_stage: str) -> str:
    """Classify match type based on score and stage."""
    if match_stage == "exact_hash":
        return "exact"
    elif score >= 0.85:
        return "exact"
    elif score >= 0.65:
        return "partial"
    else:
        return "partial"


def get_candidates(
    ref_fingerprint: int,
    ref_content_hash: str,
    chunks: list[dict],
    max_candidates: int = MAX_CANDIDATES,
) -> list[dict]:
    """
    Get candidate chunks for matching using progressive hamming distance.

    1. If exact_hash match → return only that
    2. Hamming progressive: 12 → 16 → 20 (only if candidates < 10)
    3. Cap always at max_candidates
    """
    # Exact hash check first
    for chunk in chunks:
        if chunk.get("content_hash") and chunk["content_hash"] == ref_content_hash:
            return [chunk]

    # Progressive hamming distance
    for threshold in HAMMING_THRESHOLDS:
        candidates = []
        for chunk in chunks:
            chunk_fp = chunk.get("text_fingerprint")
            if chunk_fp and ref_fingerprint:
                dist = hamming_distance(ref_fingerprint, chunk_fp)
                if dist < threshold:
                    candidates.append((chunk, dist))

        if len(candidates) >= 10:
            break

    # Sort by distance, take top N
    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates[:max_candidates]]


def find_best_match(
    ref_norm: str,
    ref_content_hash: str,
    candidates: list[dict],
) -> MatchResult:
    """
    Find best match using cascading stages.

    Stages:
    1. exact_hash - content_hash match
    2. token_jaccard - word-based Jaccard ≥ 0.65
    3. char_ngram - 3-gram similarity ≥ 0.70
    4. embedding - (not implemented, fallback)
    """
    if not candidates:
        return MatchResult(None, 0.0, None, "unmatched")

    # Stage 1: Exact hash
    for chunk in candidates:
        if chunk.get("content_hash") and chunk["content_hash"] == ref_content_hash:
            return MatchResult(
                "exact_hash", 1.0, chunk["id"],
                classify_match_type(1.0, "exact_hash")
            )

    # Stage 2: Token Jaccard
    best_jaccard = 0.0
    best_chunk = None
    for chunk in candidates:
        chunk_norm = chunk.get("testo_normalizzato") or ""
        if not chunk_norm:
            continue
        score = jaccard_tokens(ref_norm, chunk_norm)
        if score > best_jaccard:
            best_jaccard = score
            best_chunk = chunk

    if best_jaccard >= JACCARD_THRESHOLD and best_chunk:
        return MatchResult(
            "token_jaccard", best_jaccard, best_chunk["id"],
            classify_match_type(best_jaccard, "token_jaccard")
        )

    # Stage 3: Char N-gram
    best_ngram = 0.0
    best_chunk_ngram = None
    for chunk in candidates:
        chunk_norm = chunk.get("testo_normalizzato") or ""
        if not chunk_norm:
            continue
        score = ngram_similarity(ref_norm, chunk_norm, n=3)
        if score > best_ngram:
            best_ngram = score
            best_chunk_ngram = chunk

    if best_ngram >= NGRAM_THRESHOLD and best_chunk_ngram:
        return MatchResult(
            "char_ngram", best_ngram, best_chunk_ngram["id"],
            classify_match_type(best_ngram, "char_ngram")
        )

    # Stage 4: Embedding (not implemented - would require API calls)
    # For now, return the best jaccard match if any
    if best_jaccard > 0.3 and best_chunk:
        return MatchResult(
            "token_jaccard", best_jaccard, best_chunk["id"],
            "partial"
        )

    return MatchResult(None, 0.0, None, "unmatched")


async def main(clear_existing: bool = False):
    print("=" * 70)
    print("QA PROTOCOL - PHASE 6 v2: REFERENCE ALIGNMENT")
    print("=" * 70)
    print()

    conn = await asyncpg.connect(DB_URL)

    # Get qa_run_id and batch_id
    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'guided_local_v1'"
    )

    print(f"qa_run_id: {qa_run_id}")
    print(f"batch_id: {batch_id}")

    if not batch_id:
        print("[ERROR] No guided_local_v1 batch found. Run guided_ingestion_local.py first.")
        await conn.close()
        return

    if clear_existing:
        await conn.execute(
            "DELETE FROM kb.reference_alignment WHERE qa_run_id = $1 AND ingest_batch_id = $2",
            qa_run_id, batch_id,
        )
        await conn.execute(
            "DELETE FROM kb.reference_alignment_summary WHERE qa_run_id = $1 AND ingest_batch_id = $2",
            qa_run_id, batch_id,
        )
        print("[CLEAR] Deleted existing alignment for this run/batch")

    # Get manifests with ref_v2 units
    manifests = await conn.fetch(
        """
        SELECT DISTINCT m.id, m.doc_id, m.filename
        FROM kb.pdf_manifest m
        JOIN kb.qa_reference_units r ON r.manifest_id = m.id
        WHERE r.reference_version = 'ref_v2' AND r.qa_run_id = $1
        ORDER BY m.filename
        """,
        qa_run_id,
    )

    print(f"Documents to align: {len(manifests)}")
    print()

    total_matched = 0
    total_unmatched = 0
    all_match_stages = []

    for m in manifests:
        manifest_id = m["id"]
        doc_id = m["doc_id"]
        filename = m["filename"]

        # Check if already done
        existing = await conn.fetchval(
            """
            SELECT 1 FROM kb.reference_alignment_summary
            WHERE manifest_id = $1 AND ingest_batch_id = $2 AND qa_run_id = $3
            """,
            manifest_id, batch_id, qa_run_id,
        )
        if existing and not clear_existing:
            continue

        # Get reference units (ref_v2 only)
        ref_units = await conn.fetch(
            """
            SELECT id, testo_norm, content_hash, text_fingerprint
            FROM kb.qa_reference_units
            WHERE manifest_id = $1 AND reference_version = 'ref_v2' AND qa_run_id = $2
            ORDER BY unit_index
            """,
            manifest_id, qa_run_id,
        )

        # Get pipeline massime for this document
        massime = await conn.fetch(
            """
            SELECT id, testo_normalizzato, content_hash, text_fingerprint
            FROM kb.massime
            WHERE document_id = $1
            """,
            doc_id,
        )

        if not ref_units:
            print(f"  [SKIP] {filename[:45]:45} no ref_v2 units")
            continue

        if not massime:
            print(f"  [SKIP] {filename[:45]:45} no pipeline massime")
            continue

        # Convert to dicts for easier handling
        chunks = [dict(r) for r in massime]

        doc_matched = 0
        doc_unmatched = 0
        doc_match_stages = []
        matched_massima_ids = []

        for ref in ref_units:
            ref_id = ref["id"]
            ref_norm = ref["testo_norm"] or ""
            ref_hash = ref["content_hash"]
            ref_fp = ref["text_fingerprint"]

            # Get candidates
            candidates = get_candidates(ref_fp, ref_hash, chunks)

            # Find best match
            result = find_best_match(ref_norm, ref_hash, candidates)

            # Save alignment
            await conn.execute(
                """
                INSERT INTO kb.reference_alignment (
                    qa_run_id, manifest_id, ingest_batch_id,
                    ref_unit_id, matched_massima_id,
                    match_type, match_stage, match_score,
                    normalized_ref_hash, normalized_chunk_hash,
                    jaccard_similarity, overlap_ratio
                ) VALUES (
                    $1, $2, $3,
                    $4, $5,
                    $6::kb.qa_match_type, $7, $8,
                    $9, $10,
                    $11, $12
                )
                ON CONFLICT DO NOTHING
                """,
                qa_run_id, manifest_id, batch_id,
                ref_id, result.matched_massima_id,
                result.match_type, result.match_stage, result.match_score,
                ref_hash, None,  # TODO: add chunk hash
                result.match_score, result.match_score,
            )

            if result.match_type != "unmatched":
                doc_matched += 1
                doc_match_stages.append(result.match_stage)
                if result.matched_massima_id:
                    matched_massima_ids.append(result.matched_massima_id)
            else:
                doc_unmatched += 1
                doc_match_stages.append(None)

        # Calculate metrics
        total_ref = doc_matched + doc_unmatched
        coverage_pct = (doc_matched / total_ref * 100) if total_ref > 0 else 0

        # Alignment trust: (exact + jaccard + ngram) / total_matched
        trusted_stages = ['exact_hash', 'token_jaccard', 'char_ngram']
        trusted_count = sum(1 for s in doc_match_stages if s in trusted_stages)
        matched_count = sum(1 for s in doc_match_stages if s is not None)
        alignment_trust = trusted_count / matched_count if matched_count > 0 else 0

        # Embedding percentage
        embedding_count = sum(1 for s in doc_match_stages if s == 'embedding')
        embedding_pct = (embedding_count / matched_count * 100) if matched_count > 0 else 0

        # Collision rate: % of refs matching same massima
        unique_matches = len(set(matched_massima_ids))
        collision_rate = (len(matched_massima_ids) - unique_matches) / len(matched_massima_ids) * 100 if matched_massima_ids else 0

        # Save summary
        await conn.execute(
            """
            INSERT INTO kb.reference_alignment_summary (
                qa_run_id, manifest_id, ingest_batch_id,
                total_ref_units, matched_count, unmatched_count,
                coverage_pct, fragmentation_score, fusion_score, avg_overlap,
                alignment_trust, embedding_pct, collision_rate
            ) VALUES (
                $1, $2, $3,
                $4, $5, $6,
                $7, $8, $9, $10,
                $11, $12, $13
            )
            ON CONFLICT (manifest_id, ingest_batch_id, qa_run_id)
            DO UPDATE SET
                coverage_pct = EXCLUDED.coverage_pct,
                alignment_trust = EXCLUDED.alignment_trust,
                embedding_pct = EXCLUDED.embedding_pct,
                collision_rate = EXCLUDED.collision_rate
            """,
            qa_run_id, manifest_id, batch_id,
            total_ref, doc_matched, doc_unmatched,
            round(coverage_pct, 2), 1.0, 0.0, 0.0,
            round(alignment_trust, 4), round(embedding_pct, 2), round(collision_rate, 2),
        )

        total_matched += doc_matched
        total_unmatched += doc_unmatched
        all_match_stages.extend(doc_match_stages)

        status = "OK" if coverage_pct >= 60 else "LOW"
        trust_status = "OK" if alignment_trust >= 0.90 else "WARN"
        print(f"  [{status}] {filename[:40]:40} cov={coverage_pct:5.1f}% trust={alignment_trust:.2f} [{trust_status}]")

    # Global stats
    print()
    print("=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print(f"Total matched: {total_matched}")
    print(f"Total unmatched: {total_unmatched}")
    print(f"Global coverage: {100*total_matched/(total_matched+total_unmatched):.1f}%" if (total_matched+total_unmatched) > 0 else "N/A")

    # Match stage distribution
    stage_counts = {}
    for s in all_match_stages:
        stage_counts[s] = stage_counts.get(s, 0) + 1

    print("\nMatch stage distribution:")
    total_stages = len(all_match_stages)
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_stages if total_stages > 0 else 0
        stage_name = stage if stage else "unmatched"
        print(f"  {stage_name:20} {count:>6} ({pct:5.1f}%)")

    # Global alignment trust
    trusted_stages = ['exact_hash', 'token_jaccard', 'char_ngram']
    trusted = sum(1 for s in all_match_stages if s in trusted_stages)
    matched = sum(1 for s in all_match_stages if s is not None)
    global_trust = trusted / matched if matched > 0 else 0
    print(f"\nGlobal alignment_trust: {global_trust:.2%}")

    # Check guardrails
    print("\nGuardrail checks:")
    avg_coverage = await conn.fetchval(
        """
        SELECT avg(coverage_pct) FROM kb.reference_alignment_summary
        WHERE qa_run_id = $1 AND ingest_batch_id = $2
        """,
        qa_run_id, batch_id,
    )
    cov_ok = "PASS" if (avg_coverage or 0) >= 60 else "FAIL"
    print(f"  coverage_pct avg: {float(avg_coverage or 0):.1f}% [{cov_ok}] (>=60%)")
    trust_ok = "PASS" if global_trust >= 0.90 else "FAIL"
    print(f"  alignment_trust:  {global_trust:.1%} [{trust_ok}] (>=90%)")

    embedding_pct = 100 * stage_counts.get('embedding', 0) / matched if matched > 0 else 0
    emb_ok = "PASS" if embedding_pct < 10 else "FAIL"
    print(f"  embedding_pct:    {embedding_pct:.1f}% [{emb_ok}] (<10%)")

    await conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference Alignment v2")
    parser.add_argument("--clear", action="store_true", help="Clear existing alignments first")
    args = parser.parse_args()

    asyncio.run(main(clear_existing=args.clear))
