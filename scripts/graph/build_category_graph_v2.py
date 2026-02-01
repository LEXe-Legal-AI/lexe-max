#!/usr/bin/env python3
"""
Build Category Graph v2.5 - IDEMPOTENT + VERBOSE + BATCH COMMIT

Full corpus classification using three-axis taxonomy with:
- Isotonic calibration for confidence scores
- Ensemble LLM resolver (2 classifiers + judge)
- Structured JSONL logging
- Stratified subset testing
- IDEMPOTENT: skips already processed massime
- BATCH COMMIT: saves after each batch (crash-safe)
- VERBOSE: detailed progress logging

Usage:
    # Dry run (no DB writes)
    uv run python scripts/graph/build_category_graph_v2.py --dry-run

    # Resume existing run
    uv run python scripts/graph/build_category_graph_v2.py --resume-run 42 --commit

    # Test run v2.5 on stratified subset
    OPENROUTER_API_KEY="sk-or-..." uv run python scripts/graph/build_category_graph_v2.py \
        --v25 --subset-test --subset-size 2000 --llm-budget 500 --commit

    # Full corpus v2.5 with LLM resolver
    OPENROUTER_API_KEY="sk-or-..." uv run python scripts/graph/build_category_graph_v2.py \
        --v25 --batch-size 500 --llm-budget 8000 --commit
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import asyncpg
import httpx
import numpy as np


def log_info(msg: str, prefix: str = "INFO"):
    """Verbose logging with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}")


def log_warn(msg: str):
    log_info(msg, "WARN")


def log_error(msg: str):
    log_info(msg, "ERROR")


def log_success(msg: str):
    log_info(msg, "OK")


def log_progress(current: int, total: int, extra: str = ""):
    pct = (current / total * 100) if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    log_info(f"[{bar}] {current}/{total} ({pct:.1f}%) {extra}", "PROG")


def parse_embedding(emb) -> np.ndarray:
    """Parse embedding from database format to numpy array."""
    if isinstance(emb, str):
        return np.array(json.loads(emb), dtype=np.float32)
    elif isinstance(emb, (list, tuple)):
        return np.array(emb, dtype=np.float32)
    else:
        return np.array(emb, dtype=np.float32)


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.category_classifier_v2 import (
    classify_massima,
    classify_massima_v25,
    set_centroids,
    ClassificationResult,
)
from src.lexe_api.kb.graph.config import DEFAULT_THRESHOLDS
from src.lexe_api.kb.graph.classification_logger import (
    ClassificationLogger,
    ClassificationStats,
)


async def get_or_create_run(
    conn: asyncpg.Connection,
    dry_run: bool,
    resume_run_id: Optional[int] = None,
    is_test: bool = False,
    subset_ids: Optional[List[str]] = None,
    version: str = "2.5",
) -> Tuple[int, bool]:
    """
    Get existing run or create new one.
    Returns: (run_id, is_resumed)
    """
    if dry_run:
        return 0, False

    # Resume existing run?
    if resume_run_id is not None:
        existing = await conn.fetchrow("""
            SELECT id, is_active, config FROM kb.graph_runs
            WHERE id = $1 AND run_type = 'category_v2'
        """, resume_run_id)
        if existing:
            log_success(f"Resuming run_id={resume_run_id}")
            return resume_run_id, True
        else:
            log_error(f"Run {resume_run_id} not found!")
            sys.exit(1)

    # Create new run
    config = {
        "version": version,
        "is_test": is_test,
        "started_at": datetime.now().isoformat(),
    }
    if subset_ids:
        config["subset_size"] = len(subset_ids)
        config["subset_sample"] = subset_ids[:10] + subset_ids[-10:] if len(subset_ids) > 20 else subset_ids

    run_id = await conn.fetchval("""
        INSERT INTO kb.graph_runs (run_type, is_active, config)
        VALUES ('category_v2', FALSE, $1::jsonb)
        RETURNING id
    """, json.dumps(config))

    log_success(f"Created new run_id={run_id}")
    return run_id, False


async def get_already_processed(
    conn: asyncpg.Connection,
    run_id: int,
) -> Set[str]:
    """Get set of massima IDs already processed in this run."""
    if run_id == 0:
        return set()

    rows = await conn.fetch("""
        SELECT massima_id::text FROM kb.category_predictions_v2
        WHERE run_id = $1
    """, run_id)

    processed = {r["massima_id"] for r in rows}
    if processed:
        log_info(f"Found {len(processed)} already processed massime (will skip)")
    return processed


async def compute_centroids_from_golden(
    conn: asyncpg.Connection,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute centroid embeddings from golden set (train split only)."""
    log_info("Computing centroids from golden set (train split)...")

    rows = await conn.fetch("""
        SELECT
            ga.materia_l1,
            ga.natura_l1,
            ga.ambito_l1,
            e.embedding
        FROM kb.golden_category_adjudicated_v2 ga
        JOIN kb.embeddings e ON e.massima_id = ga.massima_id
        WHERE ga.split = 'train'
          AND ga.materia_l1 != 'PENDING'
          AND e.model_name = 'openai/text-embedding-3-small'
    """)

    if not rows:
        log_warn("No golden set with embeddings found!")
        return {}, {}, {}

    log_info(f"Found {len(rows)} golden samples with embeddings")

    # Group embeddings by category
    materia_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
    natura_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
    ambito_groups: Dict[str, List[np.ndarray]] = defaultdict(list)

    for row in rows:
        emb = parse_embedding(row["embedding"])
        materia_groups[row["materia_l1"]].append(emb)
        natura_groups[row["natura_l1"]].append(emb)
        if row["ambito_l1"]:
            ambito_groups[row["ambito_l1"]].append(emb)

    def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros(1536)
        return np.mean(np.stack(embeddings), axis=0)

    materia_centroids = {k: compute_centroid(v) for k, v in materia_groups.items()}
    natura_centroids = {k: compute_centroid(v) for k, v in natura_groups.items()}
    ambito_centroids = {k: compute_centroid(v) for k, v in ambito_groups.items()}

    log_info(f"  Materia centroids: {list(materia_centroids.keys())}")
    log_info(f"  Natura centroids: {list(natura_centroids.keys())}")
    log_info(f"  Ambito centroids: {list(ambito_centroids.keys())}")

    return materia_centroids, natura_centroids, ambito_centroids


async def fetch_stratified_subset(
    conn: asyncpg.Connection,
    subset_size: int,
) -> List[str]:
    """Fetch stratified subset of massima IDs."""
    log_info(f"Selecting stratified subset of {subset_size} massime...")

    per_strata = subset_size // 4

    rows = await conn.fetch("""
        WITH stratified AS (
            SELECT
                mf.massima_id::text as massima_id,
                mf.sezione,
                CASE
                    WHEN mf.sezione = 'L' THEN 'lavoro'
                    WHEN mf.sezione = 'U' THEN 'unite'
                    WHEN mf.sezione IN ('1','2','3','4','6') THEN 'civile'
                    WHEN mf.sezione = '5' THEN 'tributaria'
                    ELSE 'other'
                END as strata,
                ROW_NUMBER() OVER (
                    PARTITION BY CASE
                        WHEN mf.sezione = 'L' THEN 'lavoro'
                        WHEN mf.sezione = 'U' THEN 'unite'
                        WHEN mf.sezione IN ('1','2','3','4','6') THEN 'civile'
                        WHEN mf.sezione = '5' THEN 'tributaria'
                        ELSE 'other'
                    END
                    ORDER BY RANDOM()
                ) as rn
            FROM kb.massime_features_v2 mf
            JOIN kb.embeddings e ON e.massima_id = mf.massima_id
            WHERE e.model_name = 'openai/text-embedding-3-small'
        )
        SELECT massima_id, strata
        FROM stratified
        WHERE rn <= $1
        ORDER BY strata, massima_id
    """, per_strata)

    strata_counts = defaultdict(int)
    ids = []
    for row in rows:
        ids.append(row["massima_id"])
        strata_counts[row["strata"]] += 1

    log_info(f"Selected {len(ids)} massime:")
    for strata, count in sorted(strata_counts.items()):
        log_info(f"    {strata}: {count}")

    return ids


async def fetch_massime_batch(
    conn: asyncpg.Connection,
    offset: int,
    batch_size: int,
    subset_ids: Optional[List[str]] = None,
    already_processed: Optional[Set[str]] = None,
) -> List[Dict]:
    """Fetch a batch of massime with features and embeddings, skipping already processed."""
    if subset_ids:
        batch_ids = subset_ids[offset:offset + batch_size]
        if not batch_ids:
            return []

        # Filter out already processed
        if already_processed:
            batch_ids = [i for i in batch_ids if i not in already_processed]
            if not batch_ids:
                return []

        rows = await conn.fetch("""
            SELECT
                mf.massima_id,
                mf.sezione,
                mf.tipo,
                mf.testo_trunc,
                mf.norms_canonical,
                mf.norms_count,
                e.embedding
            FROM kb.massime_features_v2 mf
            JOIN kb.embeddings e ON e.massima_id = mf.massima_id
            WHERE e.model_name = 'openai/text-embedding-3-small'
              AND mf.massima_id = ANY($1::uuid[])
            ORDER BY mf.massima_id
        """, batch_ids)
    else:
        rows = await conn.fetch("""
            SELECT
                mf.massima_id,
                mf.sezione,
                mf.tipo,
                mf.testo_trunc,
                mf.norms_canonical,
                mf.norms_count,
                e.embedding
            FROM kb.massime_features_v2 mf
            JOIN kb.embeddings e ON e.massima_id = mf.massima_id
            WHERE e.model_name = 'openai/text-embedding-3-small'
            ORDER BY mf.massima_id
            OFFSET $1 LIMIT $2
        """, offset, batch_size)

    # Filter out already processed (for non-subset case)
    if already_processed and not subset_ids:
        rows = [r for r in rows if str(r["massima_id"]) not in already_processed]

    return [dict(r) for r in rows]


async def store_predictions_batch(
    conn: asyncpg.Connection,
    run_id: int,
    results: List[ClassificationResult],
    dry_run: bool,
):
    """Store classification results in batch with immediate commit."""
    if dry_run or not results:
        return

    values = []
    for r in results:
        values.append((
            r.massima_id,
            run_id,
            r.materia_l1,
            r.materia_confidence,
            r.materia_rule,
            r.materia_candidate_set,
            r.materia_reasons,
            r.natura_l1,
            r.natura_confidence,
            r.natura_rule,
            r.ambito_l1,
            r.ambito_confidence,
            r.ambito_rule,
            r.topic_l2,
            r.topic_l2_confidence,
            r.topic_l2_flag,
            r.abstain_reason,
            r.composite_confidence,
            r.norms_count,
        ))

    await conn.executemany("""
        INSERT INTO kb.category_predictions_v2 (
            massima_id, run_id,
            materia_l1, materia_confidence, materia_rule,
            materia_candidate_set, materia_reasons,
            natura_l1, natura_confidence, natura_rule,
            ambito_l1, ambito_confidence, ambito_rule,
            topic_l2, topic_l2_confidence, topic_l2_flag, abstain_reason,
            composite_confidence, norms_count
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19
        )
        ON CONFLICT (massima_id, run_id) DO UPDATE SET
            materia_l1 = EXCLUDED.materia_l1,
            materia_confidence = EXCLUDED.materia_confidence,
            materia_rule = EXCLUDED.materia_rule
    """, values)

    log_success(f"Committed {len(results)} predictions to DB")


async def update_run_progress(
    conn: asyncpg.Connection,
    run_id: int,
    processed: int,
    total: int,
    stats: dict,
    dry_run: bool,
):
    """Update run metadata with progress."""
    if dry_run or run_id == 0:
        return

    progress = {
        "processed": processed,
        "total": total,
        "pct": round(processed / total * 100, 1) if total > 0 else 0,
        "last_update": datetime.now().isoformat(),
        "materia_dist": dict(stats.get("materia_dist", {})),
        "llm_calls": stats.get("llm_calls", 0),
    }

    await conn.execute("""
        UPDATE kb.graph_runs
        SET metrics = COALESCE(metrics, '{}'::jsonb) || $2::jsonb
        WHERE id = $1
    """, run_id, json.dumps({"progress": progress}))


async def activate_run(conn: asyncpg.Connection, run_id: int):
    """Activate the run and deactivate previous runs."""
    await conn.execute("""
        UPDATE kb.graph_runs
        SET is_active = FALSE
        WHERE run_type = 'category_v2' AND is_active = TRUE
    """)

    await conn.execute("""
        UPDATE kb.graph_runs
        SET is_active = TRUE
        WHERE id = $1
    """, run_id)


async def main(
    batch_size: int = 500,
    llm_budget: int = 8000,
    dry_run: bool = True,
    use_v25: bool = False,
    subset_test: bool = False,
    subset_size: int = 2000,
    resume_run_id: Optional[int] = None,
    api_key_arg: Optional[str] = None,
):
    """Main classification routine."""

    # Try to load API key from: 1) argument, 2) env var, 3) .env.local file
    api_key = api_key_arg or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_file = Path(__file__).parent.parent.parent / ".env.local"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        log_warn("OPENROUTER_API_KEY not set. LLM resolver disabled.")
    else:
        log_success(f"API key loaded (ends with ...{api_key[-8:]})")

    settings = KBSettings()

    log_info("="*60)
    log_info(f"BUILD CATEGORY GRAPH v{'2.5' if use_v25 else '2.4'}")
    log_info("="*60)
    log_info(f"  Mode: {'DRY RUN' if dry_run else 'COMMIT'}")
    log_info(f"  Pipeline: {'v2.5 (calibration + ensemble)' if use_v25 else 'v2.4 (legacy)'}")
    log_info(f"  Batch size: {batch_size}")
    log_info(f"  LLM budget: {llm_budget}")
    log_info(f"  Resume run: {resume_run_id or 'NEW'}")

    conn = await asyncpg.connect(settings.kb_database_url)
    log_success("Connected to KB database")

    try:
        # Check prerequisites
        view_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.views
                WHERE table_schema = 'kb' AND table_name = 'massime_features_v2'
            )
        """)
        if not view_exists:
            log_error("View kb.massime_features_v2 does not exist!")
            log_error("Run migration 009_category_v2.sql first.")
            return

        # Get or create run
        run_id, is_resumed = await get_or_create_run(
            conn, dry_run,
            resume_run_id=resume_run_id,
            is_test=subset_test,
            subset_ids=None,  # Will be set later
            version="2.5" if use_v25 else "2.4",
        )

        # Get already processed massime (for idempotency)
        already_processed = await get_already_processed(conn, run_id)

        # Get stratified subset if requested
        subset_ids: Optional[List[str]] = None
        if subset_test:
            subset_ids = await fetch_stratified_subset(conn, subset_size)
            total = len(subset_ids)
            # Remove already processed from subset
            subset_ids = [i for i in subset_ids if i not in already_processed]
            log_info(f"After filtering: {len(subset_ids)} remaining to process")
        else:
            total = await conn.fetchval("""
                SELECT COUNT(*) FROM kb.massime_features_v2 mf
                JOIN kb.embeddings e ON e.massima_id = mf.massima_id
                WHERE e.model_name = 'openai/text-embedding-3-small'
            """)
            remaining = total - len(already_processed)
            log_info(f"Total: {total}, Already processed: {len(already_processed)}, Remaining: {remaining}")

        log_info(f"Run ID: {run_id}")
        log_info("="*60)

        # Compute centroids from golden set
        materia_c, natura_c, ambito_c = await compute_centroids_from_golden(conn)

        if materia_c:
            set_centroids(materia_c, natura_c, ambito_c)
            log_success("Centroids loaded into classifier")
        else:
            log_warn("No centroids available. Using rule-based + LLM only.")

        # Stats tracking
        stats = {
            "total": 0,
            "rule_materia": 0,
            "centroid_materia": 0,
            "llm_materia": 0,
            "processuale": 0,
            "llm_calls": 0,
            "materia_dist": defaultdict(int),
            "natura_dist": defaultdict(int),
            "ambito_dist": defaultdict(int),
            "errors": 0,
        }

        start_time = time.time()
        llm_calls_remaining = llm_budget

        # Initialize classification stats for v2.5
        class_stats = ClassificationStats() if use_v25 else None

        # Create logger for v2.5
        logger = ClassificationLogger(run_id or 0) if use_v25 and not dry_run else None

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                offset = 0
                batch_num = 0
                effective_total = len(subset_ids) if subset_ids else (total - len(already_processed))

                while True:
                    batch_num += 1
                    batch_start = time.time()

                    batch = await fetch_massime_batch(
                        conn, offset, batch_size, subset_ids, already_processed
                    )

                    if not batch:
                        log_info("No more massime to process")
                        break

                    log_info(f"")
                    log_info(f"{'='*40}")
                    log_info(f"BATCH {batch_num}: {len(batch)} massime")
                    log_info(f"{'='*40}")

                    results = []
                    batch_errors = 0

                    for i, row in enumerate(batch):
                        try:
                            massima_id = row["massima_id"]
                            embedding = parse_embedding(row["embedding"])
                            tipo = row["tipo"]
                            sezione = row["sezione"]
                            norms = row["norms_canonical"] or []
                            testo = row["testo_trunc"] or ""
                            norms_count = row["norms_count"] or 0

                            # Decide whether to use LLM
                            use_llm = api_key and llm_calls_remaining > 0

                            if use_v25:
                                result, log_entry = await classify_massima_v25(
                                    massima_id, embedding, tipo, sezione, norms, testo, norms_count,
                                    client if use_llm else None,
                                    api_key if use_llm else None,
                                    DEFAULT_THRESHOLDS,
                                )
                                results.append(result)

                                if logger:
                                    logger.log(log_entry)
                                if class_stats:
                                    class_stats.record(log_entry)

                                # Update LLM budget
                                if log_entry.llm_called:
                                    llm_cost = 2 + (1 if log_entry.llm_judge_called else 0)
                                    llm_calls_remaining -= llm_cost
                                    stats["llm_calls"] += llm_cost
                            else:
                                if use_llm:
                                    llm_calls_remaining -= 1

                                result = await classify_massima(
                                    massima_id, embedding, tipo, sezione, norms, testo, norms_count,
                                    client if use_llm else None,
                                    api_key if use_llm else None,
                                )
                                results.append(result)

                            # Update basic stats
                            stats["total"] += 1
                            stats["materia_dist"][result.materia_l1] += 1
                            stats["natura_dist"][result.natura_l1] += 1

                            if "rule" in result.materia_rule or "tipo" in result.materia_rule:
                                stats["rule_materia"] += 1
                            elif "centroid" in result.materia_rule:
                                stats["centroid_materia"] += 1
                            elif "llm" in result.materia_rule:
                                stats["llm_materia"] += 1
                                stats["llm_calls"] += 1

                            if result.natura_l1 == "PROCESSUALE":
                                stats["processuale"] += 1
                                if result.ambito_l1:
                                    stats["ambito_dist"][result.ambito_l1] += 1

                            # Progress within batch
                            if (i + 1) % 100 == 0:
                                log_info(f"  Batch progress: {i+1}/{len(batch)}")

                        except Exception as e:
                            batch_errors += 1
                            stats["errors"] += 1
                            log_error(f"Error processing {row.get('massima_id', '?')}: {e}")
                            continue

                    # COMMIT BATCH IMMEDIATELY
                    await store_predictions_batch(conn, run_id, results, dry_run)

                    # Update run progress
                    await update_run_progress(conn, run_id, stats["total"], effective_total, stats, dry_run)

                    # Batch stats
                    batch_elapsed = time.time() - batch_start
                    batch_rate = len(batch) / batch_elapsed if batch_elapsed > 0 else 0

                    log_info(f"")
                    log_info(f"Batch {batch_num} complete:")
                    log_info(f"  Processed: {len(results)}/{len(batch)} ({batch_errors} errors)")
                    log_info(f"  Time: {batch_elapsed:.1f}s ({batch_rate:.1f}/s)")
                    log_info(f"  LLM budget remaining: {llm_calls_remaining}")

                    # Overall progress
                    log_progress(stats["total"], effective_total, f"LLM calls: {stats['llm_calls']}")

                    offset += batch_size

                    # Check LLM budget
                    if llm_calls_remaining <= 0:
                        log_warn("LLM budget exhausted!")

        finally:
            if logger:
                logger.close()
                log_success(f"Logs saved to: {logger.log_file_path}")

        # Activate run (only for non-test runs)
        if not dry_run and not subset_test:
            await activate_run(conn, run_id)
            log_success(f"Activated run_id {run_id}")
        elif not dry_run and subset_test:
            log_info(f"Test run {run_id} NOT activated (is_test=true)")

        # Print final stats
        elapsed = time.time() - start_time

        log_info("")
        log_info("="*60)
        log_info("BUILD SUMMARY")
        log_info("="*60)
        log_info(f"  Run ID: {run_id}")
        log_info(f"  Total classified: {stats['total']}")
        log_info(f"  Errors: {stats['errors']}")
        log_info(f"  Elapsed time: {elapsed:.1f}s")
        log_info(f"  Rate: {stats['total'] / elapsed:.1f} massime/s" if elapsed > 0 else "  Rate: N/A")

        if stats['total'] > 0:
            log_info("")
            log_info("Materia derivation method:")
            log_info(f"    Rule-based: {stats['rule_materia']} ({100*stats['rule_materia']/stats['total']:.1f}%)")
            log_info(f"    Centroid: {stats['centroid_materia']} ({100*stats['centroid_materia']/stats['total']:.1f}%)")
            log_info(f"    LLM resolver: {stats['llm_materia']} ({100*stats['llm_materia']/stats['total']:.1f}%)")

            log_info("")
            log_info("Materia distribution:")
            for mat, cnt in sorted(stats["materia_dist"].items(), key=lambda x: -x[1]):
                log_info(f"    {mat}: {cnt} ({100*cnt/stats['total']:.1f}%)")

            log_info("")
            log_info("Natura distribution:")
            for nat, cnt in sorted(stats["natura_dist"].items(), key=lambda x: -x[1]):
                log_info(f"    {nat}: {cnt} ({100*cnt/stats['total']:.1f}%)")

            log_info(f"")
            log_info(f"Processuale: {stats['processuale']} ({100*stats['processuale']/stats['total']:.1f}%)")

            if stats["ambito_dist"] and stats['processuale'] > 0:
                log_info("")
                log_info("Ambito distribution (processuale only):")
                for amb, cnt in sorted(stats["ambito_dist"].items(), key=lambda x: -x[1]):
                    log_info(f"    {amb}: {cnt} ({100*cnt/stats['processuale']:.1f}%)")

        log_info(f"")
        log_info(f"Total LLM calls: {stats['llm_calls']}")

        if use_v25 and class_stats:
            class_stats.print_summary()

        log_info("")
        log_success("BUILD COMPLETE!")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Category Graph v2.5 (IDEMPOTENT)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--llm-budget", type=int, default=8000, help="Max LLM calls")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit")
    parser.add_argument("--commit", action="store_true", help="Commit to database")
    parser.add_argument("--v25", action="store_true", help="Use v2.5 pipeline (calibration + ensemble LLM)")
    parser.add_argument("--subset-test", action="store_true", help="Run on stratified subset only")
    parser.add_argument("--subset-size", type=int, default=2000, help="Subset size for test run")
    parser.add_argument("--resume-run", type=int, default=None, help="Resume existing run by ID")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key")
    args = parser.parse_args()

    if not args.commit and not args.dry_run:
        print("Specify --dry-run or --commit")
        sys.exit(1)

    dry_run = not args.commit
    asyncio.run(main(
        batch_size=args.batch_size,
        llm_budget=args.llm_budget,
        dry_run=dry_run,
        use_v25=args.v25,
        subset_test=args.subset_test,
        subset_size=args.subset_size,
        resume_run_id=args.resume_run,
        api_key_arg=args.api_key,
    ))
