#!/usr/bin/env python3
"""
Category Graph v2.4 Validation

Evaluates predictions against held-out test set (180 samples).
Quality gates from plan v2.4.

L1 GATES (mandatory):
- Materia L1 coverage = 100%
- Materia L1 accuracy >= 0.95
- Natura L1 coverage = 100%
- Natura L1 accuracy >= 0.90
- Top-2 accuracy (materia) >= 0.99
- Calibration error (ECE) < 0.05

AMBITO GATE (on PROCESSUALE subset):
- Ambito L1 coverage >= 0.95
- Ambito unknown rate <= 0.05

L2 TOPIC GATES (quality):
- L2 abstain rate <= 0.40
- L2 precision (audit) >= 0.90

Usage:
    uv run python scripts/qa/validate_category_v2.py
    uv run python scripts/qa/validate_category_v2.py --run-id 123
    uv run python scripts/qa/validate_category_v2.py --run-id latest
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings


@dataclass
class ValidationResult:
    """Result of validation run."""
    run_id: int

    # L1 Gates
    materia_coverage: float
    materia_accuracy: float
    natura_coverage: float
    natura_accuracy: float
    top2_accuracy: float
    calibration_error: float

    # Ambito Gate
    ambito_coverage: float
    ambito_unknown_rate: float

    # L2 Gates
    l2_abstain_rate: float
    l2_precision_audit: Optional[float]
    l2_precision_auto: Optional[float]

    # Pass/Fail
    all_gates_passed: bool
    failed_gates: List[str]


def compute_ece(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Lower is better. Target: < 0.05
    """
    if not confidences or not correct:
        return 0.0

    confidences = np.array(confidences)
    correct = np.array(correct)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(correct[in_bin])
            ece += np.abs(avg_acc - avg_conf) * prop_in_bin

    return float(ece)


async def fetch_test_predictions(
    conn: asyncpg.Connection,
    run_id: int,
) -> List[Dict]:
    """Fetch predictions for test set samples."""
    rows = await conn.fetch("""
        SELECT
            p.massima_id,
            p.materia_l1 AS pred_materia,
            p.materia_confidence,
            p.materia_candidate_set,
            p.natura_l1 AS pred_natura,
            p.natura_confidence,
            p.ambito_l1 AS pred_ambito,
            p.ambito_confidence,
            p.topic_l2 AS pred_topic,
            p.topic_l2_confidence,
            p.topic_l2_flag,
            p.composite_confidence,
            ga.materia_l1 AS true_materia,
            ga.natura_l1 AS true_natura,
            ga.ambito_l1 AS true_ambito,
            ga.topic_l2 AS true_topic,
            ga.difficulty_bucket
        FROM kb.category_predictions_v2 p
        JOIN kb.golden_category_adjudicated_v2 ga ON ga.massima_id = p.massima_id
        WHERE p.run_id = $1
          AND ga.split = 'test'
          AND ga.materia_l1 != 'PENDING'
    """, run_id)

    return [dict(r) for r in rows]


async def fetch_full_corpus_stats(
    conn: asyncpg.Connection,
    run_id: int,
) -> Dict:
    """Fetch statistics from full corpus."""
    stats = {}

    # Total predictions
    stats["total"] = await conn.fetchval("""
        SELECT COUNT(*) FROM kb.category_predictions_v2 WHERE run_id = $1
    """, run_id)

    # Materia distribution
    materia_rows = await conn.fetch("""
        SELECT materia_l1, COUNT(*) as cnt
        FROM kb.category_predictions_v2
        WHERE run_id = $1
        GROUP BY materia_l1
    """, run_id)
    stats["materia_dist"] = {r["materia_l1"]: r["cnt"] for r in materia_rows}

    # Natura distribution
    natura_rows = await conn.fetch("""
        SELECT natura_l1, COUNT(*) as cnt
        FROM kb.category_predictions_v2
        WHERE run_id = $1
        GROUP BY natura_l1
    """, run_id)
    stats["natura_dist"] = {r["natura_l1"]: r["cnt"] for r in natura_rows}

    # Processuale count
    stats["processuale"] = stats["natura_dist"].get("PROCESSUALE", 0)

    # Ambito distribution (processuale only)
    ambito_rows = await conn.fetch("""
        SELECT ambito_l1, COUNT(*) as cnt
        FROM kb.category_predictions_v2
        WHERE run_id = $1 AND natura_l1 = 'PROCESSUALE'
        GROUP BY ambito_l1
    """, run_id)
    stats["ambito_dist"] = {r["ambito_l1"]: r["cnt"] for r in ambito_rows}

    # Confidence distributions
    stats["avg_materia_conf"] = await conn.fetchval("""
        SELECT AVG(materia_confidence)
        FROM kb.category_predictions_v2 WHERE run_id = $1
    """, run_id)

    stats["avg_composite_conf"] = await conn.fetchval("""
        SELECT AVG(composite_confidence)
        FROM kb.category_predictions_v2 WHERE run_id = $1
    """, run_id)

    return stats


def validate_predictions(predictions: List[Dict]) -> ValidationResult:
    """Run all quality gate checks on predictions."""

    failed_gates = []
    n = len(predictions)

    if n == 0:
        return ValidationResult(
            run_id=0,
            materia_coverage=0.0,
            materia_accuracy=0.0,
            natura_coverage=0.0,
            natura_accuracy=0.0,
            top2_accuracy=0.0,
            calibration_error=1.0,
            ambito_coverage=0.0,
            ambito_unknown_rate=1.0,
            l2_abstain_rate=1.0,
            l2_precision_audit=None,
            l2_precision_auto=None,
            all_gates_passed=False,
            failed_gates=["no_predictions"],
        )

    # --- L1 GATES ---

    # Materia coverage (should be 100% since always assigned)
    materia_assigned = sum(1 for p in predictions if p["pred_materia"])
    materia_coverage = materia_assigned / n
    if materia_coverage < 1.0:
        failed_gates.append("materia_coverage")

    # Materia accuracy
    materia_correct = sum(1 for p in predictions if p["pred_materia"] == p["true_materia"])
    materia_accuracy = materia_correct / n
    if materia_accuracy < 0.95:
        failed_gates.append("materia_accuracy")

    # Top-2 materia accuracy
    top2_correct = 0
    for p in predictions:
        candidates = p["materia_candidate_set"] or []
        if p["true_materia"] in candidates or p["pred_materia"] == p["true_materia"]:
            top2_correct += 1
    top2_accuracy = top2_correct / n
    if top2_accuracy < 0.99:
        failed_gates.append("top2_accuracy")

    # Natura coverage
    natura_assigned = sum(1 for p in predictions if p["pred_natura"])
    natura_coverage = natura_assigned / n
    if natura_coverage < 1.0:
        failed_gates.append("natura_coverage")

    # Natura accuracy
    natura_correct = sum(1 for p in predictions if p["pred_natura"] == p["true_natura"])
    natura_accuracy = natura_correct / n
    if natura_accuracy < 0.90:
        failed_gates.append("natura_accuracy")

    # Calibration error (ECE)
    confidences = [p["materia_confidence"] for p in predictions]
    correct = [p["pred_materia"] == p["true_materia"] for p in predictions]
    calibration_error = compute_ece(confidences, correct)
    if calibration_error >= 0.05:
        failed_gates.append("calibration_error")

    # --- AMBITO GATE (processuale only) ---

    processuale_preds = [p for p in predictions if p["pred_natura"] == "PROCESSUALE"]
    n_proc = len(processuale_preds)

    if n_proc > 0:
        ambito_assigned = sum(1 for p in processuale_preds if p["pred_ambito"])
        ambito_coverage = ambito_assigned / n_proc

        ambito_unknown = sum(1 for p in processuale_preds if p["pred_ambito"] == "UNKNOWN")
        ambito_unknown_rate = ambito_unknown / n_proc

        if ambito_coverage < 0.95:
            failed_gates.append("ambito_coverage")
        if ambito_unknown_rate > 0.05:
            failed_gates.append("ambito_unknown_rate")
    else:
        ambito_coverage = 1.0
        ambito_unknown_rate = 0.0

    # --- L2 TOPIC GATES ---

    # Abstain rate
    l2_with_topic = sum(1 for p in predictions if p["pred_topic"])
    l2_abstain_rate = 1.0 - (l2_with_topic / n)
    if l2_abstain_rate > 0.40:
        failed_gates.append("l2_abstain_rate")

    # L2 precision (on auto-assigned only)
    auto_topics = [p for p in predictions if p["topic_l2_flag"] == "auto" and p["pred_topic"]]
    if auto_topics:
        auto_correct = sum(1 for p in auto_topics if p["pred_topic"] == p["true_topic"])
        l2_precision_auto = auto_correct / len(auto_topics) if auto_topics else None
    else:
        l2_precision_auto = None

    # L2 precision audit (all assigned)
    assigned_topics = [p for p in predictions if p["pred_topic"] and p["true_topic"]]
    if assigned_topics:
        audit_correct = sum(1 for p in assigned_topics if p["pred_topic"] == p["true_topic"])
        l2_precision_audit = audit_correct / len(assigned_topics)
        if l2_precision_audit < 0.90:
            failed_gates.append("l2_precision_audit")
    else:
        l2_precision_audit = None

    return ValidationResult(
        run_id=0,
        materia_coverage=round(materia_coverage, 4),
        materia_accuracy=round(materia_accuracy, 4),
        natura_coverage=round(natura_coverage, 4),
        natura_accuracy=round(natura_accuracy, 4),
        top2_accuracy=round(top2_accuracy, 4),
        calibration_error=round(calibration_error, 4),
        ambito_coverage=round(ambito_coverage, 4),
        ambito_unknown_rate=round(ambito_unknown_rate, 4),
        l2_abstain_rate=round(l2_abstain_rate, 4),
        l2_precision_audit=round(l2_precision_audit, 4) if l2_precision_audit else None,
        l2_precision_auto=round(l2_precision_auto, 4) if l2_precision_auto else None,
        all_gates_passed=len(failed_gates) == 0,
        failed_gates=failed_gates,
    )


async def store_eval_run(
    conn: asyncpg.Connection,
    run_id: int,
    result: ValidationResult,
):
    """Store evaluation results in database."""
    await conn.execute("""
        INSERT INTO kb.category_v2_eval_runs (
            run_id, materia_coverage, materia_accuracy,
            natura_coverage, natura_accuracy, top2_accuracy, calibration_error,
            ambito_coverage, ambito_unknown_rate,
            l2_abstain_rate, l2_precision_audit, l2_precision_auto,
            all_gates_passed, failed_gates
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
        )
    """,
        run_id,
        result.materia_coverage,
        result.materia_accuracy,
        result.natura_coverage,
        result.natura_accuracy,
        result.top2_accuracy,
        result.calibration_error,
        result.ambito_coverage,
        result.ambito_unknown_rate,
        result.l2_abstain_rate,
        result.l2_precision_audit,
        result.l2_precision_auto,
        result.all_gates_passed,
        result.failed_gates,
    )


async def main(run_id: Optional[int] = None):
    """Main validation routine."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Get run_id
        if run_id is None or run_id == "latest":
            run_id = await conn.fetchval("""
                SELECT id FROM kb.graph_runs
                WHERE run_type = 'category_v2' AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """)

        if not run_id:
            print("ERROR: No active category_v2 run found.")
            return

        print(f"Validating run_id: {run_id}")

        # Fetch test predictions
        predictions = await fetch_test_predictions(conn, run_id)
        print(f"Test samples: {len(predictions)}")

        if not predictions:
            print("ERROR: No predictions found for test set.")
            print("Make sure golden set is labeled and build_category_graph_v2.py was run.")
            return

        # Run validation
        result = validate_predictions(predictions)
        result.run_id = run_id

        # Fetch full corpus stats
        corpus_stats = await fetch_full_corpus_stats(conn, run_id)

        # Print results
        print("\n" + "=" * 60)
        print(f"CATEGORY V2 VALIDATION (on {len(predictions)} held-out)")
        print("=" * 60)

        print("\nL1 GATES:")
        print(f"  Materia L1 Coverage:    {result.materia_coverage:.1%} {'OK' if result.materia_coverage >= 1.0 else 'FAIL'} (target = 100%)")
        print(f"  Materia L1 Accuracy:    {result.materia_accuracy:.2f} {'OK' if result.materia_accuracy >= 0.95 else 'FAIL'} (target >= 0.95)")
        print(f"  Natura L1 Coverage:     {result.natura_coverage:.1%} {'OK' if result.natura_coverage >= 1.0 else 'FAIL'} (target = 100%)")
        print(f"  Natura L1 Accuracy:     {result.natura_accuracy:.2f} {'OK' if result.natura_accuracy >= 0.90 else 'FAIL'} (target >= 0.90)")
        print(f"  Top-2 Accuracy:         {result.top2_accuracy:.2f} {'OK' if result.top2_accuracy >= 0.99 else 'FAIL'} (target >= 0.99)")
        print(f"  Calibration Error:      {result.calibration_error:.4f} {'OK' if result.calibration_error < 0.05 else 'FAIL'} (target < 0.05)")

        print("\nAMBITO GATE (on PROCESSUALE):")
        print(f"  Ambito Coverage:        {result.ambito_coverage:.1%} {'OK' if result.ambito_coverage >= 0.95 else 'FAIL'} (target >= 0.95)")
        print(f"  Ambito Unknown Rate:    {result.ambito_unknown_rate:.1%} {'OK' if result.ambito_unknown_rate <= 0.05 else 'FAIL'} (target <= 0.05)")

        print("\nL2 TOPIC GATES:")
        print(f"  L2 Abstain Rate:        {result.l2_abstain_rate:.1%} {'OK' if result.l2_abstain_rate <= 0.40 else 'FAIL'} (target <= 0.40)")
        if result.l2_precision_audit:
            print(f"  L2 Precision (audit):   {result.l2_precision_audit:.2f} {'OK' if result.l2_precision_audit >= 0.90 else 'FAIL'} (target >= 0.90)")
        if result.l2_precision_auto:
            print(f"  L2 Precision (auto):    {result.l2_precision_auto:.2f}")

        print("\n" + "-" * 60)
        if result.all_gates_passed:
            print("ALL GATES PASSED OK")
        else:
            print(f"FAILED GATES: {', '.join(result.failed_gates)}")

        # Print corpus stats
        print("\n" + "=" * 60)
        print("FULL CORPUS STATS")
        print("=" * 60)
        print(f"  Total classified: {corpus_stats['total']}")
        print(f"  Avg materia conf: {corpus_stats['avg_materia_conf']:.3f}")
        print(f"  Avg composite conf: {corpus_stats['avg_composite_conf']:.3f}")

        print("\n  Materia distribution:")
        for mat, cnt in sorted(corpus_stats["materia_dist"].items(), key=lambda x: -x[1]):
            pct = 100 * cnt / corpus_stats["total"]
            print(f"    {mat}: {cnt} ({pct:.1f}%)")

        print("\n  Natura distribution:")
        for nat, cnt in sorted(corpus_stats["natura_dist"].items(), key=lambda x: -x[1]):
            pct = 100 * cnt / corpus_stats["total"]
            print(f"    {nat}: {cnt} ({pct:.1f}%)")

        if corpus_stats["ambito_dist"]:
            print("\n  Ambito distribution (processuale only):")
            for amb, cnt in sorted(corpus_stats["ambito_dist"].items(), key=lambda x: -x[1]):
                pct = 100 * cnt / corpus_stats["processuale"]
                print(f"    {amb}: {cnt} ({pct:.1f}%)")

        # Store eval run
        await store_eval_run(conn, run_id, result)
        print(f"\nEval results stored in kb.category_v2_eval_runs")

        # Print error analysis for failed materia
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS (materia mismatches)")
        print("=" * 60)
        errors = [p for p in predictions if p["pred_materia"] != p["true_materia"]]
        for e in errors[:10]:
            print(f"  {str(e['massima_id'])[:8]}: pred={e['pred_materia']}, true={e['true_materia']}, bucket={e['difficulty_bucket']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Category Graph v2")
    parser.add_argument("--run-id", type=str, help="Run ID or 'latest'")
    args = parser.parse_args()

    run_id = None
    if args.run_id and args.run_id != "latest":
        run_id = int(args.run_id)

    asyncio.run(main(run_id))
