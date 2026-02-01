#!/usr/bin/env python3
"""
Train Isotonic Calibrator for Category Graph v2.5

Trains on golden set (train split only!) using predictions from baseline run.

Usage:
    uv run python scripts/qa/train_calibrator.py
    uv run python scripts/qa/train_calibrator.py --run-id 10
"""

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.calibration import IsotonicCalibrator, compute_ece


async def main(run_id: int | None = None):
    """Train calibrator on golden set predictions."""
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Find the run to use
        if run_id is None:
            # Use latest completed run
            row = await conn.fetchrow("""
                SELECT id FROM kb.graph_runs
                WHERE run_type = 'category_v2' AND status = 'completed'
                ORDER BY id DESC LIMIT 1
            """)
            if not row:
                print("ERROR: No completed category_v2 runs found")
                sys.exit(1)
            run_id = row["id"]

        print(f"Training calibrator using predictions from run {run_id}")

        # Fetch predictions vs golden (train split only!)
        rows = await conn.fetch("""
            SELECT
                p.materia_confidence as score_raw,
                CASE WHEN p.materia_l1 = g.materia_l1 THEN 1 ELSE 0 END as correct
            FROM kb.category_predictions_v2 p
            JOIN kb.golden_category_adjudicated_v2 g ON g.massima_id = p.massima_id
            WHERE p.run_id = $1
              AND g.split = 'train'
        """, run_id)

        if len(rows) < 50:
            print(f"ERROR: Only {len(rows)} samples in train split. Need at least 50.")
            sys.exit(1)

        scores = np.array([r["score_raw"] for r in rows])
        labels = np.array([r["correct"] for r in rows])

        print(f"Training on {len(rows)} samples from train split")
        print(f"  Accuracy: {labels.mean():.1%}")
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

        # Compute ECE before calibration
        ece_before = compute_ece(scores, labels, n_bins=10)
        print(f"\n  ECE before calibration: {ece_before:.4f}")

        # Train calibrator
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, labels)

        # Compute ECE after calibration
        calibrated = np.array([calibrator.calibrate(s) for s in scores])
        ece_after = compute_ece(calibrated, labels, n_bins=10)
        print(f"  ECE after calibration:  {ece_after:.4f}")
        print(f"  ECE improvement: {(ece_before - ece_after) / ece_before:.1%}")

        # Show calibration curve samples
        print("\n  Calibration samples (raw -> calibrated):")
        for raw in [0.3, 0.5, 0.7, 0.8, 0.9]:
            cal = calibrator.calibrate(raw)
            print(f"    {raw:.2f} -> {cal:.2f}")

        # Analyze threshold impact
        print("\n  Threshold analysis:")
        for th in [0.65, 0.70, 0.75, 0.80]:
            raw_above = (scores >= th).sum()
            cal_above = (calibrated >= th).sum()
            print(f"    TH={th:.2f}: raw {raw_above} ({100*raw_above/len(scores):.1f}%) -> cal {cal_above} ({100*cal_above/len(scores):.1f}%)")

        # Save calibrator
        out_path = Path("data/calibrator_v1.json")
        out_path.parent.mkdir(exist_ok=True)
        calibrator.save(out_path)
        print(f"\nSaved calibrator to {out_path}")
        print(f"  Version: {calibrator.version}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train isotonic calibrator")
    parser.add_argument("--run-id", type=int, help="Run ID to use for training (default: latest)")
    args = parser.parse_args()

    asyncio.run(main(args.run_id))
