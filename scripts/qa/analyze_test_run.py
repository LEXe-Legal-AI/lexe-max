#!/usr/bin/env python3
"""
Analyze Test Run for Category Graph v2.5

Analyzes JSONL logs from a test run to determine GO/NO-GO for full corpus.

Usage:
    uv run python scripts/qa/analyze_test_run.py --run-id 11
    uv run python scripts/qa/analyze_test_run.py --run-id latest
    uv run python scripts/qa/analyze_test_run.py --log-file logs/classification/run_11_*.jsonl
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings

# GO/NO-GO thresholds
THRESHOLDS = {
    "llm_trigger_max": 0.25,  # Max 25% of corpus should trigger LLM
    "llm_accept_min": 0.50,   # Min 50% of LLM calls should be accepted
    "parse_error_max": 0.05,  # Max 5% parse errors
    "latency_p95_max_ms": 2000,  # Max 2s p95 latency
}


def analyze_jsonl_log(log_path: Path) -> Dict:
    """Analyze a JSONL classification log file."""
    stats = {
        "total": 0,
        "routing": defaultdict(int),
        "llm_triggered": 0,
        "llm_accepted": 0,
        "llm_agreement": 0,
        "llm_judge_called": 0,
        "parse_errors": 0,
        "latencies_l1": [],
        "latencies_llm": [],
        "delta_12_values": [],
        "by_strata": defaultdict(lambda: {"total": 0, "llm_triggered": 0}),
    }

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                continue

            stats["total"] += 1
            routing = entry.get("routing_decision", "UNKNOWN")
            stats["routing"][routing] += 1

            # LLM stats
            if entry.get("llm_called"):
                stats["llm_triggered"] += 1
                if entry.get("llm_accepted"):
                    stats["llm_accepted"] += 1
                if entry.get("llm_agreement"):
                    stats["llm_agreement"] += 1
                if entry.get("llm_judge_called"):
                    stats["llm_judge_called"] += 1

            # Latency
            if entry.get("latency_ms_l1"):
                stats["latencies_l1"].append(entry["latency_ms_l1"])
            if entry.get("latency_ms_llm"):
                stats["latencies_llm"].append(entry["latency_ms_llm"])

            # Delta
            if entry.get("delta_12"):
                stats["delta_12_values"].append(entry["delta_12"])

            # By sezione (for strata analysis)
            sezione = entry.get("sezione", "UNKNOWN")
            strata = _sezione_to_strata(sezione)
            stats["by_strata"][strata]["total"] += 1
            if entry.get("llm_called"):
                stats["by_strata"][strata]["llm_triggered"] += 1

    return stats


def _sezione_to_strata(sezione: Optional[str]) -> str:
    """Map sezione to strata name."""
    if not sezione:
        return "other"
    sez = sezione.lower()
    if sez == "l":
        return "lavoro"
    elif sez == "u":
        return "unite"
    elif sez in ("1", "2", "3", "4", "6"):
        return "civile"
    elif sez == "5":
        return "tributaria"
    else:
        return "other"


def compute_metrics(stats: Dict) -> Dict:
    """Compute derived metrics from raw stats."""
    total = stats["total"]
    llm_triggered = stats["llm_triggered"]

    metrics = {
        "total": total,
        "llm_trigger_rate": llm_triggered / total if total > 0 else 0,
        "llm_accept_rate": stats["llm_accepted"] / llm_triggered if llm_triggered > 0 else 0,
        "llm_agreement_rate": stats["llm_agreement"] / llm_triggered if llm_triggered > 0 else 0,
        "llm_judge_rate": stats["llm_judge_called"] / llm_triggered if llm_triggered > 0 else 0,
        "parse_error_rate": stats["parse_errors"] / (total + stats["parse_errors"]) if total > 0 else 0,
    }

    # Latency percentiles
    if stats["latencies_l1"]:
        metrics["latency_l1_p50"] = np.percentile(stats["latencies_l1"], 50)
        metrics["latency_l1_p95"] = np.percentile(stats["latencies_l1"], 95)
    else:
        metrics["latency_l1_p50"] = 0
        metrics["latency_l1_p95"] = 0

    if stats["latencies_llm"]:
        metrics["latency_llm_p50"] = np.percentile(stats["latencies_llm"], 50)
        metrics["latency_llm_p95"] = np.percentile(stats["latencies_llm"], 95)
    else:
        metrics["latency_llm_p50"] = 0
        metrics["latency_llm_p95"] = 0

    # Delta histogram
    if stats["delta_12_values"]:
        deltas = np.array(stats["delta_12_values"])
        metrics["delta_12_mean"] = np.mean(deltas)
        metrics["delta_12_std"] = np.std(deltas)
        metrics["delta_12_below_0.12"] = np.sum(deltas < 0.12) / len(deltas)

    # Routing distribution
    metrics["routing"] = dict(stats["routing"])

    # By strata
    metrics["by_strata"] = {}
    for strata, strata_stats in stats["by_strata"].items():
        strata_total = strata_stats["total"]
        metrics["by_strata"][strata] = {
            "total": strata_total,
            "llm_trigger_rate": strata_stats["llm_triggered"] / strata_total if strata_total > 0 else 0,
        }

    return metrics


def evaluate_go_nogo(metrics: Dict) -> Dict:
    """Evaluate GO/NO-GO criteria."""
    checks = {}

    # LLM trigger rate
    checks["llm_trigger"] = {
        "value": metrics["llm_trigger_rate"],
        "threshold": THRESHOLDS["llm_trigger_max"],
        "pass": metrics["llm_trigger_rate"] <= THRESHOLDS["llm_trigger_max"],
        "message": f"LLM trigger rate: {100*metrics['llm_trigger_rate']:.1f}% (max {100*THRESHOLDS['llm_trigger_max']:.0f}%)",
    }

    # LLM accept rate
    checks["llm_accept"] = {
        "value": metrics["llm_accept_rate"],
        "threshold": THRESHOLDS["llm_accept_min"],
        "pass": metrics["llm_accept_rate"] >= THRESHOLDS["llm_accept_min"],
        "message": f"LLM accept rate: {100*metrics['llm_accept_rate']:.1f}% (min {100*THRESHOLDS['llm_accept_min']:.0f}%)",
    }

    # Parse errors
    checks["parse_errors"] = {
        "value": metrics["parse_error_rate"],
        "threshold": THRESHOLDS["parse_error_max"],
        "pass": metrics["parse_error_rate"] <= THRESHOLDS["parse_error_max"],
        "message": f"Parse error rate: {100*metrics['parse_error_rate']:.1f}% (max {100*THRESHOLDS['parse_error_max']:.0f}%)",
    }

    # Latency p95
    latency_p95 = metrics.get("latency_llm_p95", 0)
    checks["latency_p95"] = {
        "value": latency_p95,
        "threshold": THRESHOLDS["latency_p95_max_ms"],
        "pass": latency_p95 <= THRESHOLDS["latency_p95_max_ms"],
        "message": f"LLM latency p95: {latency_p95:.0f}ms (max {THRESHOLDS['latency_p95_max_ms']}ms)",
    }

    # Overall GO/NO-GO
    all_pass = all(c["pass"] for c in checks.values())
    checks["overall"] = {
        "pass": all_pass,
        "message": "GO for full corpus" if all_pass else "NO-GO - fix issues first",
    }

    return checks


def print_report(metrics: Dict, checks: Dict):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("TEST RUN ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n  Total classified: {metrics['total']}")

    print("\n  ROUTING DISTRIBUTION:")
    for routing, count in sorted(metrics["routing"].items()):
        pct = 100 * count / metrics["total"] if metrics["total"] > 0 else 0
        print(f"    {routing}: {count} ({pct:.1f}%)")

    print("\n  LLM METRICS:")
    print(f"    Trigger rate: {100*metrics['llm_trigger_rate']:.1f}%")
    print(f"    Accept rate: {100*metrics['llm_accept_rate']:.1f}%")
    print(f"    Agreement rate: {100*metrics['llm_agreement_rate']:.1f}%")
    print(f"    Judge rate: {100*metrics['llm_judge_rate']:.1f}%")

    print("\n  LATENCY:")
    print(f"    L1 p50/p95: {metrics['latency_l1_p50']:.1f}ms / {metrics['latency_l1_p95']:.1f}ms")
    print(f"    LLM p50/p95: {metrics['latency_llm_p50']:.1f}ms / {metrics['latency_llm_p95']:.1f}ms")

    if "delta_12_mean" in metrics:
        print("\n  DELTA DISTRIBUTION:")
        print(f"    Mean: {metrics['delta_12_mean']:.3f} (std: {metrics['delta_12_std']:.3f})")
        print(f"    Below 0.12: {100*metrics['delta_12_below_0.12']:.1f}%")

    print("\n  BY STRATA:")
    for strata, strata_metrics in sorted(metrics["by_strata"].items()):
        print(f"    {strata}: {strata_metrics['total']} samples, LLM trigger: {100*strata_metrics['llm_trigger_rate']:.1f}%")

    print("\n" + "-" * 70)
    print("  GO/NO-GO EVALUATION:")
    print("-" * 70)
    for name, check in checks.items():
        if name == "overall":
            continue
        status = "PASS" if check["pass"] else "FAIL"
        print(f"    [{status}] {check['message']}")

    print("\n" + "=" * 70)
    status = "GO" if checks["overall"]["pass"] else "NO-GO"
    print(f"  VERDICT: [{status}] {checks['overall']['message']}")
    print("=" * 70)


async def find_latest_log() -> Optional[Path]:
    """Find the most recent classification log file."""
    log_dir = Path("logs/classification")
    if not log_dir.exists():
        return None

    logs = sorted(log_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


async def find_log_for_run(run_id: int) -> Optional[Path]:
    """Find log file for a specific run ID."""
    log_dir = Path("logs/classification")
    if not log_dir.exists():
        return None

    logs = list(log_dir.glob(f"run_{run_id}_*.jsonl"))
    return logs[0] if logs else None


async def main(run_id: Optional[str], log_file: Optional[str]):
    """Main analysis routine."""

    log_path: Optional[Path] = None

    if log_file:
        log_path = Path(log_file)
    elif run_id == "latest":
        log_path = await find_latest_log()
    elif run_id:
        log_path = await find_log_for_run(int(run_id))

    if not log_path or not log_path.exists():
        print(f"ERROR: Log file not found")
        if not log_file:
            print("Specify --log-file or check logs/classification/ directory")
        sys.exit(1)

    print(f"Analyzing: {log_path}")

    stats = analyze_jsonl_log(log_path)
    metrics = compute_metrics(stats)
    checks = evaluate_go_nogo(metrics)

    print_report(metrics, checks)

    # Exit with error code if NO-GO
    if not checks["overall"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze test run for GO/NO-GO")
    parser.add_argument("--run-id", type=str, help="Run ID or 'latest'")
    parser.add_argument("--log-file", type=str, help="Path to JSONL log file")
    args = parser.parse_args()

    if not args.run_id and not args.log_file:
        args.run_id = "latest"

    asyncio.run(main(args.run_id, args.log_file))
