"""
Classification Logger for Category Graph v2.5

Structured JSONL logging for post-hoc analysis of classification decisions.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID


@dataclass
class ClassificationLog:
    """Log strutturato per ogni classificazione."""

    massima_id: UUID
    timestamp: str
    sezione: Optional[str]
    norm_hint_applied: bool
    norm_hint_reason: Optional[str]

    # Top-3 grezzo e calibrato
    top3_labels_raw: List[str] = field(default_factory=list)
    top3_scores_raw: List[float] = field(default_factory=list)
    top3_scores_calibrated: List[float] = field(default_factory=list)

    delta_12: float = 0.0
    routing_decision: str = ""  # "TOP1", "TOP2", "LLM_RESOLVER", "NEEDS_REVIEW"

    # LLM resolver (se chiamato)
    llm_called: bool = False
    llm_candidate_set: Optional[List[str]] = None
    llm_output_label: Optional[str] = None
    llm_output_confidence: Optional[float] = None
    llm_accepted: Optional[bool] = None
    llm_agreement: Optional[bool] = None  # True se i due LLM concordano
    llm_judge_called: Optional[bool] = None

    latency_ms_l1: float = 0.0
    latency_ms_llm: float = 0.0

    # Final result
    final_label: Optional[str] = None
    final_confidence: Optional[float] = None


class ClassificationLogger:
    """
    Logger strutturato per classificazioni.
    Salva in JSONL per analisi post-hoc.
    """

    def __init__(self, run_id: int, output_dir: Path = Path("logs/classification")):
        self.run_id = run_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = (
            self.output_dir / f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._count = 0

    def log(self, classification_log: ClassificationLog) -> None:
        """Append log entry."""
        entry = asdict(classification_log)
        entry["run_id"] = self.run_id
        # Convert UUID to string for JSON serialization
        if isinstance(entry.get("massima_id"), UUID):
            entry["massima_id"] = str(entry["massima_id"])
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._count += 1
        # Flush periodically
        if self._count % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def log_file_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_path

    @property
    def entries_logged(self) -> int:
        """Get the number of entries logged."""
        return self._count


class ClassificationStats:
    """Accumulator for classification statistics."""

    def __init__(self):
        self.total = 0
        self.by_routing = {"TOP1": 0, "TOP2": 0, "LLM_RESOLVER": 0, "NEEDS_REVIEW": 0}
        self.llm_triggered = 0
        self.llm_accepted = 0
        self.llm_agreements = 0
        self.llm_judge_called = 0
        self.latencies_l1: List[float] = []
        self.latencies_llm: List[float] = []

    def record(self, log: ClassificationLog) -> None:
        """Record a classification result."""
        self.total += 1

        if log.routing_decision in self.by_routing:
            self.by_routing[log.routing_decision] += 1

        if log.llm_called:
            self.llm_triggered += 1
            if log.llm_accepted:
                self.llm_accepted += 1
            if log.llm_agreement:
                self.llm_agreements += 1
            if log.llm_judge_called:
                self.llm_judge_called += 1
            if log.latency_ms_llm > 0:
                self.latencies_llm.append(log.latency_ms_llm)

        if log.latency_ms_l1 > 0:
            self.latencies_l1.append(log.latency_ms_l1)

    def summary(self) -> dict:
        """Get summary statistics."""
        import numpy as np

        return {
            "total": self.total,
            "routing": self.by_routing,
            "llm_trigger_rate": self.llm_triggered / self.total if self.total > 0 else 0,
            "llm_accept_rate": (
                self.llm_accepted / self.llm_triggered if self.llm_triggered > 0 else 0
            ),
            "llm_agreement_rate": (
                self.llm_agreements / self.llm_triggered if self.llm_triggered > 0 else 0
            ),
            "llm_judge_rate": (
                self.llm_judge_called / self.llm_triggered if self.llm_triggered > 0 else 0
            ),
            "latency_l1_p50": float(np.percentile(self.latencies_l1, 50)) if self.latencies_l1 else 0,
            "latency_l1_p95": float(np.percentile(self.latencies_l1, 95)) if self.latencies_l1 else 0,
            "latency_llm_p50": float(np.percentile(self.latencies_llm, 50)) if self.latencies_llm else 0,
            "latency_llm_p95": float(np.percentile(self.latencies_llm, 95)) if self.latencies_llm else 0,
        }

    def print_summary(self) -> None:
        """Print a formatted summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("CLASSIFICATION STATISTICS")
        print("=" * 60)
        print(f"  Total: {s['total']}")
        print("\n  Routing Distribution:")
        for k, v in s["routing"].items():
            pct = 100 * v / s["total"] if s["total"] > 0 else 0
            print(f"    {k}: {v} ({pct:.1f}%)")
        print(f"\n  LLM Trigger Rate: {100*s['llm_trigger_rate']:.1f}%")
        print(f"  LLM Accept Rate: {100*s['llm_accept_rate']:.1f}%")
        print(f"  LLM Agreement Rate: {100*s['llm_agreement_rate']:.1f}%")
        print(f"  LLM Judge Rate: {100*s['llm_judge_rate']:.1f}%")
        print(f"\n  Latency L1 p50/p95: {s['latency_l1_p50']:.1f}ms / {s['latency_l1_p95']:.1f}ms")
        print(f"  Latency LLM p50/p95: {s['latency_llm_p50']:.1f}ms / {s['latency_llm_p95']:.1f}ms")
        print("=" * 60)
