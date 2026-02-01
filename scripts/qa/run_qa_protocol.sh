#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# QA Protocol Orchestrator
# ============================================================
# Usage:
#   ./run_qa_protocol.sh                    # Run all phases
#   ./run_qa_protocol.sh --phase 0          # Run only phase 0
#   ./run_qa_protocol.sh --phase 1-3        # Run phases 1 to 3
#   ./run_qa_protocol.sh --phase 7          # Run only phase 7
#   ./run_qa_protocol.sh --phase guided     # Run guided ingestion
#
# Logs: /opt/lexe-platform/lexe-max/logs/qa/<timestamp>/
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/opt/lexe-platform/lexe-max/logs/qa/$(date +%Y%m%d_%H%M%S)"
PHASE="all"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"
echo "============================================================"
echo "QA PROTOCOL - ORCHESTRATOR"
echo "============================================================"
echo "Phase: $PHASE"
echo "Log dir: $LOG_DIR"
echo "Started: $(date)"
echo "============================================================"

run_script() {
    local script="$1"
    local log_file="$LOG_DIR/$(basename "$script" .py).log"
    echo ""
    echo "[$(date +%H:%M:%S)] Running $script..."
    if uv run python "$SCRIPT_DIR/$script" 2>&1 | tee "$log_file"; then
        echo "[$(date +%H:%M:%S)] OK: $script"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $script (exit $?)"
        echo "Check log: $log_file"
        exit 1
    fi
}

should_run() {
    local phase_num="$1"
    if [[ "$PHASE" == "all" ]]; then
        return 0
    fi
    # Range support: "1-3"
    if [[ "$PHASE" == *-* ]]; then
        local start="${PHASE%%-*}"
        local end="${PHASE##*-}"
        if [[ "$phase_num" -ge "$start" && "$phase_num" -le "$end" ]]; then
            return 0
        fi
        return 1
    fi
    # Exact match
    if [[ "$PHASE" == "$phase_num" || "$PHASE" == "guided" && "$phase_num" == "11" ]]; then
        return 0
    fi
    return 1
}

# ── PHASE 0: Setup ────────────────────────────────────────────
if should_run 0; then
    echo ""
    echo "=== PHASE 0: Setup ==="
    run_script "s0_build_manifest.py"
    run_script "s0_extract_reference_units.py"
fi

# ── PHASE 1: Integrity & Metadata ─────────────────────────────
if should_run 1; then
    echo ""
    echo "=== PHASE 1: Integrity & Metadata ==="
    run_script "s1_page_extraction_stats.py"
    run_script "s1_year_resolution.py"
    # health_flags depends on the above
    run_script "s1_health_flags.py"
fi

# ── PHASE 2: Extraction Quality ───────────────────────────────
if should_run 2; then
    echo ""
    echo "=== PHASE 2: Extraction Quality ==="
    run_script "s2_extraction_quality.py"
    run_script "s2_noise_detection.py"
fi

# ── PHASE 3: Gate Policy Audit ────────────────────────────────
if should_run 3; then
    echo ""
    echo "=== PHASE 3: Gate Policy Audit ==="
    run_script "s3_gate_policy_audit.py"
fi

# ── PHASE 4: Chunk Analysis ──────────────────────────────────
if should_run 4; then
    echo ""
    echo "=== PHASE 4: Chunk Analysis ==="
    run_script "s4_chunk_analysis.py"
fi

# ── PHASE 5: Silver Labeling ─────────────────────────────────
if should_run 5; then
    echo ""
    echo "=== PHASE 5: Silver Labeling ==="
    run_script "s5_silver_labeling.py"
fi

# ── PHASE 6: Reference Alignment ─────────────────────────────
if should_run 6; then
    echo ""
    echo "=== PHASE 6: Reference Alignment ==="
    run_script "s6_reference_alignment.py"
fi

# ── PHASE 7: Retrieval Evaluation ─────────────────────────────
if should_run 7; then
    echo ""
    echo "=== PHASE 7: Retrieval Evaluation ==="
    run_script "s7_generate_query_set.py"
    run_script "s7_run_retrieval_eval.py"
fi

# ── PHASE 8: Profile Assignment ───────────────────────────────
if should_run 8; then
    echo ""
    echo "=== PHASE 8: Profile Assignment ==="
    run_script "s8_assign_profiles.py"
fi

# ── PHASE 9: LLM Triggers ────────────────────────────────────
if should_run 9; then
    echo ""
    echo "=== PHASE 9: LLM Triggers ==="
    run_script "s9_llm_ambiguous_year.py"
    run_script "s9_llm_borderline.py"
    run_script "s9_llm_boundary_repair.py"
fi

# ── PHASE 10: Reports ────────────────────────────────────────
if should_run 10; then
    echo ""
    echo "=== PHASE 10: Reports ==="
    run_script "s10_generate_reports.py"
    run_script "s10_recommended_actions.py"
fi

# ── PHASE 11: Guided Ingestion ────────────────────────────────
if should_run 11; then
    echo ""
    echo "=== PHASE 11: Guided Ingestion ==="
    run_script "guided_ingestion.py"
    echo ""
    echo "[!] Re-run phases 4-10 for guided batch comparison"
fi

echo ""
echo "============================================================"
echo "QA PROTOCOL COMPLETE"
echo "============================================================"
echo "Finished: $(date)"
echo "Logs: $LOG_DIR"
echo "============================================================"
