#!/usr/bin/env bash
# 04_run_pipeline.sh -- Run the full evaluation pipeline end-to-end.
#
# Executes the evaluation steps sequentially:
#   1. Sample stratified queries (per-language)
#   2. Run all retrieval configurations (5 systems x 3 rankings x 4 k values)
#   3. Compute IR metrics + Graph-Nearness + Recall-Ceiling
#
# Usage:
#   # Foreground run (see output directly):
#   bash scripts/eval/04_run_pipeline.sh
#
#   # Background run (returns immediately, logs to file):
#   bash scripts/eval/04_run_pipeline.sh --background
#
#   # Skip sampling (reuse existing eval_queries.jsonl):
#   bash scripts/eval/04_run_pipeline.sh --skip-sample
#
#   # Skip retrieval (reuse existing results, recompute metrics only):
#   bash scripts/eval/04_run_pipeline.sh --skip-sample --skip-retrieval
#
# Output (under EVAL_DIR, default data/eval):
#   eval_queries.jsonl              -- sampled queries
#   results/*.jsonl                 -- retrieval results per config
#   results/*_pool.jsonl            -- candidate pools per system
#   metrics/summary.csv             -- aggregate metric table
#   metrics/recall_ceiling.csv      -- pool quality metric
#   metrics/per_query_*.jsonl       -- per-query metric details
#   pipeline_<timestamp>.log        -- full pipeline log (background mode)
#
# Environment variables read by the underlying scripts:
#   EVAL_DIR     -- output directory (default: data/eval)
#   QDRANT_HOST  -- Qdrant host (default: localhost)
#   TEI_HOST     -- TEI embed host (default: localhost)
#   TEI_RERANK_HOST    -- TEI rerank host (default: localhost:8011)
#   TEI_RERANK_URLS    -- comma-separated rerank URLs for parallel fan-out

set -euo pipefail

# ── Parse args ───────────────────────────────────────────────────────────────
BACKGROUND=0
SKIP_SAMPLE=0
SKIP_RETRIEVAL=0
SKIP_METRICS=0

for arg in "$@"; do
    case "$arg" in
        --background|-bg) BACKGROUND=1 ;;
        --skip-sample) SKIP_SAMPLE=1 ;;
        --skip-retrieval) SKIP_RETRIEVAL=1 ;;
        --skip-metrics) SKIP_METRICS=1 ;;
        -h|--help)
            sed -n '/^#/p' "$0" | head -40
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/data/eval}"
PY="${PY:-python3}"

# ── Background mode: re-exec self with nohup ─────────────────────────────────
if [[ $BACKGROUND -eq 1 ]]; then
    TS="$(date -u +'%Y%m%d_%H%M%S')"
    LOG="$EVAL_DIR/pipeline_${TS}.log"
    INNER_ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--background" && "$arg" != "-bg" ]] && INNER_ARGS+=("$arg")
    done
    echo "Launching pipeline in background..."
    echo "Log: $LOG"
    echo "Monitor: tail -f $LOG"
    nohup bash "$0" "${INNER_ARGS[@]}" >"$LOG" 2>&1 &
    PID=$!
    echo "PID: $PID"
    exit 0
fi

echo "============================================================"
echo "Swiss Legal KG-RAG Evaluation Pipeline"
echo "$(date -u +'%F %T UTC')"
echo "Repo root:    $REPO_ROOT"
echo "EVAL_DIR:     $EVAL_DIR"
echo "Python:       $PY"
echo "Skip sample:  $SKIP_SAMPLE"
echo "Skip retrv:   $SKIP_RETRIEVAL"
echo "Skip metrics: $SKIP_METRICS"
echo "============================================================"

cd "$REPO_ROOT"
export EVAL_DIR

# ── Step 1: Sample evaluation queries ────────────────────────────────────────
if [[ $SKIP_SAMPLE -eq 0 ]]; then
    echo ""
    echo "=== Step 1: Sample evaluation queries ==="
    date -u +'%F %T UTC'
    "$PY" "$SCRIPTS_DIR/01_sample_queries.py"
else
    echo ""
    echo "=== Step 1 SKIPPED (--skip-sample) ==="
fi

# ── Step 2: Run retrieval ────────────────────────────────────────────────────
if [[ $SKIP_RETRIEVAL -eq 0 ]]; then
    echo ""
    echo "=== Step 2: Run retrieval ==="
    date -u +'%F %T UTC'
    "$PY" "$SCRIPTS_DIR/02_run_retrieval.py"
else
    echo ""
    echo "=== Step 2 SKIPPED (--skip-retrieval) ==="
fi

# ── Step 3: Compute metrics ──────────────────────────────────────────────────
if [[ $SKIP_METRICS -eq 0 ]]; then
    echo ""
    echo "=== Step 3: Compute metrics ==="
    date -u +'%F %T UTC'
    "$PY" "$SCRIPTS_DIR/03_compute_metrics.py"
else
    echo ""
    echo "=== Step 3 SKIPPED (--skip-metrics) ==="
fi

echo ""
echo "============================================================"
echo "Done. $(date -u +'%F %T UTC')"
echo "Summary:         $EVAL_DIR/metrics/summary.csv"
echo "Recall-Ceiling:  $EVAL_DIR/metrics/recall_ceiling.csv"
echo "============================================================"
