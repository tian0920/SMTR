#!/usr/bin/env bash
# Phase 23.4 independent diagnosis watcher for 45A_v3.
#
# Waits for the stream DONE marker, then runs the read-only diagnostic
# script (analysis/rima_transfer/diagnose_45a_v3.py) into $RUN_ROOT/diagnosis.
#
# This watcher is deliberately SEPARATE from run_45a_v3_post_completion.sh
# (which is already running as PID 3315584). It does NOT modify the run,
# the controller/learner/critic, beta/delta/gamma, or the probe policy
# (Phase 23.0). It only reads jsonl artefacts after DONE.
set -u

RUN_ROOT="results/rima_transfer/pilot/45a_adaptive_v3"
RUN_DIR="$RUN_ROOT/bargaining__stream0__exec0__methodrima_transfer_adaptive"
OUT_DIR="$RUN_ROOT/diagnosis"
LOG="$RUN_ROOT/45a_v3_diagnose.log"

echo "=== 45A_v3 Diagnosis Watcher started $(date '+%F %T') ===" | tee "$LOG"

while true; do
    if [ -f "$RUN_DIR/DONE" ]; then
        echo "[WAIT] DONE detected $(date '+%T')" | tee -a "$LOG"
        break
    fi
    if [ -f "$RUN_DIR/FAILED" ]; then
        echo "[WAIT] FAILED detected $(date '+%T') — aborting diagnosis" | tee -a "$LOG"
        exit 1
    fi
    COUNT=$(wc -l < "$RUN_DIR/tasks.jsonl" 2>/dev/null || echo 0)
    echo "[WAIT] $(date '+%T') tasks=$COUNT/30" >> "$LOG"
    sleep 120
done

TASK_COUNT=$(wc -l < "$RUN_DIR/tasks.jsonl" 2>/dev/null || echo 0)
echo "[GATE 23.1] tasks=$TASK_COUNT (expect 30)" | tee -a "$LOG"

echo "[RUN] diagnose_45a_v3.py" | tee -a "$LOG"
python analysis/rima_transfer/diagnose_45a_v3.py \
    --run-dir "$RUN_DIR" \
    --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

echo "=== Diagnosis watcher complete $(date '+%F %T') ===" | tee -a "$LOG"
echo "Outputs in: $OUT_DIR/" | tee -a "$LOG"
