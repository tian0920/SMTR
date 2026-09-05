#!/usr/bin/env bash
# Phase 45A_v3 post-completion pipeline
# Waits for DONE, then runs: audit → timing → mechanism check →
# prequential probe audit → Gate A summary
set -e

RUN_DIR="results/rima_transfer/pilot/45a_adaptive_v3/bargaining__stream0__exec0__methodrima_transfer_adaptive"
OUT_DIR="results/rima_transfer/pilot/45a_adaptive_v3"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== Phase 45A_v3 Post-Completion Pipeline ==="
echo "Started: $TIMESTAMP"
echo ""

# --- Wait for completion ---
echo "[WAIT] Waiting for DONE marker..."
while true; do
    if [ -f "$RUN_DIR/DONE" ]; then
        echo "[WAIT] DONE detected at $(date '+%H:%M:%S')"
        break
    fi
    if [ -f "$RUN_DIR/FAILED" ]; then
        echo "[WAIT] FAILED detected at $(date '+%H:%M:%S')"
        echo "=== ABORT: Stream FAILED, no post-processing ==="
        cat "$RUN_DIR/FAILED"
        exit 1
    fi
    COUNT=$(wc -l < "$RUN_DIR/tasks.jsonl" 2>/dev/null || echo 0)
    echo "[WAIT] $(date '+%H:%M:%S') tasks=$COUNT/30"
    sleep 120
done

# --- Verify completeness ---
echo ""
echo "=== Step 1: Verify stream completeness ==="
TASK_COUNT=$(wc -l < "$RUN_DIR/tasks.jsonl")
echo "tasks.jsonl records: $TASK_COUNT"
if [ "$TASK_COUNT" -ne 30 ]; then
    echo "WARNING: Expected 30 tasks, got $TASK_COUNT"
fi
if [ -f "$RUN_DIR/FAILED" ]; then
    echo "ERROR: FAILED marker exists despite DONE"
    exit 1
fi
echo "✓ Stream complete: $TASK_COUNT tasks, DONE=yes, FAILED=no"

# --- Step 2: Causal runtime audit ---
echo ""
echo "=== Step 2: Causal Runtime Audit ==="
python experiments/rima/audit_continual_run.py --input "$RUN_DIR" 2>&1 | tee "$OUT_DIR/causal_audit_log.txt"
AUDIT_EXIT=${PIPESTATUS[0]}
echo "Audit exit code: $AUDIT_EXIT"
if [ $AUDIT_EXIT -ne 0 ]; then
    echo "⚠ CAUSAL AUDIT FAILED — DO NOT PROCEED TO MECHANISM ANALYSIS"
    echo "=== PIPELINE STOPPED: Causal audit failure ==="
    exit 1
fi
echo "✓ Causal audit passed"

# --- Step 3: Timing breakdown ---
echo ""
echo "=== Step 3: Adaptive Timing Breakdown ==="
python scripts/rima/generate_timing_breakdown.py "$RUN_DIR"
echo "✓ Timing breakdown generated"

# --- Step 4: Mechanism check (metric D is pre-probe based, P1-2) ---
echo ""
echo "=== Step 4: Mechanism Check (5 metrics) ==="
python scripts/rima/mechanism_check.py "$RUN_DIR" 2>&1 | tee "$OUT_DIR/mechanism_check_log.txt"
MECH_EXIT=${PIPESTATUS[0]}
echo "Mechanism check exit code: $MECH_EXIT"
if [ -f "$RUN_DIR/mechanism_check.json" ]; then
    cp "$RUN_DIR/mechanism_check.json" "$OUT_DIR/mechanism_check.json"
fi

# --- Step 5: Prequential probe audit (P1-2/3/4) ---
echo ""
echo "=== Step 5: Prequential Probe Audit (P1-2/3/4) ==="
python scripts/rima/prequential_probe_audit.py "$RUN_DIR" 2>&1 | tee "$OUT_DIR/prequential_audit_log.txt"
if [ -f "$RUN_DIR/prequential_probe_audit.json" ]; then
    cp "$RUN_DIR/prequential_probe_audit.json" "$OUT_DIR/prequential_probe_audit.json"
fi

# --- Step 6: Gate A summary ---
echo ""
echo "=== Step 6: Gate A Summary ==="
echo "Check $RUN_DIR/mechanism_check.json for final GO/NO-GO/YELLOW verdict"
if [ -f "$RUN_DIR/mechanism_check.json" ]; then
    python -c "
import json
with open('$RUN_DIR/mechanism_check.json') as f:
    result = json.load(f)
gate = result.get('gate_a', {})
verdict = gate.get('verdict', 'UNKNOWN')
reason = gate.get('reason', '')
print(f'  GATE A VERDICT: {verdict}')
print(f'  Reason: {reason}')
metrics = result.get('metrics', [])
for m in metrics:
    print(f'  [{m.get(\"metric\")}] {m.get(\"direction\")}')
    for k, v in m.items():
        if k not in ('metric', 'direction'):
            print(f'      {k}: {v}')
"
fi

echo ""
echo "=== Phase 45A_v3 Pipeline Complete ==="
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in: $OUT_DIR/"
