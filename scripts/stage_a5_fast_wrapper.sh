#!/usr/bin/env bash
# Stage A.5-Fast: wait for train_base, then run pipeline
set -euo pipefail

TRAIN_BASE_PID=2894916
LOG=/home/ecs-user/SMTR/artifacts/effect_check/stageA5_pipeline.log

echo "[$(date)] Waiting for train_base generation (PID $TRAIN_BASE_PID)..." | tee "$LOG"

# Wait for train_base to complete
while kill -0 "$TRAIN_BASE_PID" 2>/dev/null; do
    # Check progress every 15 minutes
    TASKS=$(ls /home/ecs-user/SMTR/artifacts/marble/outputs/effect_check/stageA_paired_train/control_groups/ 2>/dev/null | wc -l)
    echo "[$(date)] train_base still running... tasks_started=$TASKS/10" | tee -a "$LOG"
    sleep 900
done

echo "[$(date)] train_base generation completed." | tee -a "$LOG"

# Check if paired_records.jsonl was created
RECORDS=/home/ecs-user/SMTR/artifacts/marble/outputs/effect_check/stageA_paired_train/paired_records.jsonl
if [ ! -f "$RECORDS" ]; then
    echo "[$(date)] ERROR: train_base paired_records.jsonl not found!" | tee -a "$LOG"
    exit 1
fi

RECORD_COUNT=$(wc -l < "$RECORDS")
echo "[$(date)] train_base: $RECORD_COUNT records" | tee -a "$LOG"

# Run the pipeline
echo "[$(date)] Starting Stage A.5-Fast pipeline..." | tee -a "$LOG"
cd /home/ecs-user/SMTR
python3 scripts/stage_a5_fast_pipeline.py 2>&1 | tee -a "$LOG"
PIPELINE_EXIT=$?

echo "[$(date)] Pipeline finished with exit code $PIPELINE_EXIT" | tee -a "$LOG"
