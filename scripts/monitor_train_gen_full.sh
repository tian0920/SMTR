#!/bin/bash
# Monitor full train generation progress
OUTPUT_DIR="artifacts/marble/outputs/effect_check/stageA_paired_train"
EXPECTED_EDGES=420
EXPECTED_SEEDS=3
TOTAL_EXPECTED=$((EXPECTED_EDGES * EXPECTED_SEEDS * 7 / 6))  # ~7 runs per edge (1 ctrl + 6 shares)
# Actually: 70 tasks × 3 seeds × (1 ctrl + 6 shares) = 70 × 3 × 7 = 1470 runs
TOTAL_EXPECTED=1470

LOG="artifacts/effect_check/train_gen_full_progress.log"
echo "=== Full Train Generation Monitor ===" > "$LOG"
echo "Started: $(date)" >> "$LOG"
echo "Expected: $TOTAL_EXPECTED runs (70 tasks × 3 seeds × 7 runs)" >> "$LOG"
echo "" >> "$LOG"

while true; do
    total_controls=0
    total_shares=0
    tasks_started=0

    for task_dir in "$OUTPUT_DIR"/control_groups/*/; do
        [ -d "$task_dir" ] || continue
        tasks_started=$((tasks_started + 1))
        for agent_dir in "$task_dir"agent*/; do
            [ -d "$agent_dir" ] || continue
            for seed_dir in "$agent_dir"*/; do
                [ -d "$seed_dir" ] || continue
                ctrl=$(test -f "$seed_dir/control/control/stdout.log" && echo "1" || echo "0")
                shr=$(find "$seed_dir/shares/" -name "stdout.log" 2>/dev/null | wc -l)
                total_controls=$((total_controls + ctrl))
                total_shares=$((total_shares + shr))
            done
        done
    done

    total_runs=$((total_controls + total_shares))
    pct=$((total_runs * 100 / TOTAL_EXPECTED))
    now=$(date '+%m-%d %H:%M:%S')
    
    # Estimate remaining time based on ~2.5 min/run
    remaining=$((TOTAL_EXPECTED - total_runs))
    remaining_hours=$((remaining * 25 / 10 / 60))
    remaining_mins=$(( (remaining * 25 / 10) % 60 ))
    
    echo "$now | Tasks: $tasks_started/70 | Runs: $total_runs/$TOTAL_EXPECTED ($pct%) | ETA: ${remaining_hours}h${remaining_mins}m" >> "$LOG"
    
    # Check if process is done
    if ! ps aux | grep -q "[g]enerate-database-paired-records"; then
        echo "$(date '+%m-%d %H:%M:%S') | GENERATION COMPLETED" >> "$LOG"
        if [ -f "$OUTPUT_DIR/paired_records.jsonl" ]; then
            n=$(wc -l < "$OUTPUT_DIR/paired_records.jsonl")
            echo "$(date '+%m-%d %H:%M:%S') | Paired records: $n" >> "$LOG"
        fi
        break
    fi
    
    sleep 1800  # Check every 30 minutes
done
