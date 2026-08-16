#!/bin/bash
# Background monitor for train generation - writes progress to log file
LOG="/home/ecs-user/SMTR/artifacts/effect_check/train_gen_progress.log"
echo "=== Train Generation Monitor Started at $(date) ===" > "$LOG"

while true; do
    OUTPUT_DIR="artifacts/marble/outputs/effect_check/stageA_paired_train"
    total_runs=0
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
                total_runs=$((total_runs + ctrl + shr))
            done
        done
    done

    echo "$(date '+%H:%M:%S') | Tasks: $tasks_started/10 | Runs: $total_runs/210 ($(( total_runs * 100 / 210 ))%)" >> "$LOG"

    # Check if process is done
    if ! ps aux | grep -q "[g]enerate-database-paired-records"; then
        echo "$(date '+%H:%M:%S') | GENERATION COMPLETED" >> "$LOG"
        if [ -f "$OUTPUT_DIR/paired_records.jsonl" ]; then
            n=$(wc -l < "$OUTPUT_DIR/paired_records.jsonl")
            echo "$(date '+%H:%M:%S') | Paired records: $n" >> "$LOG"
        fi
        break
    fi

    sleep 1800  # Check every 30 minutes
done

echo "=== Monitor finished at $(date) ===" >> "$LOG"
