#!/bin/bash
# Monitor train paired record generation progress
OUTPUT_DIR="artifacts/marble/outputs/effect_check/stageA_paired_train"
EXPECTED_TASKS=10
EXPECTED_SEEDS=3
EXPECTED_SHARES=6

echo "=== Train Paired Record Generation Monitor ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

total_controls=0
total_shares=0
total_runs=0
tasks_started=0

for task_dir in "$OUTPUT_DIR"/control_groups/*/; do
    [ -d "$task_dir" ] || continue
    task=$(basename "$task_dir")
    tasks_started=$((tasks_started + 1))
    for agent_dir in "$task_dir"agent*/; do
        [ -d "$agent_dir" ] || continue
        for seed_dir in "$agent_dir"*/; do
            [ -d "$seed_dir" ] || continue
            seed=$(basename "$seed_dir")
            ctrl=$(test -f "$seed_dir/control/control/stdout.log" && echo "1" || echo "0")
            shr=$(find "$seed_dir/shares/" -name "stdout.log" 2>/dev/null | wc -l)
            total_controls=$((total_controls + ctrl))
            total_shares=$((total_shares + shr))
            total_runs=$((total_runs + ctrl + shr))
        done
    done
done

total_expected=$((EXPECTED_TASKS * EXPECTED_SEEDS * (1 + EXPECTED_SHARES)))
pct=$((total_runs * 100 / total_expected))

echo "Tasks started: $tasks_started / $EXPECTED_TASKS"
echo "Completed controls: $total_controls"
echo "Completed shares: $total_shares"
echo "Total runs: $total_runs / $total_expected ($pct%)"
echo ""

# Check if paired_records.jsonl has been created
if [ -f "$OUTPUT_DIR/paired_records.jsonl" ]; then
    n_records=$(wc -l < "$OUTPUT_DIR/paired_records.jsonl")
    echo "Paired records generated: $n_records"
else
    echo "Paired records: not yet generated"
fi

# Check if process is still running
if ps aux | grep -q "[g]enerate-database-paired-records"; then
    echo "Status: RUNNING"
else
    echo "Status: COMPLETED (or stopped)"
fi
