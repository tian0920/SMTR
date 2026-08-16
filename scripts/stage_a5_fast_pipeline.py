"""Stage A.5-Fast pipeline: train_base → screening → diagnostic subset selection.

This script orchestrates the sequential pipeline after train_base generation
completes. It:
1. Analyzes train_base distribution
2. Runs control-only screening on remaining 60 tasks
3. Selects H (positive-transfer) and S (harmful-transfer) task subsets
4. Generates diagnostic paired records for selected tasks

Run AFTER train_base generation (PID 2294951) completes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/ecs-user/SMTR")
MARBLE_ROOT = Path("/home/ecs-user/MARBLE")
EFFECT_DIR = PROJECT_ROOT / "artifacts" / "marble" / "outputs" / "effect_check"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "effect_check"

TRAIN_BASE_TASKS = "2,10,11,12,13,14,16,17,19,100"

SPLIT_MANIFEST = PROJECT_ROOT / "artifacts" / "marble" / "manifests" / "effect_check" / "splits.json"
CANDIDATE_MANIFEST = EFFECT_DIR / "stageA_candidates_train"
MEMORY_POOL = EFFECT_DIR / "stageA_memories"

SCREENING_OUTPUT = ARTIFACT_DIR / "stageA5_screening_results.jsonl"
DIAGNOSTIC_OUTPUT = EFFECT_DIR / "stageA5_diagnostic_paired"


def run_cmd(cmd: list[str], label: str, log_path: Path | None = None) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{label}] Starting: {' '.join(cmd[:5])}...", flush=True)
    print(f"{'='*60}", flush=True)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"[{label}] Exit code: {proc.returncode} (log: {log_path})", flush=True)
    else:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        print(f"[{label}] Exit code: {proc.returncode}", flush=True)
    return proc.returncode


def analyze_train_base() -> dict:
    """Analyze train_base paired records distribution."""
    records_path = EFFECT_DIR / "stageA_paired_train" / "paired_records.jsonl"
    if not records_path.exists():
        print("ERROR: train_base paired_records.jsonl not found!", flush=True)
        return {}

    records = [json.loads(line) for line in records_path.read_text().strip().splitlines() if line.strip()]
    from collections import Counter
    labels = Counter(r.get("label", "unknown") for r in records)
    result = {
        "total": len(records),
        "q00": labels.get("neutral_failure", 0),
        "q01": labels.get("negative_transfer", 0),
        "q10": labels.get("positive_transfer", 0),
        "q11": labels.get("neutral_success", 0),
    }
    result["informative"] = result["q01"] + result["q10"]
    print(f"\ntrain_base distribution:", flush=True)
    for k, v in result.items():
        print(f"  {k}: {v}", flush=True)
    return result


def analyze_screening() -> tuple[list[str], list[str]]:
    """Analyze screening results and return (h_tasks, s_tasks)."""
    if not SCREENING_OUTPUT.exists():
        print("ERROR: screening results not found!", flush=True)
        return [], []

    results = [json.loads(line) for line in SCREENING_OUTPUT.read_text().strip().splitlines() if line.strip()]

    h_tasks = []  # Y0=0 (control failed → positive-transfer opportunity)
    s_tasks = []  # Y0=1 (control succeeded → harmful-transfer opportunity)

    for r in results:
        task_id = str(r["task_id"])
        if r.get("error"):
            continue
        if r.get("success"):
            s_tasks.append(task_id)
        else:
            h_tasks.append(task_id)

    # Sort by int for stable ordering
    h_tasks.sort(key=int)
    s_tasks.sort(key=int)

    print(f"\nScreening results:", flush=True)
    print(f"  H tasks (Y0=0, failure): {len(h_tasks)} → {h_tasks}", flush=True)
    print(f"  S tasks (Y0=1, success): {len(s_tasks)} → {s_tasks[:10]}...", flush=True)

    return h_tasks, s_tasks


def select_diagnostic_subset(h_tasks: list[str], s_tasks: list[str]) -> tuple[list[str], list[str]]:
    """Select 4-6 H + 4-6 S tasks for diagnostic paired generation."""
    # H: take up to 6 (prioritize all if fewer)
    h_selected = h_tasks[:6]

    # S: take up to 6 from the beginning (sorted by task_id)
    s_selected = s_tasks[:6]

    print(f"\nDiagnostic subset:", flush=True)
    print(f"  H selected: {len(h_selected)} → {h_selected}", flush=True)
    print(f"  S selected: {len(s_selected)} → {s_selected}", flush=True)

    return h_selected, s_selected


def generate_diagnostic_paired(selected_tasks: list[str]) -> int:
    """Generate diagnostic paired records for selected tasks.

    Uses 3 candidates × 3 seeds per task.
    """
    if not selected_tasks:
        print("No tasks selected for diagnostic generation!", flush=True)
        return 1

    # Build a temporary candidate manifest with only selected tasks
    # Load full candidate manifest
    with open(CANDIDATE_MANIFEST) as f:
        cand_data = json.load(f)

    selected_set = set(selected_tasks)
    filtered_candidates = [c for c in cand_data["candidates"] if str(c["task_id"]) in selected_set]

    # Write filtered candidate manifest
    filtered_manifest = ARTIFACT_DIR / "stageA5_diagnostic_candidates"
    filtered_manifest.write_text(json.dumps({
        **cand_data,
        "candidates": filtered_candidates,
    }, indent=2), encoding="utf-8")

    print(f"Filtered candidates: {len(filtered_candidates)} entries for {len(selected_tasks)} tasks", flush=True)

    # Run paired generation: 3 seeds, limit to 3 candidates per task
    # Each task has 1 candidate → 3 candidates max means we keep up to 3
    cmd = [
        sys.executable, "-m", "smtr.marble.cli",
        "generate-database-paired-records",
        "--marble-root", str(MARBLE_ROOT),
        "--dataset-manifest", str(PROJECT_ROOT / "artifacts" / "marble" / "manifests" / "effect_check" / "dataset.json"),
        "--split-manifest", str(SPLIT_MANIFEST),
        "--split", "train",
        "--candidate-manifest", str(filtered_manifest),
        "--memory-pool", str(MEMORY_POOL),
        "--generation-seeds", "0", "1", "2",
        "--limit-pairs", str(3 * len(selected_tasks)),  # 3 edges per task
        "--output", str(DIAGNOSTIC_OUTPUT),
    ]

    return run_cmd(cmd, "diagnostic_paired_gen", ARTIFACT_DIR / "diagnostic_gen_stdout.log")


def main() -> int:
    print("=" * 60, flush=True)
    print("Stage A.5-Fast Pipeline", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Analyze train_base
    print("\n--- Step 1: Analyze train_base ---", flush=True)
    base_stats = analyze_train_base()
    if not base_stats:
        return 1

    # Step 2: Run screening
    print("\n--- Step 2: Control-only screening ---", flush=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "stage_a5_control_screening.py"),
        "--marble-root", str(MARBLE_ROOT),
        "--split-manifest", str(SPLIT_MANIFEST),
        "--split", "train",
        "--candidate-manifest", str(CANDIDATE_MANIFEST),
        "--train-base-tasks", TRAIN_BASE_TASKS,
        "--generation-seed", "0",
        "--output", str(SCREENING_OUTPUT),
    ]
    rc = run_cmd(cmd, "screening", ARTIFACT_DIR / "screening_stdout.log")
    if rc != 0:
        print("Screening failed!", flush=True)
        return rc

    # Step 3: Analyze screening and select subset
    print("\n--- Step 3: Select diagnostic subset ---", flush=True)
    h_tasks, s_tasks = analyze_screening()

    if len(h_tasks) == 0:
        print("\n*** BENCHMARK PROBLEM: No H tasks found! ***", flush=True)
        print("All train controls succeeded → no positive-transfer opportunity.", flush=True)
        print("Stopping pipeline. Manual analysis required.", flush=True)
        # Still write subset file for record
        subset_path = ARTIFACT_DIR / "stageA5_diagnostic_subset.json"
        subset_path.write_text(json.dumps({
            "h_tasks": [], "s_tasks": s_tasks[:6],
            "verdict": "BENCHMARK_PROBLEM",
        }, indent=2))
        return 0

    h_selected, s_selected = select_diagnostic_subset(h_tasks, s_tasks)
    all_selected = h_selected + s_selected

    # Write subset selection
    subset_path = ARTIFACT_DIR / "stageA5_diagnostic_subset.json"
    subset_path.write_text(json.dumps({
        "h_tasks": h_selected,
        "s_tasks": s_selected,
        "all_selected": all_selected,
        "h_total_available": len(h_tasks),
        "s_total_available": len(s_tasks),
    }, indent=2))

    # Step 4: Generate diagnostic paired records
    print("\n--- Step 4: Diagnostic paired generation ---", flush=True)
    rc = generate_diagnostic_paired(all_selected)
    if rc != 0:
        print("Diagnostic generation failed!", flush=True)
        return rc

    print("\n" + "=" * 60, flush=True)
    print("Pipeline complete!", flush=True)
    print(f"  train_base: {base_stats}", flush=True)
    print(f"  H tasks available: {len(h_tasks)}, selected: {len(h_selected)}", flush=True)
    print(f"  S tasks available: {len(s_tasks)}, selected: {len(s_selected)}", flush=True)
    print(f"  Diagnostic output: {DIAGNOSTIC_OUTPUT}", flush=True)
    print("=" * 60, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
