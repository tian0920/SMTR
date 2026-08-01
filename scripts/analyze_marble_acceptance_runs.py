"""Analyze 27-run MARBLE acceptance batch evaluation results.

Scans artifacts/marble/acceptance/batch_eval/ for run directories,
extracts run-level metrics, computes method-level summaries,
paired comparisons, and completeness diagnostics.

Outputs:
  - acceptance_runs.csv / .jsonl  (per-run data)
  - acceptance_method_summary.csv / .json  (method-level aggregates)
  - acceptance_paired_comparisons.csv  (paired task-level diffs)
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

DEFAULT_BATCH_DIR = Path("artifacts/marble/acceptance/batch_eval")


def _parse_run_dir_name(name: str) -> dict[str, str] | None:
    """Parse task{N}_seed{S}_{method} directory name."""
    m = re.match(r"^task(\d+)_seed(\d+)_(.+)$", name)
    if not m:
        return None
    return {"task_id": m.group(1), "seed": m.group(2), "method": m.group(3)}


def _load_run_result(run_dir: Path) -> dict[str, Any] | None:
    """Load the run result JSON from a run directory."""
    # Try batch_summary.json style (single run result in directory)
    for candidate in [
        run_dir / "run_result.json",
        run_dir / "b0_smoke.json",
        run_dir / "paired_smoke.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _extract_from_batch_summary(
    batch_summary_path: Path,
) -> list[dict[str, Any]]:
    """Extract per-run records from the batch_summary.json file."""
    if not batch_summary_path.exists():
        return []
    data = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return []


def _extract_run_records(batch_dir: Path) -> list[dict[str, Any]]:
    """Extract per-run records by scanning subdirectories."""
    records: list[dict[str, Any]] = []
    for entry in sorted(batch_dir.iterdir()):
        if not entry.is_dir():
            continue
        parsed = _parse_run_dir_name(entry.name)
        if not parsed:
            continue
        # Look for run result inside workspace subdirectory
        for workspace_dir in entry.iterdir():
            if not workspace_dir.is_dir():
                continue
            # Check for engine process result
            engine_result_path = workspace_dir / "engine_process.json"
            marble_output = workspace_dir / "marble_output.jsonl"
            run_info = {
                "run_dir": entry.name,
                "task_id": parsed["task_id"],
                "seed": parsed["seed"],
                "method": parsed["method"],
                "workspace": workspace_dir.name,
            }
            if marble_output.exists():
                run_info["marble_output_exists"] = True
                # Try to read last line for evaluation
                lines = marble_output.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    try:
                        last = json.loads(lines[-1])
                        run_info["has_evaluation"] = "task_evaluation" in last
                    except json.JSONDecodeError:
                        run_info["has_evaluation"] = False
            records.append(run_info)
    return records


def _compute_method_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-method aggregate statistics."""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        method = run.get("method", "unknown")
        by_method.setdefault(method, []).append(run)

    summaries: dict[str, Any] = {}
    for method, method_runs in sorted(by_method.items()):
        scores = [
            r.get("outcome", {}).get("score", 0.0)
            for r in method_runs
            if isinstance(r.get("outcome"), dict)
        ]
        successes = [
            r.get("outcome", {}).get("success", False)
            for r in method_runs
            if isinstance(r.get("outcome"), dict)
        ]
        wall_clocks = [
            r.get("wall_clock_seconds", 0.0)
            for r in method_runs
            if r.get("wall_clock_seconds") is not None
        ]
        summaries[method] = {
            "run_count": len(method_runs),
            "success_count": sum(1 for s in successes if s),
            "success_rate": sum(1 for s in successes if s) / max(len(successes), 1),
            "mean_score": statistics.mean(scores) if scores else 0.0,
            "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "mean_wall_clock": statistics.mean(wall_clocks) if wall_clocks else 0.0,
        }
    return summaries


def _compute_paired_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paired comparisons: for each (task_id, seed), compare methods."""
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for run in runs:
        task_id = str(run.get("task_id", ""))
        seed = str(run.get("generation_seed", run.get("seed", "")))
        method = run.get("method", "")
        key = (task_id, seed)
        by_key.setdefault(key, {})[method] = run

    comparisons: list[dict[str, Any]] = []
    for (task_id, seed), methods in sorted(by_key.items()):
        b0 = methods.get("b0_no_memory", {})
        all_share = methods.get("all_share", {})
        smtr = methods.get("smtr", {})

        b0_score = b0.get("outcome", {}).get("score", None) if isinstance(b0.get("outcome"), dict) else None
        as_score = all_share.get("outcome", {}).get("score", None) if isinstance(all_share.get("outcome"), dict) else None
        smtr_score = smtr.get("outcome", {}).get("score", None) if isinstance(smtr.get("outcome"), dict) else None

        comparisons.append({
            "task_id": task_id,
            "seed": seed,
            "b0_score": b0_score,
            "all_share_score": as_score,
            "smtr_score": smtr_score,
            "smtr_minus_b0": (
                (smtr_score - b0_score) if smtr_score is not None and b0_score is not None else None
            ),
            "smtr_minus_all_share": (
                (smtr_score - as_score) if smtr_score is not None and as_score is not None else None
            ),
            "all_share_minus_b0": (
                (as_score - b0_score) if as_score is not None and b0_score is not None else None
            ),
        })
    return comparisons


def _check_completeness(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Check experiment completeness diagnostics."""
    methods_seen: set[str] = set()
    tasks_seen: set[str] = set()
    seeds_seen: set[str] = set()
    for run in runs:
        methods_seen.add(run.get("method", ""))
        tasks_seen.add(str(run.get("task_id", "")))
        seeds_seen.add(str(run.get("generation_seed", run.get("seed", ""))))

    # Ceiling effect: all scores = 1.0
    all_scores = [
        r.get("outcome", {}).get("score", None)
        for r in runs
        if isinstance(r.get("outcome"), dict)
    ]
    all_scores = [s for s in all_scores if s is not None]
    ceiling_effect = all(s >= 1.0 for s in all_scores) if all_scores else False

    # Neutral collapse: all methods produce same score for every task
    comparisons = _compute_paired_comparisons(runs)
    neutral_diffs = [
        c["smtr_minus_b0"] for c in comparisons
        if c["smtr_minus_b0"] is not None
    ]
    neutral_collapse = all(d == 0.0 for d in neutral_diffs) if neutral_diffs else False

    return {
        "total_runs": len(runs),
        "methods": sorted(methods_seen),
        "tasks": sorted(tasks_seen, key=lambda x: int(x) if x.isdigit() else x),
        "seeds": sorted(seeds_seen),
        "expected_methods": sorted({"b0_no_memory", "all_share", "smtr"}),
        "methods_missing": sorted({"b0_no_memory", "all_share", "smtr"} - methods_seen),
        "ceiling_effect": ceiling_effect,
        "neutral_collapse": neutral_collapse,
        "score_variance": statistics.variance(all_scores) if len(all_scores) > 1 else 0.0,
    }


def analyze_acceptance_runs(
    batch_dir: Path = DEFAULT_BATCH_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Main entry point: analyze acceptance batch runs."""
    if output_dir is None:
        output_dir = batch_dir

    # Load from batch_summary.json (preferred) or scan directories
    batch_summary_path = batch_dir / "batch_summary.json"
    runs = _extract_from_batch_summary(batch_summary_path)
    if not runs:
        runs = _extract_run_records(batch_dir)

    if not runs:
        return {"error": "no_runs_found", "batch_dir": str(batch_dir)}

    # Per-run CSV/JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv_path = output_dir / "acceptance_runs.csv"
    with run_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "generation_seed", "method", "scenario",
            "real_engine_executed", "native_evaluator_executed",
            "score", "success", "wall_clock_seconds",
        ])
        writer.writeheader()
        for run in runs:
            outcome = run.get("outcome", {}) or {}
            writer.writerow({
                "task_id": run.get("task_id", ""),
                "generation_seed": run.get("generation_seed", ""),
                "method": run.get("method", ""),
                "scenario": run.get("scenario", ""),
                "real_engine_executed": run.get("real_engine_executed", ""),
                "native_evaluator_executed": outcome.get("native_evaluator_executed", ""),
                "score": outcome.get("score", ""),
                "success": outcome.get("success", ""),
                "wall_clock_seconds": run.get("wall_clock_seconds", ""),
            })

    run_jsonl_path = output_dir / "acceptance_runs.jsonl"
    with run_jsonl_path.open("w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, sort_keys=True) + "\n")

    # Method summary
    method_summary = _compute_method_summary(runs)
    summary_csv_path = output_dir / "acceptance_method_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "run_count", "success_count", "success_rate",
            "mean_score", "score_stdev", "mean_wall_clock",
        ])
        writer.writeheader()
        for method, stats in sorted(method_summary.items()):
            writer.writerow({"method": method, **stats})

    summary_json_path = output_dir / "acceptance_method_summary.json"
    summary_json_path.write_text(
        json.dumps(method_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Paired comparisons
    comparisons = _compute_paired_comparisons(runs)
    comp_csv_path = output_dir / "acceptance_paired_comparisons.csv"
    with comp_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "seed",
            "b0_score", "all_share_score", "smtr_score",
            "smtr_minus_b0", "smtr_minus_all_share", "all_share_minus_b0",
        ])
        writer.writeheader()
        for comp in comparisons:
            writer.writerow(comp)

    # Completeness diagnostics
    completeness = _check_completeness(runs)

    result = {
        "batch_dir": str(batch_dir),
        "total_runs": len(runs),
        "method_summary": method_summary,
        "paired_comparison_count": len(comparisons),
        "completeness": completeness,
        "outputs": {
            "runs_csv": str(run_csv_path),
            "runs_jsonl": str(run_jsonl_path),
            "method_summary_csv": str(summary_csv_path),
            "method_summary_json": str(summary_json_path),
            "paired_comparisons_csv": str(comp_csv_path),
        },
    }
    result_path = output_dir / "acceptance_analysis.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze MARBLE acceptance runs")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=DEFAULT_BATCH_DIR,
        help="Path to batch evaluation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to batch-dir)",
    )
    args = parser.parse_args()
    result = analyze_acceptance_runs(
        batch_dir=args.batch_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
