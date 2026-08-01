"""Analyze B0 screening results and classify tasks as ceiling/floor/sweet-spot."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def analyze_screening(
    *,
    results_path: Path,
    output_dir: Path,
    task_selection_path: Path | None = None,
) -> dict:
    """Classify tasks based on B0 screening results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_by_task: dict[int, list[dict]] = defaultdict(list)
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "failed":
                continue
            results_by_task[r["task_id"]].append(r)

    # Load task metadata
    task_meta: dict[int, dict] = {}
    if task_selection_path and task_selection_path.exists():
        selection = json.loads(task_selection_path.read_text(encoding="utf-8"))
        for t in selection:
            task_meta[t["task_id"]] = t

    # Analyze each task
    task_stats: list[dict] = []
    for task_id in sorted(results_by_task.keys()):
        runs = results_by_task[task_id]
        n_seeds = len(runs)
        successes = sum(1 for r in runs if r["success"])
        success_rate = successes / max(n_seeds, 1)
        mean_f1 = sum(r["f1"] for r in runs) / max(n_seeds, 1)
        mean_recall = sum(r["recall"] for r in runs) / max(n_seeds, 1)
        mean_precision = sum(r["precision"] for r in runs) / max(n_seeds, 1)
        mean_tp = sum(r["tp"] for r in runs) / max(n_seeds, 1)
        mean_fp = sum(r["fp"] for r in runs) / max(n_seeds, 1)

        # Classify
        if successes == n_seeds or mean_f1 >= 0.9:
            category = "ceiling"
        elif successes == 0 and mean_f1 <= 0.2:
            category = "floor"
        else:
            category = "sweet_spot"

        meta = task_meta.get(task_id, {})
        stat = {
            "task_id": task_id,
            "domain": meta.get("domain", "unknown"),
            "rc_type": meta.get("rc_type", "unknown"),
            "root_causes": meta.get("root_causes", []),
            "n_seeds": n_seeds,
            "successes": successes,
            "success_rate": round(success_rate, 4),
            "mean_f1": round(mean_f1, 4),
            "mean_recall": round(mean_recall, 4),
            "mean_precision": round(mean_precision, 4),
            "mean_tp": round(mean_tp, 2),
            "mean_fp": round(mean_fp, 2),
            "category": category,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "success": r["success"],
                    "f1": r["f1"],
                    "tp": r["tp"],
                    "fp": r["fp"],
                    "predicted": r["predicted_labels"],
                }
                for r in runs
            ],
        }
        task_stats.append(stat)

    # Write task_screening.csv
    csv_path = output_dir / "task_screening.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "domain", "rc_type", "n_seeds", "successes",
            "success_rate", "mean_f1", "mean_recall", "mean_precision",
            "mean_tp", "mean_fp", "category",
        ])
        for s in task_stats:
            writer.writerow([
                s["task_id"], s["domain"], s["rc_type"], s["n_seeds"],
                s["successes"], s["success_rate"], s["mean_f1"],
                s["mean_recall"], s["mean_precision"], s["mean_tp"],
                s["mean_fp"], s["category"],
            ])

    # Select sweet spot tasks
    sweet_spot = [s for s in task_stats if s["category"] == "sweet_spot"]
    sweet_spot.sort(key=lambda x: abs(x["mean_f1"] - 0.5))  # closest to 0.5 first

    # If fewer than 8, add near-sweet-spot tasks
    near_sweet = [
        s for s in task_stats
        if s["category"] != "sweet_spot"
        and 0.2 < s["mean_f1"] < 0.9
    ]
    near_sweet.sort(key=lambda x: abs(x["mean_f1"] - 0.5))

    selected = sweet_spot[:8]
    if len(selected) < 8:
        needed = 8 - len(selected)
        selected_ids = {s["task_id"] for s in selected}
        for s in near_sweet:
            if s["task_id"] not in selected_ids:
                selected.append(s)
                selected_ids.add(s["task_id"])
                if len(selected) >= 8:
                    break

    selected_json = {
        "selected_task_ids": [s["task_id"] for s in selected],
        "count": len(selected),
        "selection_criteria": "sweet_spot (success_rate in {1/3,2/3} OR mean_f1 in [0.3,0.8])",
        "fallback_used": len(sweet_spot) < 8,
        "tasks": [
            {
                "task_id": s["task_id"],
                "domain": s["domain"],
                "rc_type": s["rc_type"],
                "root_causes": s["root_causes"],
                "success_rate": s["success_rate"],
                "mean_f1": s["mean_f1"],
                "category": s["category"],
            }
            for s in selected
        ],
    }
    selected_path = output_dir / "selected_sweet_spot_tasks.json"
    selected_path.write_text(
        json.dumps(selected_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Summary
    summary = {
        "total_tasks_analyzed": len(task_stats),
        "ceiling": sum(1 for s in task_stats if s["category"] == "ceiling"),
        "floor": sum(1 for s in task_stats if s["category"] == "floor"),
        "sweet_spot": len(sweet_spot),
        "selected_count": len(selected),
        "selected_task_ids": [s["task_id"] for s in selected],
    }

    print(f"\n=== B0 Screening Summary ===")
    print(f"Total tasks analyzed: {summary['total_tasks_analyzed']}")
    print(f"  Ceiling:    {summary['ceiling']}")
    print(f"  Floor:      {summary['floor']}")
    print(f"  Sweet Spot: {summary['sweet_spot']}")
    print(f"Selected (up to 8): {summary['selected_task_ids']}")
    print(f"\nPer-task breakdown:")
    for s in task_stats:
        marker = " <-- SELECTED" if s["task_id"] in [x["task_id"] for x in selected] else ""
        print(f"  Task {s['task_id']:3d} ({s['domain']:16s}) "
              f"success={s['successes']}/{s['n_seeds']} "
              f"f1={s['mean_f1']:.3f} "
              f"-> {s['category']}{marker}")

    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze B0 screening results")
    parser.add_argument(
        "--results",
        default="artifacts/marble/outputs/b0_screening/b0_screening_results.jsonl",
    )
    parser.add_argument(
        "--task-selection",
        default="artifacts/paper_experiments/memory_intervention/b0_screening/task_selection.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/paper_experiments/memory_intervention/b0_screening",
    )
    args = parser.parse_args()

    analyze_screening(
        results_path=Path(args.results),
        output_dir=Path(args.output),
        task_selection_path=Path(args.task_selection),
    )


if __name__ == "__main__":
    main()
