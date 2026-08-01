"""Analyze diagnostic 64-pair experiment results.

Reads diagnostic_results.jsonl and produces:
- diagnostic_pairs.csv
- diagnostic_pair_summary.json
- diagnostic_memory_type_summary.csv
- diagnostic_task_summary.csv
- diagnostic_order_effect.csv
- diagnostic_invalidity_summary.csv
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def analyze_diagnostic_pairs(
    run_dir: Path,
    output_dir: Path | None = None,
    epsilon: float = 0.0,
) -> dict[str, Any]:
    """Analyze diagnostic pair results."""
    if output_dir is None:
        output_dir = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "diagnostic_results.jsonl"
    if not results_path.exists():
        return {"error": "diagnostic_results.jsonl not found"}

    results: list[dict] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(json.loads(line))

    if not results:
        return {"error": "no results found"}

    total = len(results)
    valid = [r for r in results if r.get("status") == "valid_complete"]
    invalid = [r for r in results if r.get("status") == "invalid_complete"]
    failed_list = [r for r in results if r.get("status") == "failed"]

    # Transfer labels (among valid pairs)
    label_counts: dict[str, int] = {}
    for r in valid:
        label = r.get("paired_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    # Score effects
    score_deltas = [
        r["treatment_effect"]["score_delta"]
        for r in valid
        if "treatment_effect" in r
    ]
    success_deltas = [
        r["treatment_effect"].get("success_delta", 0.0)
        for r in valid
        if "treatment_effect" in r
    ]

    # Transfer classification using epsilon
    positive = sum(1 for d in score_deltas if d > epsilon)
    negative = sum(1 for d in score_deltas if d < -epsilon)
    neutral = sum(1 for d in score_deltas if -epsilon <= d <= epsilon)

    # Overall summary
    summary: dict[str, Any] = {
        "total_pairs": total,
        "valid_pairs": len(valid),
        "invalid_pairs": len(invalid),
        "failed_pairs": len(failed_list),
        "validity_rate": len(valid) / max(total, 1),
        "first_attempt_valid": sum(1 for r in results if r.get("first_attempt_valid")),
        "label_counts": label_counts,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "positive_rate": positive / max(len(score_deltas), 1),
        "negative_rate": negative / max(len(score_deltas), 1),
        "neutral_rate": neutral / max(len(score_deltas), 1),
    }

    if score_deltas:
        summary["score_effect"] = {
            "mean": statistics.mean(score_deltas),
            "median": statistics.median(score_deltas),
            "stdev": statistics.stdev(score_deltas) if len(score_deltas) > 1 else 0.0,
            "min": min(score_deltas),
            "max": max(score_deltas),
        }
    if success_deltas:
        summary["success_effect"] = {
            "mean": statistics.mean(success_deltas),
            "median": statistics.median(success_deltas),
        }

    # Token/round effects
    for metric in ("tokens", "rounds"):
        deltas = [
            r["treatment_effect"][f"{metric}_delta"]
            for r in valid
            if "treatment_effect" in r and f"{metric}_delta" in r["treatment_effect"]
        ]
        if deltas:
            summary[f"{metric}_effect"] = {
                "mean": statistics.mean(deltas),
                "median": statistics.median(deltas),
                "count": len(deltas),
            }

    # --- Per memory type ---
    mem_type_data: dict[str, list[dict]] = {}
    for r in valid:
        mt = r.get("memory_type", "unknown")
        mem_type_data.setdefault(mt, []).append(r)

    mem_type_rows = []
    for mt in sorted(mem_type_data):
        rows = mem_type_data[mt]
        deltas = [
            r["treatment_effect"]["score_delta"]
            for r in rows
            if "treatment_effect" in r
        ]
        pos = sum(1 for d in deltas if d > epsilon)
        neg = sum(1 for d in deltas if d < -epsilon)
        neu = sum(1 for d in deltas if -epsilon <= d <= epsilon)
        mem_type_rows.append({
            "memory_type": mt,
            "pairs": len(rows),
            "mean_effect": statistics.mean(deltas) if deltas else 0.0,
            "median_effect": statistics.median(deltas) if deltas else 0.0,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
        })

    # --- Per task ---
    task_data: dict[str, list[dict]] = {}
    for r in valid:
        tid = r.get("task_id", "unknown")
        task_data.setdefault(tid, []).append(r)

    task_rows = []
    for tid in sorted(task_data, key=lambda x: int(x) if x.isdigit() else 0):
        rows = task_data[tid]
        deltas = [
            r["treatment_effect"]["score_delta"]
            for r in rows
            if "treatment_effect" in r
        ]
        task_rows.append({
            "task_id": tid,
            "pairs": len(rows),
            "mean_effect": statistics.mean(deltas) if deltas else 0.0,
            "median_effect": statistics.median(deltas) if deltas else 0.0,
            "positive": sum(1 for d in deltas if d > epsilon),
            "negative": sum(1 for d in deltas if d < -epsilon),
            "neutral": sum(1 for d in deltas if -epsilon <= d <= epsilon),
            "all_share_score_1": sum(
                1 for r in rows
                if r.get("share_score") == 1.0 and r.get("withhold_score") == 1.0
            ),
            "all_share_score_0": sum(
                1 for r in rows
                if r.get("share_score") == 0.0 and r.get("withhold_score") == 0.0
            ),
        })

    # --- Order effect ---
    order_data: dict[str, list[dict]] = {}
    for r in valid:
        order = r.get("branch_execution_order", "unknown")
        order_data.setdefault(order, []).append(r)

    order_rows = []
    for order in sorted(order_data):
        rows = order_data[order]
        deltas = [
            r["treatment_effect"]["score_delta"]
            for r in rows
            if "treatment_effect" in r
        ]
        order_rows.append({
            "execution_order": order,
            "pairs": len(rows),
            "mean_effect": statistics.mean(deltas) if deltas else 0.0,
            "validity_rate": len(rows) / max(
                sum(1 for r in results if r.get("branch_execution_order") == order), 1
            ),
        })

    # --- Invalidity analysis ---
    invalid_reasons: dict[str, int] = {}
    for r in invalid:
        reason = r.get("invalid_reason", "unknown")
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    invalidity_rows = []
    # By memory type
    for mt in sorted(set(r.get("memory_type", "") for r in invalid)):
        mt_invalid = sum(1 for r in invalid if r.get("memory_type") == mt)
        mt_total = sum(1 for r in results if r.get("memory_type") == mt)
        invalidity_rows.append({
            "dimension": "memory_type",
            "value": mt,
            "invalid_count": mt_invalid,
            "total_count": mt_total,
            "invalid_rate": mt_invalid / max(mt_total, 1),
        })
    # By task
    for tid in sorted(set(r.get("task_id", "") for r in invalid)):
        t_invalid = sum(1 for r in invalid if r.get("task_id") == tid)
        t_total = sum(1 for r in results if r.get("task_id") == tid)
        invalidity_rows.append({
            "dimension": "task_id",
            "value": tid,
            "invalid_count": t_invalid,
            "total_count": t_total,
            "invalid_rate": t_invalid / max(t_total, 1),
        })

    # --- Write outputs ---
    # diagnostic_pairs.csv
    csv_path = output_dir / "diagnostic_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_key", "pair_id", "task_id", "memory_id", "memory_type",
            "seed", "status", "paired_label", "share_success", "withhold_success",
            "share_score", "withhold_score", "score_delta",
            "share_tokens", "withhold_tokens",
            "share_rounds", "withhold_rounds",
            "share_runtime_visibility_verified",
            "withhold_runtime_visibility_verified",
            "initial_fingerprint_match", "first_attempt_valid",
            "invalid_reason",
        ])
        writer.writeheader()
        for r in results:
            te = r.get("treatment_effect", {})
            sm = r.get("share_metrics", {})
            wm = r.get("withhold_metrics", {})
            writer.writerow({
                "pair_key": r.get("pair_key", ""),
                "pair_id": r.get("pair_id", ""),
                "task_id": r.get("task_id", ""),
                "memory_id": r.get("memory_id", ""),
                "memory_type": r.get("memory_type", ""),
                "seed": r.get("seed", ""),
                "status": r.get("status", ""),
                "paired_label": r.get("paired_label", ""),
                "share_success": r.get("share_success", ""),
                "withhold_success": r.get("withhold_success", ""),
                "share_score": r.get("share_score", ""),
                "withhold_score": r.get("withhold_score", ""),
                "score_delta": te.get("score_delta", ""),
                "share_tokens": sm.get("tokens", ""),
                "withhold_tokens": wm.get("tokens", ""),
                "share_rounds": sm.get("rounds", ""),
                "withhold_rounds": wm.get("rounds", ""),
                "share_runtime_visibility_verified": r.get("share_runtime_visibility_verified", ""),
                "withhold_runtime_visibility_verified": r.get("withhold_runtime_visibility_verified", ""),
                "initial_fingerprint_match": r.get("initial_fingerprint_match", ""),
                "first_attempt_valid": r.get("first_attempt_valid", ""),
                "invalid_reason": r.get("invalid_reason", ""),
            })

    # diagnostic_pair_summary.json
    summary["memory_type_summary"] = mem_type_rows
    summary["task_summary"] = task_rows
    summary["order_effect"] = order_rows
    summary["invalidity"] = {
        "reasons": invalid_reasons,
        "by_dimension": invalidity_rows,
    }
    summary["epsilon"] = epsilon

    (output_dir / "diagnostic_pair_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    # diagnostic_memory_type_summary.csv
    mt_path = output_dir / "diagnostic_memory_type_summary.csv"
    with mt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "memory_type", "pairs", "mean_effect", "median_effect",
            "positive", "neutral", "negative",
        ])
        writer.writeheader()
        for row in mem_type_rows:
            writer.writerow(row)

    # diagnostic_task_summary.csv
    ts_path = output_dir / "diagnostic_task_summary.csv"
    with ts_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "pairs", "mean_effect", "median_effect",
            "positive", "negative", "neutral",
            "all_share_score_1", "all_share_score_0",
        ])
        writer.writeheader()
        for row in task_rows:
            writer.writerow(row)

    # diagnostic_order_effect.csv
    oe_path = output_dir / "diagnostic_order_effect.csv"
    with oe_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "execution_order", "pairs", "mean_effect", "validity_rate",
        ])
        writer.writeheader()
        for row in order_rows:
            writer.writerow(row)

    # diagnostic_invalidity_summary.csv
    inv_path = output_dir / "diagnostic_invalidity_summary.csv"
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dimension", "value", "invalid_count", "total_count", "invalid_rate",
        ])
        writer.writeheader()
        for row in invalidity_rows:
            writer.writerow(row)

    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze diagnostic pairs")
    parser.add_argument(
        "--run-dir",
        default="artifacts/paper_experiments/diagnostic_64/run_output",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    result = analyze_diagnostic_pairs(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        epsilon=args.epsilon,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
