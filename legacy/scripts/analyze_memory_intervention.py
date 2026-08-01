"""Analyze memory intervention experiment results.

Computes per-(task, seed, memory_type) effects and generates summary reports.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def analyze_intervention(
    *,
    results_path: Path,
    output_dir: Path,
    memory_manifest_path: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results: list[dict] = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            results.append(r)

    # Filter to valid runs
    valid = [r for r in results if r.get("status") == "valid_complete"]
    invalid = [r for r in results if r.get("status") == "invalid_complete"]
    failed = [r for r in results if r.get("status") == "failed"]

    # Write all_runs.csv
    all_runs_path = output_dir / "all_runs.csv"
    with all_runs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_key", "task_id", "seed", "memory_type", "status",
            "score_delta", "f1_effect", "share_f1", "withhold_f1",
            "share_success", "withhold_success",
            "share_predicted", "withhold_predicted",
        ])
        for r in valid:
            te = r.get("treatment_effect", {})
            writer.writerow([
                r["pair_key"],
                te.get("task_id", r.get("pair_key", "").split("_")[1]),
                r.get("seed", ""),
                r.get("memory_type", ""),
                r["status"],
                te.get("score_delta", 0),
                te.get("f1_effect", 0),
                te.get("share_f1", 0),
                te.get("withhold_f1", 0),
                te.get("share_success", False),
                te.get("withhold_success", False),
                te.get("share_predicted", []),
                te.get("withhold_predicted", []),
            ])

    # Aggregate by (task_id, seed, memory_type)
    effects_by_key: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        te = r.get("treatment_effect", {})
        # Extract task_id from pair_key
        pair_key = r["pair_key"]
        parts = pair_key.split("_")
        task_id = parts[1]  # interv_51_beneficial_s41
        key = f"{task_id}_{r.get('seed')}_{r.get('memory_type')}"
        effects_by_key[key].append({
            "task_id": task_id,
            "seed": r.get("seed"),
            "memory_type": r.get("memory_type"),
            "score_delta": te.get("score_delta", 0),
            "f1_effect": te.get("f1_effect", 0),
            "native_effect": te.get("native_effect", 0),
        })

    # Compute per-(task, seed, memory_type) mean effects
    task_seed_effects: list[dict] = []
    for key, entries in sorted(effects_by_key.items()):
        mean_score = sum(e["score_delta"] for e in entries) / len(entries)
        mean_f1 = sum(e["f1_effect"] for e in entries) / len(entries)
        task_seed_effects.append({
            "task_id": entries[0]["task_id"],
            "seed": entries[0]["seed"],
            "memory_type": entries[0]["memory_type"],
            "score_delta": round(mean_score, 4),
            "f1_effect": round(mean_f1, 4),
            "n_runs": len(entries),
        })

    # Write task_seed_effects.csv
    tse_path = output_dir / "task_seed_effects.csv"
    with tse_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "seed", "memory_type", "score_delta", "f1_effect", "n_runs"])
        for e in task_seed_effects:
            writer.writerow([
                e["task_id"], e["seed"], e["memory_type"],
                e["score_delta"], e["f1_effect"], e["n_runs"],
            ])

    # Aggregate by memory_type
    by_memory_type: dict[str, list[dict]] = defaultdict(list)
    for e in task_seed_effects:
        by_memory_type[e["memory_type"]].append(e)

    memory_type_summary: list[dict] = []
    for mt in ["beneficial", "irrelevant", "conflicting", "role_mismatched"]:
        entries = by_memory_type.get(mt, [])
        if not entries:
            continue
        mean_score = sum(e["score_delta"] for e in entries) / len(entries)
        mean_f1 = sum(e["f1_effect"] for e in entries) / len(entries)
        positive = sum(1 for e in entries if e["f1_effect"] > 0)
        negative = sum(1 for e in entries if e["f1_effect"] < 0)
        neutral = sum(1 for e in entries if e["f1_effect"] == 0)
        memory_type_summary.append({
            "memory_type": mt,
            "n_cases": len(entries),
            "mean_score_delta": round(mean_score, 4),
            "mean_f1_effect": round(mean_f1, 4),
            "positive_f1": positive,
            "negative_f1": negative,
            "neutral_f1": neutral,
        })

    # Write memory_type_summary.csv
    mts_path = output_dir / "memory_type_summary.csv"
    with mts_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "memory_type", "n_cases", "mean_score_delta", "mean_f1_effect",
            "positive_f1", "negative_f1", "neutral_f1",
        ])
        for s in memory_type_summary:
            writer.writerow([
                s["memory_type"], s["n_cases"], s["mean_score_delta"],
                s["mean_f1_effect"], s["positive_f1"], s["negative_f1"],
                s["neutral_f1"],
            ])

    # Classify cases
    positive_cases = [e for e in task_seed_effects if e["f1_effect"] > 0]
    negative_cases = [e for e in task_seed_effects if e["f1_effect"] < 0]
    neutral_cases = [e for e in task_seed_effects if e["f1_effect"] == 0]

    # Write case files
    for name, cases in [
        ("positive_cases.json", positive_cases),
        ("negative_cases.json", negative_cases),
        ("neutral_cases.json", neutral_cases),
    ]:
        path = output_dir / name
        path.write_text(
            json.dumps(cases, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Coverage checks
    positive_tasks = set(e["task_id"] for e in positive_cases)
    negative_tasks = set(e["task_id"] for e in negative_cases)

    # Experiment report
    report_lines = [
        "# Memory Intervention Experiment Report",
        "",
        "## Summary",
        f"- Total runs: {len(results)}",
        f"- Valid: {len(valid)}, Invalid: {len(invalid)}, Failed: {len(failed)}",
        f"- Validity rate: {len(valid)/max(len(results),1):.1%}",
        "",
        "## Memory Type Effects",
        "",
        "| Memory Type | N Cases | Mean Score Δ | Mean F1 Effect | Positive | Negative | Neutral |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in memory_type_summary:
        report_lines.append(
            f"| {s['memory_type']} | {s['n_cases']} | {s['mean_score_delta']:+.4f} | "
            f"{s['mean_f1_effect']:+.4f} | {s['positive_f1']} | {s['negative_f1']} | {s['neutral_f1']} |"
        )

    report_lines.extend([
        "",
        "## Diagnostic Thresholds",
        f"- Positive cases (F1 effect > 0): {len(positive_cases)} (need ≥8)",
        f"- Negative cases (F1 effect < 0): {len(negative_cases)} (need ≥8)",
        f"- Positive task coverage: {len(positive_tasks)} tasks: {sorted(positive_tasks)} (need ≥3)",
        f"- Negative task coverage: {len(negative_tasks)} tasks: {sorted(negative_tasks)} (need ≥3)",
        "",
        "## Per-Memory-Type Analysis",
        "",
    ])

    for s in memory_type_summary:
        mt = s["memory_type"]
        report_lines.append(f"### {mt}")
        report_lines.append(f"- Mean F1 effect: {s['mean_f1_effect']:+.4f}")
        report_lines.append(f"- Direction: {s['positive_f1']}↑ {s['negative_f1']}↓ {s['neutral_f1']}→")
        report_lines.append("")

    # Check beneficial > 0 and conflicting < 0
    beneficial_f1 = next(
        (s["mean_f1_effect"] for s in memory_type_summary if s["memory_type"] == "beneficial"),
        None,
    )
    conflicting_f1 = next(
        (s["mean_f1_effect"] for s in memory_type_summary if s["memory_type"] == "conflicting"),
        None,
    )
    direction_benificial_ok = beneficial_f1 is not None and beneficial_f1 > 0
    direction_conflicting_ok = conflicting_f1 is not None and conflicting_f1 < 0

    thresholds_met = (
        len(positive_cases) >= 8
        and len(negative_cases) >= 8
        and len(positive_tasks) >= 3
        and len(negative_tasks) >= 3
        and direction_benificial_ok
        and direction_conflicting_ok
    )
    if beneficial_f1 is not None:
        report_lines.append(f"Beneficial avg F1 effect: {beneficial_f1:+.4f} (target: > 0)")
    if conflicting_f1 is not None:
        report_lines.append(f"Conflicting avg F1 effect: {conflicting_f1:+.4f} (target: < 0)")

    report_lines.extend([
        "",
        f"## Overall Assessment: {'PASS' if thresholds_met else 'BELOW THRESHOLD'}",
        "",
        f"Thresholds met: {thresholds_met}",
    ])

    report_path = output_dir / "experiment_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Print summary
    print("\n=== Memory Intervention Analysis ===")
    print(f"Total runs: {len(results)} (valid={len(valid)}, invalid={len(invalid)}, failed={len(failed)})")
    print(f"\nMemory type effects:")
    for s in memory_type_summary:
        print(f"  {s['memory_type']:18s}  n={s['n_cases']:2d}  "
              f"score_Δ={s['mean_score_delta']:+.4f}  f1_effect={s['mean_f1_effect']:+.4f}  "
              f"({s['positive_f1']}↑ {s['negative_f1']}↓ {s['neutral_f1']}→)")
    print(f"\nThresholds: positive={len(positive_cases)}(≥8) negative={len(negative_cases)}(≥8) "
          f"pos_tasks={len(positive_tasks)}(≥3) neg_tasks={len(negative_tasks)}(≥3)")
    print(f"Overall: {'PASS' if thresholds_met else 'BELOW THRESHOLD'}")

    return {
        "total_runs": len(results),
        "valid": len(valid),
        "thresholds_met": thresholds_met,
        "positive_cases": len(positive_cases),
        "negative_cases": len(negative_cases),
        "memory_type_summary": memory_type_summary,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze memory intervention results")
    parser.add_argument(
        "--results",
        default="artifacts/paper_experiments/memory_intervention/run_output/intervention_results.jsonl",
    )
    parser.add_argument(
        "--memory-manifest",
        default="artifacts/paper_experiments/memory_intervention/memory_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/paper_experiments/memory_intervention/analysis",
    )
    args = parser.parse_args()

    analyze_intervention(
        results_path=Path(args.results),
        output_dir=Path(args.output),
        memory_manifest_path=Path(args.memory_manifest),
    )


if __name__ == "__main__":
    main()
