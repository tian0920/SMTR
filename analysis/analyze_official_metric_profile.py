"""Analyze official metric profiling results for backbone saturation assessment.

Input: results/marble/official_metric_profile/
Output: docs/audit/official_metric_backbone_profile.md

Computes:
- Per-scenario statistics (mean, std, min, max, quartiles)
- Valid evaluator rate
- Headroom distribution
- Saturation assessment
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_DIR = PROJECT_ROOT / "results" / "marble" / "official_metric_profile"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "audit"


def load_episodes() -> list[dict[str, Any]]:
    path = INPUT_DIR / "episode_scores.csv"
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Parse types
    for row in rows:
        row["seed"] = int(row["seed"])
        row["runtime"] = float(row["runtime"]) if row["runtime"] else 0.0
        row["normalized_task_score"] = (
            float(row["normalized_task_score"])
            if row["normalized_task_score"] and row["normalized_task_score"] != "None"
            else None
        )
        row["raw_task_score"] = (
            float(row["raw_task_score"])
            if row["raw_task_score"] and row["raw_task_score"] != "None"
            else None
        )
        row["metric_valid"] = row["metric_valid"] == "True"
        row["team_success"] = row["team_success"] == "True"
    return rows


def compute_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"N": 0}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "N": n,
        "mean": sum(values) / n,
        "std": (sum((v - sum(values)/n)**2 for v in values) / n) ** 0.5,
        "median": values_sorted[n // 2],
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "Q10": values_sorted[max(0, int(n * 0.10))],
        "Q25": values_sorted[max(0, int(n * 0.25))],
        "Q75": values_sorted[min(n - 1, int(n * 0.75))],
        "Q90": values_sorted[min(n - 1, int(n * 0.90))],
    }


def analyze_scenario(
    scenario: str, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    valid_eps = [e for e in episodes if e["metric_valid"]]
    valid_scores = [e["normalized_task_score"] for e in valid_eps]

    stats = compute_stats(valid_scores)
    n_total = len(episodes)
    n_valid = len(valid_eps)

    # Headroom = 1 - score
    headroom_values = [1.0 - s for s in valid_scores]
    frac_headroom_05 = sum(1 for h in headroom_values if h > 0.05) / n_valid if n_valid else 0
    frac_headroom_10 = sum(1 for h in headroom_values if h > 0.10) / n_valid if n_valid else 0
    frac_headroom_20 = sum(1 for h in headroom_values if h > 0.20) / n_valid if n_valid else 0

    frac_at_max = sum(1 for s in valid_scores if s >= 0.99) / n_valid if n_valid else 0
    frac_at_min = sum(1 for s in valid_scores if s <= 0.01) / n_valid if n_valid else 0

    team_success_rate = sum(1 for e in episodes if e["team_success"]) / n_total if n_total else 0

    return {
        "scenario": scenario,
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_evaluator_rate": n_valid / n_total if n_total else 0,
        "team_success_rate": team_success_rate,
        **stats,
        "fraction_at_max": frac_at_max,
        "fraction_at_min": frac_at_min,
        "fraction_headroom_gt_0_05": frac_headroom_05,
        "fraction_headroom_gt_0_10": frac_headroom_10,
        "fraction_headroom_gt_0_20": frac_headroom_20,
    }


def generate_report(scenario_stats: list[dict[str, Any]]) -> str:
    lines = [
        "# Official Metric Backbone Profile",
        "",
        f"**Date**: auto-generated",
        f"**Model**: qwen3-30b-a3b (no_memory baseline)",
        "",
        "## 1. Per-Scenario Statistics",
        "",
        "| Scenario | N | Valid Rate | Mean | Std | Min | Max | Q25 | Q75 | At Max | At Min |",
        "|----------|---|-----------|------|-----|-----|-----|-----|-----|--------|--------|",
    ]

    for s in scenario_stats:
        mean_str = f"{s['mean']:.3f}" if "mean" in s else "N/A"
        std_str = f"{s['std']:.3f}" if "std" in s else "N/A"
        min_str = f"{s['min']:.3f}" if "min" in s else "N/A"
        max_str = f"{s['max']:.3f}" if "max" in s else "N/A"
        q25_str = f"{s['Q25']:.3f}" if "Q25" in s else "N/A"
        q75_str = f"{s['Q75']:.3f}" if "Q75" in s else "N/A"
        lines.append(
            f"| {s['scenario']} | {s['n_total']} | "
            f"{s['valid_evaluator_rate']:.1%} | "
            f"{mean_str} | {std_str} | {min_str} | {max_str} | "
            f"{q25_str} | {q75_str} | "
            f"{s['fraction_at_max']:.1%} | {s['fraction_at_min']:.1%} |"
        )

    lines.extend([
        "",
        "## 2. Headroom Distribution",
        "",
        "| Scenario | Headroom>5% | Headroom>10% | Headroom>20% |",
        "|----------|-------------|--------------|--------------|",
    ])
    for s in scenario_stats:
        lines.append(
            f"| {s['scenario']} | "
            f"{s['fraction_headroom_gt_0_05']:.1%} | "
            f"{s['fraction_headroom_gt_0_10']:.1%} | "
            f"{s['fraction_headroom_gt_0_20']:.1%} |"
        )

    lines.extend([
        "",
        "## 3. Saturation Assessment",
        "",
        "### Comparison: Official Metric vs Binary Heuristic",
        "",
        "| Scenario | Binary team_success Rate | Official Mean Score | Saturated? |",
        "|----------|-------------------------|--------------------:|:----------:|",
    ])
    for s in scenario_stats:
        mean_str = f"{s['mean']:.3f}" if "mean" in s else "N/A"
        saturated = "❌ YES" if s.get("fraction_at_max", 0) > 0.8 else "✅ NO"
        lines.append(
            f"| {s['scenario']} | {s['team_success_rate']:.1%} | "
            f"{mean_str} | {saturated} |"
        )

    # Overall GO/NO-GO
    lines.extend(["", "## 4. Backbone GO/NO-GO", ""])
    overall_valid = sum(s["n_valid"] for s in scenario_stats) / sum(s["n_total"] for s in scenario_stats) if scenario_stats else 0
    all_means = [s.get("mean", 0) for s in scenario_stats if "mean" in s]
    overall_var = (
        sum((m - sum(all_means)/len(all_means))**2 for m in all_means) / len(all_means)
        if len(all_means) > 1 else 0
    )
    n_scenarios_with_std = sum(1 for s in scenario_stats if s.get("std", 0) >= 0.05)
    overall_headroom_10 = (
        sum(s["fraction_headroom_gt_0_10"] for s in scenario_stats) / len(scenario_stats)
        if scenario_stats else 0
    )
    max_frac_at_max = max(s.get("fraction_at_max", 0) for s in scenario_stats) if scenario_stats else 0

    c1 = overall_valid >= 0.95
    c2 = overall_var > 0
    c3 = n_scenarios_with_std >= 3
    c4 = overall_headroom_10 >= 0.20
    c5 = max_frac_at_max < 0.80

    go = all([c1, c2, c3, c4, c5])
    lines.extend([
        f"| Criterion | Value | Threshold | Pass? |",
        f"|-----------|-------|-----------|:-----:|",
        f"| Valid evaluator rate | {overall_valid:.1%} | ≥ 95% | {'✅' if c1 else '❌'} |",
        f"| Overall score variance | {overall_var:.4f} | > 0 | {'✅' if c2 else '❌'} |",
        f"| Scenarios with std ≥ 0.05 | {n_scenarios_with_std}/5 | ≥ 3 | {'✅' if c3 else '❌'} |",
        f"| Avg headroom > 10% | {overall_headroom_10:.1%} | ≥ 20% | {'✅' if c4 else '❌'} |",
        f"| Max fraction_at_max | {max_frac_at_max:.1%} | < 80% | {'✅' if c5 else '❌'} |",
        "",
        f"**Decision**: {'✅ **BACKBONE_GO**' if go else '❌ **BACKBONE_NO_GO**'}",
        "",
        "If BACKBONE_GO → proceed to TCI activation pilot (Phase H)",
        "If BACKBONE_NO-GO → run backbone difficulty sweep (Phase L)",
    ])

    return "\n".join(lines)


def main() -> None:
    episodes = load_episodes()
    scenarios = sorted(set(e["scenario"] for e in episodes))

    scenario_stats = []
    for scenario in scenarios:
        s_episodes = [e for e in episodes if e["scenario"] == scenario]
        stats = analyze_scenario(scenario, s_episodes)
        scenario_stats.append(stats)

    report = generate_report(scenario_stats)
    output_path = OUTPUT_DIR / "official_metric_backbone_profile.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")
    print(report)


if __name__ == "__main__":
    main()
