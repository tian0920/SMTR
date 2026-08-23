"""Analyze hard-task baseline pilot results.

Input:  results/marble/pilot_hard_baseline/episode_metrics.csv
Output: docs/audit/hard_task_baseline_report.md

Pass criteria:
  - mean_reward < 0.9   OR
  - failure_rate > 20%
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def load_episodes(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["team_reward"] = float(row["team_reward"])
            row["team_success"] = row["team_success"] in ("True", "true", "1")
            rows.append(row)
    return rows


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"error": "no data"}

    rewards = [r["team_reward"] for r in rows]
    successes = [r["team_success"] for r in rows]
    n = len(rows)

    mean_reward = sum(rewards) / n
    failure_rate = 1.0 - (sum(successes) / n)

    # Per-receiver disagreement (if applicable)
    domains = sorted(set(r.get("scenario", r.get("domain", "?")) for r in rows))

    # Per-domain breakdown
    per_domain: dict[str, dict[str, float]] = {}
    for d in domains:
        d_rows = [r for r in rows if r.get("scenario", r.get("domain", "?")) == d]
        d_rewards = [r["team_reward"] for r in d_rows]
        d_succ = [r["team_success"] for r in d_rows]
        per_domain[d] = {
            "n": len(d_rows),
            "mean_reward": sum(d_rewards) / len(d_rewards) if d_rewards else 0.0,
            "failure_rate": 1.0 - sum(d_succ) / len(d_succ) if d_succ else 0.0,
        }

    # Improvement margin: 1.0 - mean_reward
    improvement_margin = 1.0 - mean_reward

    # Pass criteria
    passes = mean_reward < 0.9 or failure_rate > 0.2

    return {
        "n_episodes": n,
        "mean_reward": round(mean_reward, 4),
        "std_reward": round((sum((r - mean_reward)**2 for r in rewards) / n) ** 0.5, 4),
        "failure_rate": round(failure_rate, 4),
        "improvement_margin": round(improvement_margin, 4),
        "per_domain": per_domain,
        "passes": passes,
    }


def write_report(stats: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Hard Task Baseline Report\n")

    if "error" in stats:
        lines.append(f"**ERROR**: {stats['error']}\n")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"**Episodes**: {stats['n_episodes']}\n")
    lines.append(f"**Mean reward**: {stats['mean_reward']:.4f}\n")
    lines.append(f"**Std reward**: {stats['std_reward']:.4f}\n")
    lines.append(f"**Failure rate**: {stats['failure_rate']:.2%}\n")
    lines.append(f"**Improvement margin**: {stats['improvement_margin']:.4f}\n")

    lines.append("\n## Per-Domain Breakdown\n")
    lines.append("| Domain | N | Mean Reward | Failure Rate |")
    lines.append("|--------|---|-------------|--------------|")
    for d, v in sorted(stats["per_domain"].items()):
        lines.append(f"| {d} | {v['n']} | {v['mean_reward']:.4f} | {v['failure_rate']:.2%} |")

    lines.append("\n## Pass Criteria\n")
    lines.append("| Criterion | Threshold | Actual | Pass? |")
    lines.append("|-----------|-----------|--------|-------|")
    lines.append(
        f"| Mean reward < 0.9 | < 0.9 | {stats['mean_reward']:.4f} "
        f"| {'PASS' if stats['mean_reward'] < 0.9 else 'FAIL'} |"
    )
    lines.append(
        f"| Failure rate > 20% | > 0.20 | {stats['failure_rate']:.2%} "
        f"| {'PASS' if stats['failure_rate'] > 0.2 else 'FAIL'} |"
    )

    verdict = "PASS" if stats["passes"] else "FAIL"
    lines.append(f"\n**Verdict**: {verdict}\n")
    if not stats["passes"]:
        lines.append("Action required: expand hard task set or investigate why tasks are too easy.")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_path}")
    print(f"Verdict: {'PASS' if stats['passes'] else 'FAIL'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze hard baseline pilot")
    parser.add_argument(
        "--input", type=str,
        default=str(_PROJECT_ROOT / "results" / "marble" / "pilot_hard_baseline" / "episode_metrics.csv"),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    rows = load_episodes(input_path)
    stats = analyze(rows)
    report_path = _PROJECT_ROOT / "docs" / "audit" / "hard_task_baseline_report.md"
    write_report(stats, report_path)


if __name__ == "__main__":
    main()
