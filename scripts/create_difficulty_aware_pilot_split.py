"""Create difficulty-aware pilot task split.

Reads ``results/marble/difficulty_profile/task_difficulty_ranking.csv``
and generates:
  - data/marble_split/pilot_easy_tasks.json
  - data/marble_split/pilot_medium_tasks.json
  - data/marble_split/pilot_hard_tasks.json

Each file: 2 easy + 2 medium + 2 hard tasks per domain.

Output:
  - data/marble_split/pilot_{easy,medium,hard}_tasks.json
  - docs/audit/pilot_task_selection.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")

# How many tasks per difficulty per domain
TASKS_PER_DIFFICULTY_PER_DOMAIN = 2


def load_ranking(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["mean_reward"] = float(row["mean_reward"])
            rows.append(row)
    return rows


def classify(difficulty: str) -> str:
    """Normalize the difficulty column."""
    d = difficulty.strip().lower()
    if d in ("easy", "medium", "hard"):
        return d
    return "medium"


def select_pilot_tasks(
    ranked: list[dict[str, Any]],
    n_per_difficulty: int = TASKS_PER_DIFFICULTY_PER_DOMAIN,
) -> dict[str, list[dict[str, str]]]:
    """Select pilot tasks stratified by difficulty and domain.

    Returns dict with keys: easy, medium, hard.
    Each value is a list of {domain, task_id} dicts.
    """
    result: dict[str, list[dict[str, str]]] = {"easy": [], "medium": [], "hard": []}

    for difficulty in ("easy", "medium", "hard"):
        for domain in ALL_SCENARIOS:
            candidates = [
                r for r in ranked
                if r["domain"] == domain and classify(r["difficulty"]) == difficulty
            ]
            selected = candidates[:n_per_difficulty]
            for s in selected:
                result[difficulty].append({
                    "domain": s["domain"],
                    "task_id": s["task_id"],
                    "mean_reward": str(s["mean_reward"]),
                })

    return result


def write_json(tasks: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write only domain + task_id (strip metadata)
    clean = [{"domain": t["domain"], "task_id": t["task_id"]} for t in tasks]
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(f"Written: {path} ({len(clean)} tasks)")


def write_selection_report(
    pilot_tasks: dict[str, list[dict[str, str]]],
    ranked: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Pilot Task Selection\n")
    lines.append("**Method**: Difficulty-aware stratified sampling\n")
    lines.append(f"**Source**: {len(ranked)} tasks from difficulty profiling\n")
    lines.append(f"**Per difficulty per domain**: {TASKS_PER_DIFFICULTY_PER_DOMAIN} tasks\n")

    lines.append("\n## Selected Tasks\n")
    for difficulty in ("hard", "medium", "easy"):
        tasks = pilot_tasks[difficulty]
        lines.append(f"\n### {difficulty.title()} ({len(tasks)} tasks)\n")
        lines.append("| Domain | Task ID | Mean Reward |")
        lines.append("|--------|---------|-------------|")
        for t in tasks:
            lines.append(f"| {t['domain']} | {t['task_id']} | {t.get('mean_reward', 'N/A')} |")

    lines.append("\n## Selection Rationale\n")
    lines.append("Tasks are selected based on **measured difficulty** from the no_memory baseline,")
    lines.append("not manual choice. Each difficulty tier is represented across all 5 domains.")
    lines.append("")
    lines.append("- **Hard**: reward ≤ 0.5 → baseline fails, memory opportunity exists")
    lines.append("- **Medium**: 0.5 < reward ≤ 0.9 → partial improvement margin")
    lines.append("- **Easy**: reward > 0.9 → ceiling effect, control group")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create difficulty-aware pilot task split"
    )
    parser.add_argument(
        "--input", type=str,
        default=str(_PROJECT_ROOT / "results" / "marble" / "difficulty_profile" / "task_difficulty_ranking.csv"),
        help="Input ranking CSV",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_PROJECT_ROOT / "data" / "marble_split"),
        help="Output directory for JSON split files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run analyze_task_difficulty.py first.")
        sys.exit(1)

    ranked = load_ranking(input_path)
    pilot_tasks = select_pilot_tasks(ranked)

    for difficulty in ("easy", "medium", "hard"):
        write_json(pilot_tasks[difficulty], output_dir / f"pilot_{difficulty}_tasks.json")

    report_path = _PROJECT_ROOT / "docs" / "audit" / "pilot_task_selection.md"
    write_selection_report(pilot_tasks, ranked, report_path)

    # Summary
    print(f"\nSummary:")
    for d, tasks in pilot_tasks.items():
        domains = set(t["domain"] for t in tasks)
        print(f"  {d}: {len(tasks)} tasks across {len(domains)} domains")


if __name__ == "__main__":
    main()
