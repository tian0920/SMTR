"""Phase 3: Analyze iteration-level signals from MARBLE engine outputs.

For each trajectory, extracts per-iteration signals:
- iteration_id, summary_length, task_results_count, message_count,
  tool_errors, execution_errors, continue_simulation

Output:
- results/marble/native_signal_audit/iteration_signal.csv
- docs/audit/marble_iteration_signal_analysis.md
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DIFFICULTY_CSV = BASE_DIR / "results" / "marble" / "difficulty_profile" / "difficulty_episode.csv"
WORKSPACE_ROOTS = [
    Path("/tmp/smtr_traj_rogjzuzb"),
    Path("/tmp/smtr_traj_d53j9wnm"),
]
OUTPUT_DIR = BASE_DIR / "results" / "marble" / "native_signal_audit"


def find_marble_output(episode_id: str) -> Path | None:
    for ws in WORKSPACE_ROOTS:
        candidate = ws / episode_id / "marble_output.jsonl"
        if candidate.exists():
            return candidate
    return None


def parse_raw_output(path: Path) -> dict[str, Any]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        return {}
    output: dict[str, Any] = {}
    for rec in records:
        output.update(rec)
    output["_records"] = records
    return output


def count_error_patterns(text: str) -> int:
    """Count error-related keywords in text."""
    patterns = [
        r"(?i)error",
        r"(?i)fail",
        r"(?i)exception",
        r"(?i)timeout",
        r"(?i)not found",
        r"(?i)invalid",
    ]
    count = 0
    for p in patterns:
        count += len(re.findall(p, text))
    return count


def extract_iteration_signals(
    raw: dict[str, Any],
    *,
    scenario: str,
    task_id: str,
    seed: int,
    episode_id: str,
) -> list[dict[str, Any]]:
    """Extract per-iteration signals from raw output."""
    iterations = raw.get("iterations", [])
    if not isinstance(iterations, list) or not iterations:
        return []

    rows = []
    for it in iterations:
        if not isinstance(it, dict):
            continue

        iteration_id = it.get("iteration", 0)
        summary = str(it.get("summary", ""))
        task_results = it.get("task_results", [])
        communications = it.get("communications", [])
        continue_sim = it.get("continue_simulation", None)

        # Count task results
        n_task_results = len(task_results) if isinstance(task_results, list) else 0
        task_results_text = " ".join(str(tr) for tr in task_results) if isinstance(task_results, list) else ""

        # Count messages
        n_messages = len(communications) if isinstance(communications, list) else 0
        comm_text = " ".join(str(c) for c in communications) if isinstance(communications, list) else ""

        # Error counting
        all_text = summary + task_results_text + comm_text
        n_errors = count_error_patterns(all_text)

        # Summary analysis
        summary_len = len(summary)

        # Check if summary changed between iterations (task progress indicator)
        # This is tracked via summary length trajectory

        rows.append({
            "scenario": scenario,
            "task_id": task_id,
            "seed": seed,
            "episode_id": episode_id,
            "iteration": iteration_id,
            "summary_length": summary_len,
            "n_task_results": n_task_results,
            "n_messages": n_messages,
            "n_errors": n_errors,
            "continue_simulation": continue_sim,
        })

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read episode list
    episodes = []
    with open(DIFFICULTY_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(row)

    all_iter_rows = []
    score_trajectories: dict[str, list[int]] = {}  # episode_id -> list of summary_lengths

    for ep in episodes:
        domain = ep["domain"]
        task_id = ep["task_id"]
        seed = ep["seed"]
        episode_id = ep["episode_id"]

        out_path = find_marble_output(episode_id)
        if out_path is None:
            continue

        raw = parse_raw_output(out_path)
        iter_rows = extract_iteration_signals(
            raw,
            scenario=domain,
            task_id=task_id,
            seed=seed,
            episode_id=episode_id,
        )
        all_iter_rows.extend(iter_rows)

        # Build score trajectory (using summary_length as proxy)
        if iter_rows:
            key = f"{domain}/{task_id}/seed{seed}"
            score_trajectories[key] = [r["summary_length"] for r in iter_rows]

    # Write iteration signal CSV
    csv_path = OUTPUT_DIR / "iteration_signal.csv"
    if all_iter_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_iter_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_iter_rows)
        print(f"Written: {csv_path} ({len(all_iter_rows)} iteration rows)")
    else:
        print("WARNING: No iteration data found!")
        csv_path.write_text("scenario,task_id,seed,episode_id,iteration,summary_length,n_task_results,n_messages,n_errors,continue_simulation\n")

    # Analyze score trajectories
    n_episodes_with_progress = 0
    n_episodes_with_shrinking_summary = 0
    n_episodes_with_growing_summary = 0

    for key, traj in score_trajectories.items():
        if len(traj) > 1:
            n_episodes_with_progress += 1
            if traj[-1] > traj[0]:
                n_episodes_with_growing_summary += 1
            elif traj[-1] < traj[0]:
                n_episodes_with_shrinking_summary += 1

    # Generate report
    report_lines = [
        "# MARBLE Iteration Signal Analysis (Phase 3)",
        "",
        f"**Total episodes with iteration data**: {len(score_trajectories)}",
        f"**Total iteration records**: {len(all_iter_rows)}",
        "",
        "## Score Trajectory Summary",
        "",
        f"Episodes with multi-iteration progress: {n_episodes_with_progress}",
        f"Episodes with growing summaries: {n_episodes_with_growing_summary}",
        f"Episodes with shrinking summaries: {n_episodes_with_shrinking_summary}",
        "",
        "## Key Finding: Per-Iteration Evaluator Signals",
        "",
        "**All per-iteration evaluator signals are DISABLED in MARBLE graph mode.**",
        "",
        "The MARBLE engine graph_coordinate() loop has the following evaluators **commented out**:",
        "- `evaluate_communication()` → replaced with hardcoded `-1`",
        "- `evaluate_planning()` → replaced with hardcoded `-1`",
        "- `evaluate_kpi()` → replaced with hardcoded `-1`",
        "",
        "This means NO continuous per-iteration signal is available from the MARBLE evaluator.",
        "",
        "## Available Iteration-Level Signals",
        "",
        "| Signal | Type | Source | Usable? |",
        "|--------|------|--------|---------|",
        "| summary_length | Continuous (proxy) | Planner summary text length | ✅ Yes (proxy) |",
        "| n_task_results | Discrete | Number of agent task completions | ✅ Yes |",
        "| n_messages | Discrete | Inter-agent communication count | ✅ Yes |",
        "| n_errors | Discrete | Error keyword count in output | ✅ Yes (proxy) |",
        "| continue_simulation | Binary | Planner termination decision | ✅ Yes |",
        "| token_usage (global) | Continuous | Total token consumption | ✅ Yes |",
        "| planning_scores | Ordinal 1-5 | Per-iteration evaluator | ❌ No (hardcoded -1) |",
        "| communication_scores | Ordinal 1-5 | Per-iteration evaluator | ❌ No (hardcoded -1) |",
        "",
        "## Can We Define Delta(m,r) = P_expose(m,r) - P_withhold(r)?",
        "",
        "**With binary success (current)**: No. P is always {0, 1}, ceiling effect.",
        "",
        "**With iteration-level proxies**:",
        "- P = summary_length_delta (last_iter - first_iter): Possible, but noisy",
        "- P = token_usage: Possible, but not directly related to task quality",
        "- P = n_errors: Possible, inverse proxy for quality",
        "",
        "**With native final evaluator signals** (if available):",
        "- P = task_evaluation (research): {innovation, safety, feasibility} 1-5 → average",
        "- P = task_evaluation (minecraft): block_hit_rate * 5",
        "- P = task_evaluation (database): root_cause recall (0.0, 0.5, 1.0)",
        "",
        "**Recommendation**: Use native final evaluator signals where available (minecraft, database),",
        "and iteration-level proxies (summary_length trajectory) for domains without native evaluators.",
    ]

    report_path = BASE_DIR / "docs" / "audit" / "marble_iteration_signal_analysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
