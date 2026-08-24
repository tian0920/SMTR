"""Phase 2: Inspect raw MARBLE engine outputs for all 5 domains.

Reads the raw marble_output.jsonl files from the difficulty profiling run
and extracts the full inventory of native signals.

Output:
- results/marble/native_signal_audit/raw_signal_inventory.csv
- docs/audit/marble_native_output_examples.md
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DIFFICULTY_CSV = BASE_DIR / "results" / "marble" / "difficulty_profile" / "difficulty_episode.csv"
WORKSPACE_ROOTS = [
    Path("/tmp/smtr_traj_rogjzuzb"),  # difficulty profiling run
    Path("/tmp/smtr_traj_d53j9wnm"),  # earlier pilot runs
]
OUTPUT_DIR = BASE_DIR / "results" / "marble" / "native_signal_audit"


def find_marble_output(episode_id: str) -> Path | None:
    """Find the marble_output.jsonl for a given episode_id."""
    for ws in WORKSPACE_ROOTS:
        candidate = ws / episode_id / "marble_output.jsonl"
        if candidate.exists():
            return candidate
    return None


def parse_raw_output(path: Path) -> dict[str, Any]:
    """Parse JSONL into merged dict."""
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


def extract_raw_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract all native signal fields from raw output."""
    iterations = raw.get("iterations", [])
    n_iters = len(iterations) if isinstance(iterations, list) else 0

    # Per-iteration signals
    planning_scores = raw.get("planning_scores", [])
    communication_scores = raw.get("communication_scores", [])

    # Final signals
    task_evaluation = raw.get("task_evaluation")
    code_quality = raw.get("code_quality")
    token_usage = raw.get("token_usage", 0)
    agent_kpis = raw.get("agent_kpis", {})
    total_milestones = raw.get("total_milestones", 0)
    final_output = raw.get("final_output", "")

    # Iteration-level details
    iter_summaries = []
    iter_task_results_counts = []
    iter_communications_counts = []
    for it in (iterations if isinstance(iterations, list) else []):
        if not isinstance(it, dict):
            continue
        summ = it.get("summary", "")
        iter_summaries.append(len(str(summ)))
        tr = it.get("task_results", [])
        iter_task_results_counts.append(len(tr) if isinstance(tr, list) else 0)
        comm = it.get("communications", [])
        iter_communications_counts.append(len(comm) if isinstance(comm, list) else 0)

    # Determine raw evaluator fields present
    raw_eval_fields = []
    if task_evaluation is not None:
        raw_eval_fields.append("task_evaluation")
    if code_quality is not None:
        raw_eval_fields.append("code_quality")
    if planning_scores and any(s != -1 for s in planning_scores):
        raw_eval_fields.append("planning_scores(active)")
    else:
        raw_eval_fields.append("planning_scores(-1)")
    if communication_scores and any(s != -1 for s in communication_scores):
        raw_eval_fields.append("communication_scores(active)")
    else:
        raw_eval_fields.append("communication_scores(-1)")

    return {
        "n_iterations": n_iters,
        "planning_scores": planning_scores,
        "communication_scores": communication_scores,
        "task_evaluation": task_evaluation,
        "code_quality": code_quality,
        "token_usage": token_usage,
        "total_milestones": total_milestones,
        "agent_kpis": agent_kpis,
        "final_output_len": len(str(final_output)),
        "raw_eval_fields": ";".join(raw_eval_fields),
        "iter_summary_lengths": iter_summaries,
        "iter_task_results_counts": iter_task_results_counts,
        "iter_communications_counts": iter_communications_counts,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read difficulty CSV for episode list
    episodes = []
    with open(DIFFICULTY_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(row)

    rows = []
    examples_by_domain: dict[str, list[dict[str, Any]]] = {}

    for ep in episodes:
        domain = ep["domain"]
        task_id = ep["task_id"]
        seed = ep["seed"]
        episode_id = ep["episode_id"]
        team_reward = ep["team_reward"]

        out_path = find_marble_output(episode_id)
        if out_path is None:
            continue

        raw = parse_raw_output(out_path)
        fields = extract_raw_fields(raw)

        row = {
            "scenario": domain,
            "task_id": task_id,
            "seed": seed,
            "episode_id": episode_id,
            "team_success": ep["team_success"],
            "team_reward": team_reward,
            "n_iterations": fields["n_iterations"],
            "planning_scores": str(fields["planning_scores"]),
            "communication_scores": str(fields["communication_scores"]),
            "task_evaluation": str(fields["task_evaluation"])[:200] if fields["task_evaluation"] else "None",
            "code_quality": str(fields["code_quality"])[:200] if fields["code_quality"] else "None",
            "token_usage": fields["token_usage"],
            "total_milestones": fields["total_milestones"],
            "raw_eval_fields": fields["raw_eval_fields"],
            "final_output_len": fields["final_output_len"],
            "iter_summary_lengths": str(fields["iter_summary_lengths"]),
            "iter_task_results_counts": str(fields["iter_task_results_counts"]),
            "runtime": ep.get("engine_duration_seconds", ""),
        }
        rows.append(row)

        # Collect examples for report
        if domain not in examples_by_domain:
            examples_by_domain[domain] = []
        if len(examples_by_domain[domain]) < 2:
            examples_by_domain[domain].append({
                "task_id": task_id,
                "seed": seed,
                "fields": fields,
                "raw_keys": list(raw.keys()),
                "task_evaluation_raw": fields["task_evaluation"],
            })

    # Write CSV
    csv_path = OUTPUT_DIR / "raw_signal_inventory.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {csv_path} ({len(rows)} rows)")

    # Generate report with raw examples
    report_lines = [
        "# MARBLE Native Output Examples (Phase 2 Dynamic Audit)",
        "",
        f"**Total episodes inspected**: {len(rows)}",
        "",
    ]

    for domain, examples in sorted(examples_by_domain.items()):
        report_lines.append(f"## {domain}")
        report_lines.append("")
        for i, ex in enumerate(examples):
            report_lines.append(f"### Example {i+1}: task_id={ex['task_id']} seed={ex['seed']}")
            report_lines.append("")
            report_lines.append(f"- **Raw keys**: {ex['raw_keys']}")
            report_lines.append(f"- **n_iterations**: {ex['fields']['n_iterations']}")
            report_lines.append(f"- **token_usage**: {ex['fields']['token_usage']}")
            report_lines.append(f"- **planning_scores**: {ex['fields']['planning_scores']}")
            report_lines.append(f"- **communication_scores**: {ex['fields']['communication_scores']}")
            report_lines.append(f"- **task_evaluation**: {ex['task_evaluation_raw']}")
            report_lines.append(f"- **code_quality**: {ex['fields']['code_quality']}")
            report_lines.append(f"- **final_output_len**: {ex['fields']['final_output_len']}")
            report_lines.append(f"- **iter_summary_lengths**: {ex['fields']['iter_summary_lengths']}")
            report_lines.append(f"- **iter_task_results_counts**: {ex['fields']['iter_task_results_counts']}")
            report_lines.append("")
        report_lines.append("")

    report_path = BASE_DIR / "docs" / "audit" / "marble_native_output_examples.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
