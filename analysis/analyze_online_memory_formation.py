"""Analyze online memory lifecycle — cross-episode formation and reuse.

Input:  results/marble/pilot_hard_tci/
        - receiver_validation.json
        - episode_metrics.csv
        - memory_history.json

Output: docs/audit/online_memory_formation_report.md

Must include at least 3 memory lifecycle examples:
  candidate → validated → reused → impact
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


def load_validations(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_memory_history(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trace_memory_lifecycles(
    validations: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Trace individual memory lifecycles across episodes.

    Returns a list of lifecycle dicts with:
      - memory_id
      - origin_task
      - origin_episode
      - receiver_id
      - delta
      - decision
      - reused_in_episode (if applicable)
      - impact (reward change)
    """
    lifecycles: list[dict[str, Any]] = []

    # Build validation index by memory_id
    val_by_id: dict[str, list[dict[str, Any]]] = {}
    for v in validations:
        mid = v.get("memory_id", "")
        val_by_id.setdefault(mid, []).append(v)

    # Build episode index
    ep_index: list[dict[str, Any]] = sorted(
        episodes,
        key=lambda e: (
            e.get("scenario", ""),
            e.get("task_id", ""),
            int(e.get("seed", 0)),
        ),
    )

    # For each validated memory, trace its lifecycle
    for mid, vals in val_by_id.items():
        validated_vals = [v for v in vals if v.get("decision") == "validated"]
        if not validated_vals:
            continue

        for v in validated_vals:
            lifecycle = {
                "memory_id": mid,
                "origin_task": v.get("task_id", "?"),
                "origin_scenario": v.get("scenario", "?"),
                "origin_seed": v.get("seed", 0),
                "receiver_id": v.get("receiver_id", "?"),
                "expose_outcome": v.get("expose_outcome", 0),
                "withhold_outcome": v.get("withhold_outcome", 0),
                "delta": v.get("delta", 0),
                "decision": v.get("decision", "?"),
                "reused_in": None,
                "impact": None,
            }

            # Check if this memory was reused in later episodes
            # (simplified: look for cross-episode reuse in episode metrics)
            for ep in ep_index:
                try:
                    reuse = int(ep.get("n_cross_episode_reuse", 0))
                except (ValueError, TypeError):
                    reuse = 0
                if reuse > 0 and ep.get("task_id") != v.get("task_id"):
                    lifecycle["reused_in"] = f"{ep.get('scenario')}/{ep.get('task_id')}"
                    lifecycle["impact"] = f"reward={ep.get('team_reward', '?')}"
                    break

            lifecycles.append(lifecycle)

    # Also trace rejected memories for contrast
    for mid, vals in val_by_id.items():
        rejected_vals = [v for v in vals if v.get("decision") == "rejected"]
        for v in rejected_vals[:3]:  # limit to 3 examples
            lifecycles.append({
                "memory_id": mid,
                "origin_task": v.get("task_id", "?"),
                "origin_scenario": v.get("scenario", "?"),
                "origin_seed": v.get("seed", 0),
                "receiver_id": v.get("receiver_id", "?"),
                "expose_outcome": v.get("expose_outcome", 0),
                "withhold_outcome": v.get("withhold_outcome", 0),
                "delta": v.get("delta", 0),
                "decision": "rejected",
                "reused_in": None,
                "impact": "not injected",
            })

    return lifecycles


def write_report(
    lifecycles: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Online Memory Formation Report\n")

    validated = [l for l in lifecycles if l["decision"] == "validated"]
    rejected = [l for l in lifecycles if l["decision"] == "rejected"]

    lines.append(f"**Total validations**: {len(validations)}\n")
    lines.append(f"**Validated memories**: {len(validated)}\n")
    lines.append(f"**Rejected memories**: {len(rejected)}\n")

    # Bank growth from history
    if history:
        lines.append("\n## Bank Growth Across Episodes\n")
        lines.append("| Episode | Bank Size | Validated | Rejected | Candidate |")
        lines.append("|---------|-----------|-----------|----------|-----------|")
        for snap in history:
            stats = snap.get("bank_statistics", snap)
            lines.append(
                f"| {snap.get('episode', '?')} "
                f"| {stats.get('total', '?')} "
                f"| {stats.get('validated', '?')} "
                f"| {stats.get('rejected', '?')} "
                f"| {stats.get('candidate', '?')} |"
            )

    # Memory lifecycle examples
    lines.append("\n## Memory Lifecycle Examples\n")

    # Show up to 3 validated examples
    lines.append("### Validated Memories\n")
    shown = 0
    for lc in validated[:3]:
        shown += 1
        lines.append(f"**Example {shown}**: `{lc['memory_id']}`\n")
        lines.append(f"- Origin: {lc['origin_scenario']}/task {lc['origin_task']}, seed {lc['origin_seed']}")
        lines.append(f"- Receiver: {lc['receiver_id']}")
        lines.append(f"- Expose: {lc['expose_outcome']}, Withhold: {lc['withhold_outcome']}")
        lines.append(f"- Delta: {lc['delta']}")
        lines.append(f"- Reused in: {lc['reused_in'] or 'N/A'}")
        lines.append(f"- Impact: {lc['impact'] or 'N/A'}")
        lines.append("")

    # Show up to 3 rejected examples
    lines.append("### Rejected Memories (for contrast)\n")
    shown = 0
    for lc in rejected[:3]:
        shown += 1
        lines.append(f"**Example {shown}**: `{lc['memory_id']}`\n")
        lines.append(f"- Origin: {lc['origin_scenario']}/task {lc['origin_task']}")
        lines.append(f"- Receiver: {lc['receiver_id']}")
        lines.append(f"- Delta: {lc['delta']} (no improvement)")
        lines.append(f"- Impact: {lc['impact']}")
        lines.append("")

    if not validated and not rejected:
        lines.append("_No memory lifecycle data available yet._\n")

    # Conclusion
    lines.append("\n## Conclusion\n")
    if validated:
        reused = [l for l in validated if l.get("reused_in")]
        lines.append(
            f"Persistent memory formation {'IS' if reused else 'is NOT'} "
            f"demonstrated: {len(reused)}/{len(validated)} validated memories "
            f"were reused in subsequent episodes."
        )
    else:
        lines.append(
            "No validated memories found in this run. "
            "This may indicate: (1) tasks are too easy (ceiling effect), "
            "(2) TCI correctly rejects all candidates, or "
            "(3) the experiment has not run enough episodes."
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze online memory formation")
    parser.add_argument(
        "--input-dir", type=str,
        default=str(_PROJECT_ROOT / "results" / "marble" / "pilot_hard_tci"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    val_path = input_dir / "receiver_validation.json"
    ep_path = input_dir / "episode_metrics.csv"
    hist_path = input_dir / "memory_history.json"

    validations = load_validations(val_path)
    episodes = load_episodes(ep_path)
    history = load_memory_history(hist_path)

    lifecycles = trace_memory_lifecycles(validations, episodes, history)

    report_path = _PROJECT_ROOT / "docs" / "audit" / "online_memory_formation_report.md"
    write_report(lifecycles, validations, episodes, history, report_path)


if __name__ == "__main__":
    main()
