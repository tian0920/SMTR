"""Analyze TCI activation from hard-task pilot.

Input:  results/marble/pilot_hard_tci/receiver_validation.json
Output: docs/audit/tci_activation_report.md + tci_activation.csv

Key metrics:
  - candidate memories
  - total interventions
  - positive / negative / zero delta count
  - validated memories
  - cross-episode reuse
  - Memory Opportunity Rate (MOR) = positive_delta / total_candidates
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
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyze(
    validations: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    n_total = len(validations)
    positive = [v for v in validations if v.get("delta", 0) > 0]
    negative = [v for v in validations if v.get("delta", 0) < 0]
    zero = [v for v in validations if v.get("delta", 0) == 0]
    validated = [v for v in validations if v.get("decision") == "validated"]

    # Unique candidates
    candidate_ids = set(v.get("memory_id", "") for v in validations)
    n_candidates = len(candidate_ids)

    # Cross-episode reuse from episode metrics
    cross_reuse = 0
    for ep in episodes:
        try:
            cross_reuse += int(ep.get("n_cross_episode_reuse", 0))
        except (ValueError, TypeError):
            pass

    # MOR: Memory Opportunity Rate
    mor = len(positive) / n_total if n_total > 0 else 0.0

    # Delta distribution
    deltas = [v.get("delta", 0) for v in validations]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    # Per-receiver breakdown
    per_receiver: dict[str, dict[str, int]] = {}
    for v in validations:
        rid = v.get("receiver_id", "?")
        if rid not in per_receiver:
            per_receiver[rid] = {"total": 0, "positive": 0, "negative": 0, "zero": 0, "validated": 0}
        per_receiver[rid]["total"] += 1
        d = v.get("delta", 0)
        if d > 0:
            per_receiver[rid]["positive"] += 1
        elif d < 0:
            per_receiver[rid]["negative"] += 1
        else:
            per_receiver[rid]["zero"] += 1
        if v.get("decision") == "validated":
            per_receiver[rid]["validated"] += 1

    return {
        "total_interventions": n_total,
        "n_candidates": n_candidates,
        "positive_delta": len(positive),
        "negative_delta": len(negative),
        "zero_delta": len(zero),
        "validated_memories": len(validated),
        "cross_episode_reuse": cross_reuse,
        "MOR": round(mor, 4),
        "mean_delta": round(mean_delta, 4),
        "per_receiver": per_receiver,
        "validations": validations,
    }


def write_csv(stats: dict[str, Any], output_path: Path) -> None:
    fieldnames = [
        "memory_id", "receiver_id", "task_id", "scenario", "seed",
        "expose_outcome", "withhold_outcome", "delta", "decision",
        "expose_real_engine", "withhold_real_engine",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in stats["validations"]:
            writer.writerow(v)


def write_report(stats: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# TCI Activation Report\n")

    lines.append(f"**Total interventions**: {stats['total_interventions']}\n")
    lines.append(f"**Unique candidates**: {stats['n_candidates']}\n")
    lines.append(f"**Positive delta**: {stats['positive_delta']}\n")
    lines.append(f"**Negative delta**: {stats['negative_delta']}\n")
    lines.append(f"**Zero delta**: {stats['zero_delta']}\n")
    lines.append(f"**Validated memories**: {stats['validated_memories']}\n")
    lines.append(f"**Cross-episode reuse**: {stats['cross_episode_reuse']}\n")
    lines.append(f"**Memory Opportunity Rate (MOR)**: {stats['MOR']:.4f}\n")
    lines.append(f"**Mean delta**: {stats['mean_delta']:.4f}\n")

    lines.append("\n## Per-Receiver Breakdown\n")
    lines.append("| Receiver | Total | Positive | Negative | Zero | Validated |")
    lines.append("|----------|-------|----------|----------|------|-----------|")
    for rid, v in sorted(stats["per_receiver"].items()):
        lines.append(
            f"| {rid} | {v['total']} | {v['positive']} | {v['negative']} "
            f"| {v['zero']} | {v['validated']} |"
        )

    lines.append("\n## Go/No-Go Signals\n")
    lines.append("| Criterion | Threshold | Actual | Pass? |")
    lines.append("|-----------|-----------|--------|-------|")
    lines.append(
        f"| MOR > 5% | > 0.05 | {stats['MOR']:.4f} "
        f"| {'PASS' if stats['MOR'] > 0.05 else 'FAIL'} |"
    )
    lines.append(
        f"| Validated > 0 | > 0 | {stats['validated_memories']} "
        f"| {'PASS' if stats['validated_memories'] > 0 else 'FAIL'} |"
    )
    lines.append(
        f"| Cross-episode reuse | ≥ 1 | {stats['cross_episode_reuse']} "
        f"| {'PASS' if stats['cross_episode_reuse'] >= 1 else 'FAIL'} |"
    )

    all_pass = (
        stats["MOR"] > 0.05
        and stats["validated_memories"] > 0
        and stats["cross_episode_reuse"] >= 1
    )
    lines.append(f"\n**Verdict**: {'ALL PASS → proceed to full run' if all_pass else 'SOME FAIL → investigate'}\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TCI activation")
    parser.add_argument(
        "--input-dir", type=str,
        default=str(_PROJECT_ROOT / "results" / "marble" / "pilot_hard_tci"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    val_path = input_dir / "receiver_validation.json"
    ep_path = input_dir / "episode_metrics.csv"

    if not val_path.exists():
        print(f"ERROR: {val_path} not found.")
        sys.exit(1)

    validations = load_validations(val_path)
    episodes = load_episodes(ep_path)
    stats = analyze(validations, episodes)

    # Write CSV
    csv_path = input_dir / "tci_activation.csv"
    write_csv(stats, csv_path)
    print(f"Written: {csv_path}")

    # Write report
    report_path = _PROJECT_ROOT / "docs" / "audit" / "tci_activation_report.md"
    write_report(stats, report_path)


if __name__ == "__main__":
    main()
