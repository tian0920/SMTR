"""MARBLE receiver-conditioned knowledge analysis (Phase 6).

Analyzes how memory utility varies across receivers.
Currently limited to agent1 (only receiver in existing paired records).

Metrics:
  1. Receiver disagreement: P(decision_i != decision_j) for same memory
  2. Receiver-conditioned utility: delta reward per receiver
  3. Harmful transfer: memory useful for A, harmful for B

Output: receiver_conditioned_results.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def load_paired_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def receiver_conditioned_analysis(records: list[dict]) -> list[dict]:
    """Analyze per-receiver treatment effects."""
    valid = [r for r in records if r.get("valid")]

    # Group by (task_id, candidate_memory_id)
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in valid:
        key = (r["task_id"], r["candidate_memory_id"])
        recv = r["receiver_agent_id"]
        groups[key][recv].append(r)

    results: list[dict] = []
    for (task_id, memory_id), recv_map in sorted(groups.items()):
        for recv, recv_records in sorted(recv_map.items()):
            taus: list[int] = []
            for r in recv_records:
                share_ok = bool(r.get("share", {}).get("team_success", False))
                withhold_ok = bool(r.get("withhold", {}).get("team_success", False))
                taus.append(int(share_ok) - int(withhold_ok))

            results.append({
                "task_id": task_id,
                "candidate_memory_id": memory_id,
                "receiver_agent_id": recv,
                "n_observations": len(taus),
                "mean_tau": float(np.mean(taus)),
                "std_tau": float(np.std(taus)),
                "positive_rate": sum(1 for t in taus if t > 0) / len(taus),
                "negative_rate": sum(1 for t in taus if t < 0) / len(taus),
                "neutral_rate": sum(1 for t in taus if t == 0) / len(taus),
            })

    return results


def compute_receiver_disagreement(records: list[dict]) -> dict[str, Any]:
    """Compute P(decision_i != decision_j) for same (task, memory) across receivers."""
    valid = [r for r in records if r.get("valid")]

    # Group by (task_id, candidate_memory_id) -> {receiver: [tau]}
    groups: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in valid:
        key = (r["task_id"], r["candidate_memory_id"])
        recv = r["receiver_agent_id"]
        share_ok = bool(r.get("share", {}).get("team_success", False))
        withhold_ok = bool(r.get("withhold", {}).get("team_success", False))
        tau = int(share_ok) - int(withhold_ok)
        groups[key][recv].append(tau)

    disagreements = 0
    comparisons = 0
    for key, recv_map in groups.items():
        receivers = sorted(recv_map.keys())
        if len(receivers) < 2:
            continue
        for i in range(len(receivers)):
            for j in range(i + 1, len(receivers)):
                tau_i = int(np.mean(recv_map[receivers[i]]))
                tau_j = int(np.mean(recv_map[receivers[j]]))
                if tau_i != tau_j:
                    disagreements += 1
                comparisons += 1

    return {
        "disagreement_rate": disagreements / max(comparisons, 1),
        "n_comparisons": comparisons,
        "n_disagreements": disagreements,
    }


def main() -> None:
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    results = receiver_conditioned_analysis(records)

    # Write CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "receiver_conditioned_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Disagreement analysis
    disagreement = compute_receiver_disagreement(records)

    # Summary
    print("=== Receiver-Conditioned Analysis ===")
    print(f"Total records: {len(results)}")

    receivers = sorted(set(r["receiver_agent_id"] for r in results))
    print(f"Receivers: {receivers}")

    for recv in receivers:
        recv_results = [r for r in results if r["receiver_agent_id"] == recv]
        mean_tau = float(np.mean([r["mean_tau"] for r in recv_results]))
        pos_rate = float(np.mean([r["positive_rate"] for r in recv_results]))
        neg_rate = float(np.mean([r["negative_rate"] for r in recv_results]))
        print(f"  {recv}: n={len(recv_results)}, mean_tau={mean_tau:.4f}, "
              f"positive_rate={pos_rate:.4f}, negative_rate={neg_rate:.4f}")

    print(f"\nDisagreement: {disagreement}")

    # Write summary JSON
    summary = {
        "receivers": receivers,
        "per_receiver": {
            recv: {
                "n_records": len([r for r in results if r["receiver_agent_id"] == recv]),
                "mean_tau": float(np.mean([r["mean_tau"] for r in results if r["receiver_agent_id"] == recv])),
                "positive_rate": float(np.mean([r["positive_rate"] for r in results if r["receiver_agent_id"] == recv])),
                "negative_rate": float(np.mean([r["negative_rate"] for r in results if r["receiver_agent_id"] == recv])),
            }
            for recv in receivers
        },
        "disagreement": disagreement,
    }
    summary_path = output_dir / "receiver_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {csv_path}, {summary_path}")


if __name__ == "__main__":
    main()
