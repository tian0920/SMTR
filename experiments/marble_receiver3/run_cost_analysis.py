"""MARBLE Receiver=3 Cost Analysis (Phase 8).

Measures the validation cost of receiver-conditioned TCI vs uniform TCI.

Metrics:
  - expose rollouts (per receiver, per memory)
  - withhold rollouts (per receiver, per memory)
  - total receiver validations
  - reward / validation cost ratio
  - knowledge quality / cost ratio

Output: results/marble/receiver3/cost/
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    # Load main experiment details
    detail_path = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "main" / "main_receiver_details.csv"
    if not detail_path.exists():
        print("ERROR: run main experiment first")
        sys.exit(1)

    with detail_path.open() as f:
        details = list(csv.DictReader(f))

    print(f"Loaded {len(details)} detail rows")

    # Cost model: each validation requires 1 expose + 1 withhold rollout
    # For smtr_uniform: 1 validation per memory (aggregate across receivers)
    # For smtr_receiver: 3 validations per memory (1 per receiver)

    # Count unique (task, seed, memory) groups per method
    method_costs: dict[str, dict] = {}

    for method in ["smtr_uniform", "smtr_receiver"]:
        method_details = [d for d in details if d["method"] == method]

        # Count unique memories validated
        unique_memories = set()
        for d in method_details:
            unique_memories.add((d["task_id"], d["seed"], d["memory_id"]))

        n_memories = len(unique_memories)
        n_receivers = 3 if method == "smtr_receiver" else 1

        # Each validation = 1 expose + 1 withhold rollout
        n_validations = n_memories * n_receivers
        n_expose_rollouts = n_validations
        n_withhold_rollouts = n_validations
        total_rollouts = n_expose_rollouts + n_withhold_rollouts

        # Compute reward for this method
        episode_path = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "main" / "main_episodes.csv"
        with episode_path.open() as f:
            episodes = list(csv.DictReader(f))
        method_episodes = [e for e in episodes if e["method"] == method]
        mean_reward = float(np.mean([float(e["team_reward"]) for e in method_episodes])) if method_episodes else 0.0

        # Reward per validation cost
        reward_per_cost = mean_reward / max(n_validations, 1)

        # Knowledge quality: fraction of positive decisions that are actually positive
        positive_decisions = sum(1 for d in method_details if float(d["delta"]) > 0)
        quality = positive_decisions / max(len(method_details), 1)
        quality_per_cost = quality / max(n_validations, 1)

        method_costs[method] = {
            "n_unique_memories": n_memories,
            "n_receivers_validated": n_receivers,
            "n_validations": n_validations,
            "n_expose_rollouts": n_expose_rollouts,
            "n_withhold_rollouts": n_withhold_rollouts,
            "total_rollouts": total_rollouts,
            "mean_team_reward": round(mean_reward, 4),
            "reward_per_validation": round(reward_per_cost, 6),
            "knowledge_quality": round(quality, 4),
            "quality_per_validation": round(quality_per_cost, 6),
        }

    # Write cost CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "cost"
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_rows = [
        {
            "method": method,
            **metrics,
        }
        for method, metrics in method_costs.items()
    ]

    csv_path = output_dir / "cost_analysis.csv"
    fieldnames = list(cost_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cost_rows)
    print(f"Written: {csv_path}")

    # Summary JSON
    summary_path = output_dir / "cost_summary.json"
    summary_path.write_text(json.dumps(method_costs, indent=2))
    print(f"Written: {summary_path}")

    # Print comparison
    print()
    print("=" * 70)
    print("Receiver=3 Cost Analysis")
    print("=" * 70)
    print(f"{'Metric':<35} {'smtr_uniform':>15} {'smtr_receiver':>15}")
    print("-" * 70)

    u = method_costs["smtr_uniform"]
    r = method_costs["smtr_receiver"]

    for key in ["n_unique_memories", "n_receivers_validated", "n_validations",
                "n_expose_rollouts", "n_withhold_rollouts", "total_rollouts",
                "mean_team_reward", "reward_per_validation",
                "knowledge_quality", "quality_per_validation"]:
        u_val = u[key]
        r_val = r[key]
        if isinstance(u_val, float):
            print(f"{key:<35} {u_val:>15.6f} {r_val:>15.6f}")
        else:
            print(f"{key:<35} {u_val:>15} {r_val:>15}")

    cost_ratio = r["n_validations"] / max(u["n_validations"], 1)
    reward_gain = (r["mean_team_reward"] - u["mean_team_reward"]) / max(abs(u["mean_team_reward"]), 1e-9) * 100
    print()
    print(f"Cost multiplier (receiver/uniform): {cost_ratio:.1f}×")
    print(f"Reward gain: {reward_gain:+.1f}%")
    print(f"Cost-effectiveness: {reward_gain:+.1f}% reward for {cost_ratio:.1f}× cost")


if __name__ == "__main__":
    main()
