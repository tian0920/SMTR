"""MARBLE Receiver=3 Full-Scale Main Experiment.

Runs the receiver-conditioned TCI experiment across all available
paired records with 5 seeds and all 7 methods.

Output: results/marble/receiver3/main/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.marble_receiver3.pilot.run_pilot import (
    ALL_POLICIES,
    RECEIVER_IDS,
    load_paired_records,
    run_pilot as run_main_experiment,
)


def main() -> None:
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    # Full config (from marble_receiver3_main.yaml scale.main)
    methods = [
        "no_memory", "full_memory", "retrieval",
        "smtr_uniform", "smtr_receiver",
    ]
    seeds = [0, 1, 2, 3, 4]
    n_tasks = None  # all tasks

    print("=== MARBLE Receiver=3 Main Experiment ===")
    print(f"  methods: {methods}")
    print(f"  seeds: {seeds}")
    print(f"  n_tasks: all")
    print(f"  receivers: {RECEIVER_IDS}")
    print(f"  total paired records: {len(records)}")
    print()

    episode_rows, detail_rows = run_main_experiment(
        paired_records=records,
        methods=methods,
        seeds=seeds,
        n_tasks=n_tasks,
    )

    # Write results
    output_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)

    if episode_rows:
        import csv
        csv_path = output_dir / "main_episodes.csv"
        fieldnames = list(episode_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"Written: {csv_path} ({len(episode_rows)} rows)")

    if detail_rows:
        import csv
        detail_path = output_dir / "main_receiver_details.csv"
        detail_fields = list(detail_rows[0].keys())
        with detail_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields)
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"Written: {detail_path} ({len(detail_rows)} rows)")

    # Summary JSON
    import numpy as np
    summary: list[dict] = []
    for method_name in methods:
        m_rows = [r for r in episode_rows if r["method"] == method_name]
        if not m_rows:
            summary.append({"method": method_name, "n_episodes": 0})
            continue
        rewards = [r["team_reward"] for r in m_rows]
        r1_rewards = [r["receiver_1_reward"] for r in m_rows]
        r2_rewards = [r["receiver_2_reward"] for r in m_rows]
        r3_rewards = [r["receiver_3_reward"] for r in m_rows]
        disagreements = [r["receiver_disagreement_std"] for r in m_rows]
        # Negative transfer count
        total_neg = sum(
            r["receiver_1_negative"] + r["receiver_2_negative"] + r["receiver_3_negative"]
            for r in m_rows
        )
        total_pos = sum(
            r["receiver_1_positive"] + r["receiver_2_positive"] + r["receiver_3_positive"]
            for r in m_rows
        )
        summary.append({
            "method": method_name,
            "n_episodes": len(m_rows),
            "mean_team_reward": float(np.mean(rewards)),
            "std_team_reward": float(np.std(rewards)),
            "mean_receiver_1_reward": float(np.mean(r1_rewards)),
            "mean_receiver_2_reward": float(np.mean(r2_rewards)),
            "mean_receiver_3_reward": float(np.mean(r3_rewards)),
            "mean_disagreement_std": float(np.mean(disagreements)),
            "total_positive_injected": int(total_pos),
            "total_negative_injected": int(total_neg),
        })

    summary_path = output_dir / "main_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Written: {summary_path}")

    # Print summary
    import numpy as np
    print()
    print("=" * 90)
    print(f"{'Method':<18} {'Eps':>5} {'Team':>8} {'R1':>8} {'R2':>8} {'R3':>8} {'Pos':>5} {'Neg':>5} {'Disagree':>10}")
    print("-" * 90)
    for s in summary:
        if s["n_episodes"] == 0:
            print(f"{s['method']:<18} {'0':>5}")
            continue
        print(
            f"{s['method']:<18} "
            f"{s['n_episodes']:>5} "
            f"{s['mean_team_reward']:>8.4f} "
            f"{s['mean_receiver_1_reward']:>8.4f} "
            f"{s['mean_receiver_2_reward']:>8.4f} "
            f"{s['mean_receiver_3_reward']:>8.4f} "
            f"{s['total_positive_injected']:>5} "
            f"{s['total_negative_injected']:>5} "
            f"{s['mean_disagreement_std']:>10.4f}"
        )
    print("=" * 90)

    # Improvement over best baseline
    no_mem = next((s for s in summary if s["method"] == "no_memory"), None)
    smtr_r = next((s for s in summary if s["method"] == "smtr_receiver"), None)
    if no_mem and smtr_r:
        base = no_mem["mean_team_reward"]
        impr = (smtr_r["mean_team_reward"] - base) / max(abs(base), 1e-9) * 100
        print(f"\nSMTR-receiver improvement over no_memory: {impr:+.1f}%")


if __name__ == "__main__":
    main()
