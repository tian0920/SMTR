"""Task 7: Multi-agent knowledge propagation.

Setup: 1 writer agent + 3 receiver agents (agent1..agent3, mirroring the
MARBLE 3-receiver configuration whose heterogeneity was confirmed on the
live engine: receiver_effect_variance=0.333).

Flow:
  writer:    collects experience -> extracts candidates -> TCI validation
             -> shared persistent memory bank
  receivers: retrieve from the shared bank and execute their own tasks

Sharing modes compared:
  naive: all extracted memories are shared
  smtr:  only TCI-validated memories are shared

Metrics:
  team reward                     mean receiver reward per episode
  knowledge propagation accuracy  fraction of injected memories that are
                                  genuine knowledge (contamination == none)
  contamination propagation       fraction of injected memories that are
                                  contaminated

Output: results/multi_agent/multi_agent_knowledge_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import (
    LifelongEnvironment,
    StoredMemory,
    topic_affinity,
)

EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
RECEIVERS = ["agent1", "agent2", "agent3"]
CONTAMINATION_RATIO = 0.2
TOPICS = tuple(range(10))


def run_one(seed: int, sharing_mode: str) -> list[dict]:
    writer_env = LifelongEnvironment(
        seed=seed, method_seed=zlib.crc32(b"writer") % 100000
    )
    receiver_envs = {
        recv: LifelongEnvironment(
            seed=seed,
            method_seed=zlib.crc32(recv.encode()) % 100000,
        )
        for recv in RECEIVERS
    }
    # shared bank: memory -> validated flag decided by the writer's TCI gate
    shared: list[tuple[StoredMemory, bool]] = []
    rows: list[dict] = []
    for episode in range(EPISODES):
        task = writer_env.sample_task(episode, TOPICS)
        candidate = writer_env.extract_candidate(task, CONTAMINATION_RATIO)
        delta = writer_env.tci_probe_delta(candidate, episode=episode)
        validated = delta > 0
        shared.append((candidate, validated))

        for recv in RECEIVERS:
            env = receiver_envs[recv]
            r_task = env.sample_task(episode, TOPICS)
            if sharing_mode == "naive":
                pool = [m for m, _ in shared[:-1]]  # exclude just-written
            else:
                pool = [m for m, v in shared[:-1] if v]
            injected = [
                m for m in pool if topic_affinity(m.topic, r_task.topic) > 0
            ]
            success, reward = env.execute(r_task, injected)
            n_contaminated = sum(
                1 for m in injected if m.contamination != "none"
            )
            rows.append({
                "episode": episode,
                "seed": seed,
                "sharing_mode": sharing_mode,
                "receiver": recv,
                "topic": r_task.topic,
                "reward": reward,
                "n_injected": len(injected),
                "n_contaminated_injected": n_contaminated,
                "bank_size": len(shared[:-1]),
                "bank_validated": sum(1 for _, v in shared[:-1] if v),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/multi_agent")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for seed in SEEDS:
        for mode in ("naive", "smtr"):
            all_rows.extend(run_one(seed, mode))

    with (output_dir / "multi_agent_trajectory.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    summary: list[dict] = []
    for mode in ("naive", "smtr"):
        mode_rows = [r for r in all_rows if r["sharing_mode"] == mode]
        team_rewards: list[float] = []
        for seed in SEEDS:
            seed_rows = [r for r in mode_rows if r["seed"] == seed]
            team_rewards.append(
                float(np.mean([r["reward"] for r in seed_rows]))
            )
        injected = [r for r in mode_rows if r["n_injected"] > 0]
        total_injected = sum(r["n_injected"] for r in mode_rows)
        total_contaminated = sum(r["n_contaminated_injected"] for r in mode_rows)
        contamination_propagation = (
            total_contaminated / total_injected if total_injected else 0.0
        )
        summary.append({
            "sharing_mode": mode,
            "team_reward_mean": float(np.mean(team_rewards)),
            "team_reward_std": float(np.std(team_rewards)),
            "knowledge_propagation_accuracy": 1.0 - contamination_propagation,
            "contamination_propagation": contamination_propagation,
            "mean_bank_size": float(np.mean([r["bank_size"] for r in mode_rows])),
        })

    out_path = output_dir / "multi_agent_knowledge_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved: {out_path}")
    for row in summary:
        print(
            f"  {row['sharing_mode']:<6} team={row['team_reward_mean']:.3f}"
            f"±{row['team_reward_std']:.3f}"
            f" accuracy={row['knowledge_propagation_accuracy']:.3f}"
            f" contamination={row['contamination_propagation']:.3f}"
        )


if __name__ == "__main__":
    main()
