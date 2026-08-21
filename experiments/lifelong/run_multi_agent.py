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
LONG_EPISODES = 500
SEEDS = [0, 1, 2, 3, 4]
RECEIVERS = ["agent1", "agent2", "agent3"]
CONTAMINATION_RATIO = 0.2
TOPICS = tuple(range(10))


def run_one(seed: int, sharing_mode: str, episodes: int = EPISODES) -> list[dict]:
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
    for episode in range(episodes):
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
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help="number of episodes (use 500 for long horizon)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", default="results/multi_agent")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for seed in args.seeds:
        for mode in ("naive", "smtr"):
            all_rows.extend(run_one(seed, mode, episodes=args.episodes))

    with (output_dir / "multi_agent_trajectory.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    summary: list[dict] = []
    for mode in ("naive", "smtr"):
        mode_rows = [r for r in all_rows if r["sharing_mode"] == mode]
        team_rewards: list[float] = []
        early_rewards: list[float] = []  # first 20% episodes
        late_rewards: list[float] = []   # last 20% episodes
        ep_limit = args.episodes
        early_cutoff = int(ep_limit * 0.2)
        late_cutoff = int(ep_limit * 0.8)
        for seed in args.seeds:
            seed_rows = [r for r in mode_rows if r["seed"] == seed]
            team_rewards.append(
                float(np.mean([r["reward"] for r in seed_rows]))
            )
            early = [r for r in seed_rows if r["episode"] < early_cutoff]
            late = [r for r in seed_rows if r["episode"] >= late_cutoff]
            if early:
                early_rewards.append(float(np.mean([r["reward"] for r in early])))
            if late:
                late_rewards.append(float(np.mean([r["reward"] for r in late])))
        injected = [r for r in mode_rows if r["n_injected"] > 0]
        total_injected = sum(r["n_injected"] for r in mode_rows)
        total_contaminated = sum(r["n_contaminated_injected"] for r in mode_rows)
        contamination_propagation = (
            total_contaminated / total_injected if total_injected else 0.0
        )
        # Late-stage contamination
        late_injected = [r for r in mode_rows
                        if r["n_injected"] > 0 and r["episode"] >= late_cutoff]
        late_total = sum(r["n_injected"] for r in late_injected)
        late_contam = sum(r["n_contaminated_injected"] for r in late_injected)
        late_contamination = late_contam / late_total if late_total else 0.0
        summary.append({
            "sharing_mode": mode,
            "episodes": ep_limit,
            "team_reward_mean": float(np.mean(team_rewards)),
            "team_reward_std": float(np.std(team_rewards)),
            "early_reward": float(np.mean(early_rewards)) if early_rewards else 0.0,
            "late_reward": float(np.mean(late_rewards)) if late_rewards else 0.0,
            "knowledge_propagation_accuracy": 1.0 - contamination_propagation,
            "contamination_propagation": contamination_propagation,
            "late_contamination": late_contamination,
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
            f"  {row['sharing_mode']:<6} ep={row['episodes']:<4}"
            f" team={row['team_reward_mean']:.3f}"
            f"\u00b1{row['team_reward_std']:.3f}"
            f" early={row['early_reward']:.3f}"
            f" late={row['late_reward']:.3f}"
            f" contam={row['contamination_propagation']:.3f}"
            f" late_contam={row['late_contamination']:.3f}"
        )


if __name__ == "__main__":
    main()
