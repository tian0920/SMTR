"""P2-2: Receiver-Conditioned Knowledge Consolidation.

Tests the core SMTR property: the same memory has different causal
effects for different receivers. This is what makes TCI
receiver-conditioned rather than one-size-fits-all.

Setup:
  - Same memory candidate
  - 3 receivers (agent1, agent2, agent3)
  - TCI probe per receiver → τ(receiver)
  - Decision: validated for some receivers, rejected for others

Output:
  results/receiver_conditioned/receiver_knowledge_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import (
    HELPFUL_EFFECT,
    HARMFUL_EFFECT,
    LifelongEnvironment,
    StoredMemory,
    topic_affinity,
)

RECEIVERS = ["agent1", "agent2", "agent3"]
EPISODES = 50
SEEDS = [0, 1, 2, 3, 4]
TOPICS = tuple(range(10))


def run_receiver_conditioned(
    seed: int, episodes: int = EPISODES
) -> list[dict]:
    """For each candidate, probe TCI effect per receiver."""
    results: list[dict] = []

    # Writer generates candidates
    writer_env = LifelongEnvironment(
        seed=seed, method_seed=zlib.crc32(b"writer") % 100000
    )

    # Each receiver has its own environment (different RNG)
    receiver_envs = {
        recv: LifelongEnvironment(
            seed=seed,
            method_seed=zlib.crc32(recv.encode()) % 100000,
        )
        for recv in RECEIVERS
    }

    for episode in range(episodes):
        task = writer_env.sample_task(episode, TOPICS)
        candidate = writer_env.extract_candidate(task, contamination_ratio=0.2)

        # Probe TCI effect for each receiver
        tci_effects: dict[str, float] = {}
        decisions: dict[str, str] = {}
        for recv in RECEIVERS:
            env = receiver_envs[recv]
            delta = env.tci_probe_delta(candidate, episode=episode)
            tci_effects[recv] = delta
            decisions[recv] = "validated" if delta > 0 else "rejected"

        # Ground truth
        is_harmful = candidate.contamination == "false"
        true_effect = candidate.true_effect

        results.append({
            "seed": seed,
            "episode": episode,
            "memory_id": candidate.memory_id,
            "topic": candidate.topic,
            "contamination": candidate.contamination,
            "true_effect": true_effect,
            **{f"tau_{recv}": tci_effects[recv] for recv in RECEIVERS},
            **{f"decision_{recv}": decisions[recv] for recv in RECEIVERS},
            # Heterogeneity: variance of TCI effects across receivers
            "tau_variance": float(np.var(list(tci_effects.values()))),
            # Agreement: all receivers make same decision?
            "all_agree": len(set(decisions.values())) == 1,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", default="results/receiver_conditioned")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for seed in args.seeds:
        all_results.extend(run_receiver_conditioned(seed, args.episodes))

    # Write CSV
    fieldnames = list(all_results[0].keys())
    with (output_dir / "receiver_knowledge_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Analysis
    print(f"Total memories probed: {len(all_results)}")
    print(f"\n── TCI effect by receiver (mean ± std) ──")
    for recv in RECEIVERS:
        taus = [r[f"tau_{recv}"] for r in all_results]
        print(f"  {recv}: {np.mean(taus):+.3f} ± {np.std(taus):.3f}")

    # Receiver-conditional decisions
    disagree_count = sum(1 for r in all_results if not r["all_agree"])
    print(f"\n── Disagreement (receivers split decision): "
          f"{disagree_count}/{len(all_results)} "
          f"({disagree_count/len(all_results)*100:.1f}%) ──")

    # For harmful memories: how often does each receiver correctly reject?
    harmful = [r for r in all_results if r["contamination"] == "false"]
    print(f"\n── Harmful memories ({len(harmful)} total) ──")
    for recv in RECEIVERS:
        rejected = sum(1 for r in harmful if r[f"decision_{recv}"] == "rejected")
        print(f"  {recv} rejection rate: {rejected}/{len(harmful)} "
              f"({rejected/len(harmful)*100:.1f}%)" if harmful else "  N/A")

    # For helpful memories: how often does each receiver correctly validate?
    helpful = [r for r in all_results if r["contamination"] == "none"]
    print(f"\n── Helpful memories ({len(helpful)} total) ──")
    for recv in RECEIVERS:
        validated = sum(1 for r in helpful if r[f"decision_{recv}"] == "validated")
        print(f"  {recv} validation rate: {validated}/{len(helpful)} "
              f"({validated/len(helpful)*100:.1f}%)" if helpful else "  N/A")

    # Variance analysis
    variances = [r["tau_variance"] for r in all_results]
    print(f"\n── TCI effect variance across receivers ──")
    print(f"  Mean variance: {np.mean(variances):.4f}")
    print(f"  > 0 means receivers genuinely disagree (receiver-conditioned)")

    # Save summary
    summary = {
        "n_memories": len(all_results),
        "n_disagreements": disagree_count,
        "disagreement_rate": disagree_count / len(all_results),
        "mean_tau_variance": float(np.mean(variances)),
        "per_receiver": {
            recv: {
                "mean_tau": float(np.mean([r[f"tau_{recv}"] for r in all_results])),
                "harmful_rejection_rate": (
                    sum(1 for r in harmful if r[f"decision_{recv}"] == "rejected")
                    / len(harmful) if harmful else None
                ),
                "helpful_validation_rate": (
                    sum(1 for r in helpful if r[f"decision_{recv}"] == "validated")
                    / len(helpful) if helpful else None
                ),
            }
            for recv in RECEIVERS
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
