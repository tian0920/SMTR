"""Validation coverage scaling experiment.

Tests how performance scales with validation budget:
  coverage = fraction of candidate memories that are validated

Settings:
  coverage ∈ {0.1, 0.25, 0.5, 1.0}

No new thresholds introduced — only controls how many candidates
go through the TCI gate. Unvalidated candidates are rejected by default.
"""

import argparse
import csv
import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from smtr.analysis.cost_tracker import TCICostTracker
from smtr.memory.consolidation import MemoryAdmissionController, AdmissionDecision
from smtr.memory.persistent_memory import PersistentMemoryBank
from experiments.lifelong.lifelong_env import (
    LifelongEnvironment,
    StoredMemory,
    TaskSample,
    topic_affinity,
)
from experiments.lifelong.methods import SMTRTCIPolicy
from experiments.lifelong.run_lifelong import run_episode_sequence, ALL_TOPICS

from datetime import UTC, datetime

COVERAGE_LEVELS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 100
CONTAMINATION_RATIO = 0.2


class CoverageSMTRTCIPolicy(SMTRTCIPolicy):
    """SMTR-TCI with limited validation coverage.

    Only a fraction of candidate memories go through TCI validation.
    Unvalidated candidates are rejected by default (conservative).
    """

    def __init__(self, env: LifelongEnvironment, coverage: float = 1.0,
                 cost_tracker: TCICostTracker | None = None) -> None:
        super().__init__(env)
        self._coverage = coverage
        if cost_tracker is not None:
            self.admission = MemoryAdmissionController(self.bank, cost_tracker=cost_tracker)

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        self._store(candidate)

        # Coverage gate: only validate a fraction of candidates
        if self._env._rng.random() < self._coverage:
            # Normal TCI validation
            delta = self._env.tci_probe_delta(candidate, episode=task.episode)
            self.admission.admit(
                candidate.memory_id,
                reward_expose=delta,
                reward_withhold=0.0,
                episode_id=task.episode,
            )
        else:
            # Skip validation: reject by default (conservative)
            self.bank.reject_memory(candidate.memory_id, 0.0)

        self._revalidate_topic(task)
        self._enforce_capacity()


def run_budget_scaling(
    seeds: list[int] = SEEDS,
    episodes: int = EPISODES,
    output_dir: str = "results/cost_analysis",
) -> list[dict]:
    """Run validation coverage scaling experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for coverage in COVERAGE_LEVELS:
        print(f"\n=== Coverage {coverage:.0%} ===")

        for seed in seeds:
            cost_tracker = TCICostTracker(enabled=True)
            env = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32(f"coverage_{coverage}".encode()) % 100000,
            )
            policy = CoverageSMTRTCIPolicy(env, coverage=coverage, cost_tracker=cost_tracker)

            perf_rows: list[dict] = []
            hist_rows: list[dict] = []
            traj_rows: list[dict] = []

            run_episode_sequence(
                policy=policy, env=env, episodes=episodes,
                topics=ALL_TOPICS, topics_after_change=None,
                contamination_ratio=CONTAMINATION_RATIO, seed=seed,
                trajectory_rows=traj_rows, history_rows=hist_rows,
                performance_rows=perf_rows,
            )

            rewards = [float(r["reward"]) for r in perf_rows]
            final_reward = float(np.mean(rewards[-20:]))
            cost_summary = cost_tracker.summary()
            bank_stats = policy.bank.get_statistics()

            # Contamination in validated set
            validated_hist = [r for r in hist_rows if r["status"] == "validated"]
            contaminated_validated = [r for r in validated_hist
                                     if r.get("contamination", "none") != "none"]
            contam_rate = len(contaminated_validated) / len(validated_hist) if validated_hist else 0.0

            result = {
                "coverage": coverage,
                "seed": seed,
                "final_reward": round(final_reward, 4),
                "cumulative_reward": round(sum(rewards), 2),
                "intervention_count": cost_summary["intervention_count"],
                "validated_memories": bank_stats["validated"],
                "rejected_memories": bank_stats["rejected"],
                "contamination_rate": round(contam_rate, 4),
            }
            all_results.append(result)

            print(f"  seed={seed}  reward={final_reward:.3f}  "
                  f"interventions={cost_summary['intervention_count']}  "
                  f"validated={bank_stats['validated']}")

    # Save CSV
    csv_path = output_path / "cost_scaling_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    print("\n── Scaling Summary ──")
    for coverage in COVERAGE_LEVELS:
        rows = [r for r in all_results if r["coverage"] == coverage]
        reward = f"{np.mean([r['final_reward'] for r in rows]):.3f}±{np.std([r['final_reward'] for r in rows]):.3f}"
        interventions = f"{np.mean([r['intervention_count'] for r in rows]):.0f}"
        print(f"  coverage={coverage:.0%}  reward={reward}  interventions={interventions}")

    print(f"\nSaved: {csv_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--output", type=str, default="results/cost_analysis")
    args = parser.parse_args()

    run_budget_scaling(seeds=args.seeds, episodes=args.episodes, output_dir=args.output)
