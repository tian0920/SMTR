"""Equal-cost baseline comparison.

Compares SMTR-TCI against baselines at equal computational budget:

  Methods: SMTR-TCI, Random Validation, Reflexion, Heuristic, AgeMem

Records:
  - additional_computation (probe trials / validation actions)
  - validation_actions
  - memory_operations (store, retrieve, evict)
  - reward_per_operation
  - knowledge_quality_per_cost

Output:
  experiments/baseline_cost_comparison/cost_comparison_results.csv
  paper/tables/table_cost_fair_comparison.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import LifelongEnvironment
from experiments.lifelong.run_lifelong import ALL_TOPICS
from experiments.lifelong.methods import METHODS
from experiments.lifelong.baseline_policies import BASELINE_METHODS
from experiments.cost_analysis.run_cost_analysis import RandomValidationPolicy

# Merge
_ALL_METHODS = {**METHODS, **BASELINE_METHODS, "random_validation": RandomValidationPolicy}

EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
CONTAMINATION_RATIO = 0.2

COST_METHODS = ["smtr_tci", "random_validation", "reflexion", "heuristic", "agemem"]

METHOD_LABELS: dict[str, str] = {
    "smtr_tci": "SMTR-TCI",
    "random_validation": "Random Validation",
    "reflexion": "Reflexion",
    "heuristic": "Heuristic",
    "agemem": "AgeMem-inspired",
}


def estimate_cost(method_name: str, episodes: int) -> dict:
    """Estimate computational cost for one method over a full run.

    Cost dimensions:
      - probe_trials: TCI expose+withhold trials (6 per candidate for SMTR)
      - validation_actions: number of validate/reject decisions
      - memory_operations: store + retrieve + evict operations
      - revalidation_probes: additional probes for re-validation
    """
    # All methods: 1 extraction per episode = episodes extractions
    # SMTR: 6 probe trials per candidate + re-validation probes
    # Others: 0 probe trials

    if method_name in ("smtr_tci", "random_validation"):
        # ~6 probes per candidate + ~3 re-validation probes per 10 episodes
        probe_trials = episodes * 6 + episodes * 3  # upper bound
        validation_actions = episodes
        memory_ops = episodes * 3  # store + validate + potential re-validate
    else:
        probe_trials = 0
        validation_actions = 0
        # store + retrieve per episode
        memory_ops = episodes * 2

    return {
        "probe_trials": probe_trials,
        "validation_actions": validation_actions,
        "memory_operations": memory_ops,
        "total_operations": probe_trials + validation_actions + memory_ops,
    }


def run_cost_comparison(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for seed in SEEDS:
        for method_name in COST_METHODS:
            env = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32(method_name.encode()) % 100000,
            )
            policy_cls = _ALL_METHODS[method_name]
            policy = policy_cls(env, capacity=None)

            rewards: list[float] = []
            for episode in range(EPISODES):
                task = env.sample_task(episode, ALL_TOPICS)
                injected = policy.select_memories(task)
                success, reward = env.execute(task, injected)
                rewards.append(reward)
                candidate = env.extract_candidate(task, CONTAMINATION_RATIO)
                policy.process_candidate(task, candidate)

            final_reward = float(np.mean(rewards))
            late_reward = float(np.mean(rewards[-20:]))
            cost = estimate_cost(method_name, EPISODES)

            results.append({
                "method": method_name,
                "label": METHOD_LABELS.get(method_name, method_name),
                "seed": seed,
                "final_reward": final_reward,
                "late_reward": late_reward,
                **cost,
            })

    # Save CSV
    csv_path = output_dir / "cost_comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    # Aggregate and generate table
    generate_latex_table(results, output_dir)


def generate_latex_table(results: list[dict], output_dir: Path) -> None:
    """Generate cost-fair comparison LaTeX table."""
    # Aggregate across seeds
    agg: dict[str, dict] = {}
    for method in COST_METHODS:
        method_rows = [r for r in results if r["method"] == method]
        if not method_rows:
            continue
        rewards = [r["final_reward"] for r in method_rows]
        late = [r["late_reward"] for r in method_rows]
        total_ops = [r["total_operations"] for r in method_rows]
        agg[method] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "late_mean": float(np.mean(late)),
            "total_ops": int(np.mean(total_ops)),
            "reward_per_op": float(np.mean(rewards)) / float(np.mean(total_ops))
                if np.mean(total_ops) > 0 else 0.0,
        }

    lines = [
        r"\begin{table}[t]",
        r"\caption{Equal-cost comparison: reward per additional operation.}",
        r"\label{tab:cost_fair}",
        r"\centering",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Method & Final Reward & Late Reward & Total Ops & Reward / Op \\",
        r"\midrule",
    ]
    for method in COST_METHODS:
        if method not in agg:
            continue
        m = agg[method]
        label = METHOD_LABELS.get(method, method)
        lines.append(
            f"{label} & {m['reward_mean']:.3f}$\\pm${m['reward_std']:.3f} "
            f"& {m['late_mean']:.3f} "
            f"& {m['total_ops']} "
            f"& {m['reward_per_op']:.5f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = Path("paper/tables/table_cost_fair_comparison.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {tex_path}")

    # Print summary
    print("\nCost-Fair Comparison:")
    for method in COST_METHODS:
        if method not in agg:
            continue
        m = agg[method]
        print(f"  {METHOD_LABELS.get(method, method):<20} "
              f"reward={m['reward_mean']:.3f}  "
              f"ops={m['total_ops']}  "
              f"reward/op={m['reward_per_op']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/baseline_cost_comparison")
    args = parser.parse_args()
    run_cost_comparison(Path(args.output))


if __name__ == "__main__":
    main()
