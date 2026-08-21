"""Task 8: Memory budget experiment.

Caps the persistent memory bank at {10, 50, 100} slots and compares
full_memory (FIFO eviction of raw experience) against smtr_tci (the TCI
gate decides what is worth one of the scarce slots).

No adaptive threshold is introduced; the only parameter is the budget
itself. Metric: performance per memory slot = cumulative reward / budget.

Output: results/budget/memory_budget_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.run_lifelong import ALL_TOPICS, run_experiment

BUDGETS = [10, 50, 100]
EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["full_memory", "smtr_tci"]
CONTAMINATION_RATIO = 0.2


def generate(output_root: Path) -> None:
    for budget in BUDGETS:
        run_experiment(
            experiment=f"budget_{budget}",
            output_dir=output_root / f"budget_{budget}",
            episodes=EPISODES,
            seeds=SEEDS,
            methods=METHODS,
            contamination_ratio=CONTAMINATION_RATIO,
            change_episode=None,
            changed_topics=(),
            topics=ALL_TOPICS,
            topics_after_change=None,
            capacity=budget,
        )


def analyze(output_root: Path) -> None:
    results: list[dict] = []
    for budget in BUDGETS:
        perf_path = output_root / f"budget_{budget}" / "performance.csv"
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        with perf_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["method"], int(row["seed"]))].append(
                    float(row["cumulative_reward"])
                )
        for method in METHODS:
            cumulative = [vals[-1] for (m, _s), vals in grouped.items()
                          if m == method]
            mean_cum = float(np.mean(cumulative))
            results.append({
                "budget": budget,
                "method": method,
                "cumulative_reward_mean": mean_cum,
                "cumulative_reward_std": float(np.std(cumulative)),
                "performance_per_memory_slot": mean_cum / budget,
            })

    out_path = output_root / "memory_budget_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {out_path}")
    for row in results:
        print(
            f"  budget={row['budget']:<4} {row['method']:<12}"
            f" cum={row['cumulative_reward_mean']:.2f}"
            f"±{row['cumulative_reward_std']:.2f}"
            f" per_slot={row['performance_per_memory_slot']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/budget")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output)
    if not args.analyze_only:
        generate(output_root)
    analyze(output_root)


if __name__ == "__main__":
    main()
