"""Task 6: Cross-task knowledge transfer experiment.

Protocol:
  Training:  episodes 0..49 on task distribution A (topics 0-4)
  Testing:   episodes 50..99 on unseen distribution B (topics 5-9)

Memory consolidation during training uses the TCI gate (smtr_tci) or the
baselines. Transfer gain = performance(memory) - performance(no_memory)
on distribution B. The key question: does TCI select long-term knowledge
that generalizes, rather than task shortcuts?

Outputs: results/transfer/transfer_results.csv, figures/transfer_plot.png
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

from experiments.lifelong.run_lifelong import run_experiment

EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["no_memory", "full_memory", "retrieval", "smtr_tci"]
CONTAMINATION_RATIO = 0.2


def generate(output_dir: Path) -> None:
    run_experiment(
        experiment="transfer",
        output_dir=output_dir,
        episodes=EPISODES,
        seeds=SEEDS,
        methods=METHODS,
        contamination_ratio=CONTAMINATION_RATIO,
        change_episode=EPISODES // 2,   # distribution switch A -> B
        changed_topics=(),              # no environment drift, new topics only
        topics=tuple(range(5)),
        topics_after_change=tuple(range(5, 10)),
        capacity=None,
    )


def analyze(output_dir: Path) -> None:
    perf_path = output_dir / "performance.csv"
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with perf_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["method"], int(row["seed"]))].append(
                (int(row["episode"]), float(row["reward"]))
            )

    switch = EPISODES // 2
    results: list[dict] = []
    baseline_b: dict[int, float] = {}
    # no_memory B-phase reward per seed defines the no-memory reference
    for seed in SEEDS:
        curve = sorted(grouped[("no_memory", seed)])
        baseline_b[seed] = float(np.mean([r for e, r in curve if e >= switch]))

    for method in METHODS:
        b_rewards, gains = [], []
        for seed in SEEDS:
            curve = sorted(grouped[(method, seed)])
            b_mean = float(np.mean([r for e, r in curve if e >= switch]))
            b_rewards.append(b_mean)
            gains.append(b_mean - baseline_b[seed])
        results.append({
            "method": method,
            "distribution_b_reward_mean": float(np.mean(b_rewards)),
            "distribution_b_reward_std": float(np.std(b_rewards)),
            "transfer_gain_mean": float(np.mean(gains)),
            "transfer_gain_std": float(np.std(gains)),
        })

    out_path = output_dir / "transfer_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {out_path}")
    for row in results:
        print(
            f"  {row['method']:<12} B-reward={row['distribution_b_reward_mean']:.3f}"
            f"±{row['distribution_b_reward_std']:.3f}"
            f"  transfer_gain={row['transfer_gain_mean']:+.3f}"
        )
    _plot(output_dir, grouped, switch)


def _plot(output_dir: Path, grouped: dict, switch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in METHODS:
        seeds = sorted(s for (m, s) in grouped if m == method)
        curves = []
        for seed in seeds:
            curve = sorted(grouped[(method, seed)])
            curves.append([r for _, r in curve])
        arr = np.array(curves)
        window = 10
        kernel = np.ones(window) / window
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=arr
        )
        episodes = np.arange(smoothed.shape[1])
        ax.plot(episodes, smoothed.mean(axis=0), label=method)
    ax.axvline(switch, color="grey", linestyle="--", alpha=0.7,
               label="distribution A -> B")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward (10-ep moving avg)")
    ax.set_title("Cross-task transfer: train on A (topics 0-4), test on B (topics 5-9)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = Path("figures/transfer_plot.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/transfer")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output)
    if not args.analyze_only:
        generate(output_dir)
    analyze(output_dir)


if __name__ == "__main__":
    main()
