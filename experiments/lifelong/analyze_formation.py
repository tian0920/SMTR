"""Task 4: Persistent Behavioral Knowledge Formation analysis.

Reads results/lifelong/formation/performance.csv and produces:
  - results/table_lifelong.csv    per-method summary metrics
  - figures/lifelong_curve.png    mean reward curve + cumulative reward

Metrics:
  1. average reward curve (seed-averaged, smoothed)
  2. cumulative reward (final, mean +/- std over seeds)
  3. late-stage performance (mean reward over last 20% episodes)
  4. memory efficiency = cumulative reward / stored memories
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

METHOD_ORDER = ["no_memory", "full_memory", "retrieval", "smtr_tci"]


def load_performance(path: Path) -> dict[tuple[str, int], list[dict]]:
    """Group performance rows by (method, seed), ordered by episode."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["episode"] = int(row["episode"])
            row["reward"] = float(row["reward"])
            row["cumulative_reward"] = float(row["cumulative_reward"])
            row["n_stored"] = int(row["n_stored"])
            grouped[(row["method"], int(row["seed"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["episode"])
    return grouped


def summarize(grouped: dict[tuple[str, int], list[dict]]) -> list[dict]:
    methods = sorted({m for m, _ in grouped}, key=METHOD_ORDER.index)
    table = []
    for method in methods:
        seeds = sorted(s for m, s in grouped if m == method)
        finals, lates, efficiencies, stored = [], [], [], []
        for seed in seeds:
            rows = grouped[(method, seed)]
            n = len(rows)
            late_start = n - max(1, n // 5)  # last 20%
            finals.append(rows[-1]["cumulative_reward"])
            lates.append(float(np.mean([r["reward"] for r in rows[late_start:]])))
            n_mem = max(1, rows[-1]["n_stored"])
            stored.append(rows[-1]["n_stored"])
            efficiencies.append(rows[-1]["cumulative_reward"] / n_mem)
        table.append({
            "method": method,
            "n_seeds": len(seeds),
            "cumulative_reward_mean": float(np.mean(finals)),
            "cumulative_reward_std": float(np.std(finals)),
            "late_stage_reward_mean": float(np.mean(lates)),
            "late_stage_reward_std": float(np.std(lates)),
            "memory_efficiency_mean": float(np.mean(efficiencies)),
            "stored_memories_mean": float(np.mean(stored)),
        })
    return table


def plot_curves(grouped: dict[tuple[str, int], list[dict]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({m for m, _ in grouped}, key=METHOD_ORDER.index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in methods:
        seeds = sorted(s for m, s in grouped if m == method)
        curves = np.array([[r["reward"] for r in grouped[(method, s)]] for s in seeds])
        cum_curves = np.array(
            [[r["cumulative_reward"] for r in grouped[(method, s)]] for s in seeds]
        )
        episodes = np.arange(curves.shape[1])
        # 10-episode moving average for the reward curve
        window = min(10, curves.shape[1])
        kernel = np.ones(window) / window
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=curves
        )
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        axes[0].plot(episodes, mean, label=method)
        axes[0].fill_between(episodes, mean - std, mean + std, alpha=0.12)
        cum_mean, cum_std = cum_curves.mean(axis=0), cum_curves.std(axis=0)
        axes[1].plot(episodes, cum_mean, label=method)
        axes[1].fill_between(episodes, cum_mean - cum_std, cum_mean + cum_std, alpha=0.12)

    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("reward (10-ep moving avg)")
    axes[0].set_title("Average reward curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("cumulative reward")
    axes[1].set_title("Cumulative reward")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/lifelong/formation/performance.csv")
    parser.add_argument("--table", default="results/table_lifelong.csv")
    parser.add_argument("--figure", default="figures/lifelong_curve.png")
    args = parser.parse_args()

    grouped = load_performance(Path(args.input))
    table = summarize(grouped)

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved table: {table_path}")
    for row in table:
        print(
            f"  {row['method']:<12} cum={row['cumulative_reward_mean']:6.2f}±"
            f"{row['cumulative_reward_std']:.2f}  late={row['late_stage_reward_mean']:.3f}"
            f"  eff={row['memory_efficiency_mean']:.3f}  stored={row['stored_memories_mean']:.0f}"
        )

    plot_curves(grouped, Path(args.figure))


if __name__ == "__main__":
    main()
