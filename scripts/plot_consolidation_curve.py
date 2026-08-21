"""P1-2: Knowledge Consolidation Curve (core paper figure).

Reads lifelong experiment outputs and generates a 3-panel figure showing
knowledge formation dynamics over episodes:

  Subplot 1: validated memory count (accumulation curve)
  Subplot 2: task performance (10-episode rolling mean)
  Subplot 3: contamination ratio in active memory

Methods compared: Full Memory, Retrieval, SMTR-TCI.

Input:
  results/lifelong/formation/performance.csv
  results/lifelong/formation/memory_history.jsonl

Output:
  figures/consolidation_curve.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def load_data(results_dir: Path) -> tuple[dict, dict]:
    """Load performance.csv and memory_history.jsonl."""
    import csv

    perf: dict[str, list[dict]] = {}
    with (results_dir / "performance.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            perf.setdefault(method, []).append(row)

    hist: dict[str, list[dict]] = {}
    with (results_dir / "memory_history.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            method = row["method"]
            hist.setdefault(method, []).append(row)

    return perf, hist


def compute_curves(
    perf: dict[str, list[dict]],
    hist: dict[str, list[dict]],
    episodes: int,
    seeds: list[int],
) -> dict[str, dict]:
    """Compute per-method curves averaged across seeds."""
    methods = ["full_memory", "retrieval", "smtr_tci"]
    curves: dict[str, dict] = {}

    for method in methods:
        if method not in perf:
            continue
        method_perf = perf[method]
        method_hist = hist.get(method, [])

        # Per-episode arrays (mean across seeds)
        rewards = np.zeros((len(seeds), episodes))
        validated_count = np.zeros((len(seeds), episodes))
        total_count = np.zeros((len(seeds), episodes))
        contaminated_active = np.zeros((len(seeds), episodes))

        for si, seed in enumerate(seeds):
            seed_perf = [r for r in method_perf if int(r["seed"]) == seed]
            seed_hist = [r for r in method_hist if r["seed"] == seed]

            for ep in range(min(episodes, len(seed_perf))):
                rewards[si, ep] = float(seed_perf[ep]["reward"])
                validated_count[si, ep] = int(seed_perf[ep].get("n_validated", 0))
                total_count[si, ep] = int(seed_perf[ep].get("n_stored", 0))

            # Contamination ratio per episode (cumulative)
            for ep in range(episodes):
                active = [r for r in seed_hist
                         if r["episode"] <= ep
                         and r["status"] in ("validated", "candidate")]
                contaminated = [r for r in active if r.get("contamination", "none") != "none"]
                contaminated_active[si, ep] = (
                    len(contaminated) / len(active) if active else 0.0
                )

        # Rolling mean (window=10)
        window = 10
        rolling_rewards = np.zeros_like(rewards)
        for si in range(len(seeds)):
            for ep in range(episodes):
                start = max(0, ep - window + 1)
                rolling_rewards[si, ep] = np.mean(rewards[si, start:ep + 1])

        curves[method] = {
            "reward_mean": rolling_rewards.mean(axis=0),
            "reward_std": rolling_rewards.std(axis=0),
            "validated_mean": validated_count.mean(axis=0),
            "validated_std": validated_count.std(axis=0),
            "contamination_mean": contaminated_active.mean(axis=0),
            "contamination_std": contaminated_active.std(axis=0),
        }

    return curves


def plot_consolidation(
    curves: dict[str, dict],
    episodes: int,
    output_dir: Path,
) -> None:
    """Generate the 3-panel consolidation curve figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    method_labels = {
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "smtr_tci": "SMTR-TCI",
    }
    colors = {
        "full_memory": "#e74c3c",
        "retrieval": "#f39c12",
        "smtr_tci": "#2ecc71",
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Knowledge Consolidation Curve (P1-2)", fontsize=14, fontweight="bold")
    x = np.arange(episodes)

    # Panel 1: Validated memory count
    ax = axes[0]
    for method, label in method_labels.items():
        if method not in curves:
            continue
        c = curves[method]
        ax.plot(x, c["validated_mean"], label=label, color=colors[method], linewidth=2)
        ax.fill_between(x,
                        c["validated_mean"] - c["validated_std"],
                        c["validated_mean"] + c["validated_std"],
                        alpha=0.15, color=colors[method])
    ax.set_ylabel("Validated Memory Count")
    ax.legend(loc="upper left")
    ax.set_title("Accumulated Validated Knowledge")

    # Panel 2: Task performance (rolling mean)
    ax = axes[1]
    for method, label in method_labels.items():
        if method not in curves:
            continue
        c = curves[method]
        ax.plot(x, c["reward_mean"], label=label, color=colors[method], linewidth=2)
        ax.fill_between(x,
                        c["reward_mean"] - c["reward_std"],
                        c["reward_mean"] + c["reward_std"],
                        alpha=0.15, color=colors[method])
    ax.set_ylabel("Task Performance (10-ep rolling mean)")
    ax.legend(loc="lower right")
    ax.set_title("Performance Over Time")
    ax.set_ylim(0, 1.05)

    # Panel 3: Contamination ratio
    ax = axes[2]
    for method, label in method_labels.items():
        if method not in curves:
            continue
        c = curves[method]
        ax.plot(x, c["contamination_mean"], label=label, color=colors[method], linewidth=2)
        ax.fill_between(x,
                        c["contamination_mean"] - c["contamination_std"],
                        c["contamination_mean"] + c["contamination_std"],
                        alpha=0.15, color=colors[method])
    ax.set_ylabel("Contamination Ratio in Active Memory")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper right")
    ax.set_title("Memory Quality: Contamination in Injected Knowledge")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "consolidation_curve.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "consolidation_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_dir}/consolidation_curve.{{pdf,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/lifelong/formation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not (results_dir / "performance.csv").exists():
        print(f"No data at {results_dir}. Running formation experiment first...")
        import subprocess
        subprocess.run([
            sys.executable, "experiments/lifelong/run_lifelong.py",
            "--experiment", "formation",
            "--episodes", str(args.episodes),
            "--seeds", *[str(s) for s in args.seeds],
            "--output", str(results_dir.parent),
        ], check=True)

    perf, hist = load_data(results_dir)
    curves = compute_curves(perf, hist, args.episodes, args.seeds)
    plot_consolidation(curves, args.episodes, Path(args.output))


if __name__ == "__main__":
    main()
