"""Plot baseline long-term performance curves.

Generates three figures from the baseline comparison results:

  Figure 1: Reward vs episode (cumulative moving average)
  Figure 2: Stored memory count vs episode
  Figure 3: Contamination retention vs episode

Methods: Full Memory, Retrieval, Reflexion, Heuristic, AgeMem, SMTR

Output: figures/baseline_longterm/
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# Use non-interactive backend so this works on headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS: dict[str, str] = {
    "full_memory": "Full Memory",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "agile": "AGILE-inspired",
    "heuristic": "Heuristic",
    "agemem": "AgeMem-inspired",
    "smtr_tci": "SMTR-TCI",
}

METHOD_COLORS: dict[str, str] = {
    "full_memory": "#888888",
    "retrieval": "#4488CC",
    "reflexion": "#CC8844",
    "agile": "#88CC44",
    "heuristic": "#CC4488",
    "agemem": "#44CC88",
    "smtr_tci": "#CC4444",
}

PLOT_METHODS = ["full_memory", "retrieval", "reflexion", "heuristic", "agemem", "smtr_tci"]


def load_performance(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_history(path: Path) -> list[dict]:
    import json
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _moving_average(values: list[float], window: int = 10) -> list[float]:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode="valid"))


def _episode_mean_per_method(perf: list[dict], key: str):
    """Group by (method, episode), compute mean across seeds."""
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in perf:
        ep = int(row["episode"])
        grouped[row["method"]][ep].append(float(row[key]))
    result: dict[str, tuple[list[int], list[float]]] = {}
    for method, ep_data in grouped.items():
        eps = sorted(ep_data.keys())
        means = [float(np.mean(ep_data[ep])) for ep in eps]
        result[method] = (eps, means)
    return result


def plot_reward_curves(perf: list[dict], output_dir: Path, window: int = 10) -> None:
    """Figure 1: Reward vs episode with moving average."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = _episode_mean_per_method(perf, "reward")
    for method in PLOT_METHODS:
        if method not in data:
            continue
        eps, means = data[method]
        smoothed = _moving_average(means, window)
        smooth_eps = eps[window - 1:] if len(eps) >= window else eps
        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        ax.plot(smooth_eps, smoothed, label=label, color=color, linewidth=1.5)
        # Raw data as faint line
        ax.plot(eps, means, color=color, alpha=0.15, linewidth=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (moving avg)")
    ax.set_title("Baseline Long-term Reward Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_vs_episode.png", dpi=150)
    fig.savefig(output_dir / "reward_vs_episode.pdf")
    plt.close(fig)
    print(f"  Saved reward_vs_episode.png/pdf")


def plot_memory_size_curves(perf: list[dict], output_dir: Path) -> None:
    """Figure 2: Stored memory count vs episode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = _episode_mean_per_method(perf, "n_stored")
    for method in PLOT_METHODS:
        if method not in data:
            continue
        eps, means = data[method]
        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        ax.plot(eps, means, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Stored Memories")
    ax.set_title("Memory Bank Size Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_size_vs_episode.png", dpi=150)
    fig.savefig(output_dir / "memory_size_vs_episode.pdf")
    plt.close(fig)
    print(f"  Saved memory_size_vs_episode.png/pdf")


def plot_contamination_curves(hist: list[dict], perf: list[dict], output_dir: Path) -> None:
    """Figure 3: Harmful memory retention vs episode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # For each method, track contaminated memory retention rate per episode
    for method in PLOT_METHODS:
        m_hist = [r for r in hist if r["method"] == method]
        if not m_hist:
            continue
        # Group by (seed, episode)
        seed_ep_data: dict[int, dict[int, tuple[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0, 0))
        )
        for r in m_hist:
            seed = r["seed"]
            ep = r["episode"]
            contam = r.get("contamination", "none")
            status = r["status"]
            total, contam_count = seed_ep_data[seed][ep]
            if contam != "none" and status in ("validated", "candidate"):
                contam_count += 1
            total += 1
            seed_ep_data[seed][ep] = (total, contam_count)

        # Average across seeds
        ep_ratios: dict[int, list[float]] = defaultdict(list)
        for seed, ep_data in seed_ep_data.items():
            for ep, (total, contam_count) in ep_data.items():
                if total > 0:
                    ep_ratios[ep].append(contam_count / total)

        eps = sorted(ep_ratios.keys())
        means = [float(np.mean(ep_ratios[ep])) for ep in eps]

        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        ax.plot(eps, means, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Harmful Retention Rate")
    ax.set_title("Contaminated Memory Retention Over Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "contamination_vs_episode.png", dpi=150)
    fig.savefig(output_dir / "contamination_vs_episode.pdf")
    plt.close(fig)
    print(f"  Saved contamination_vs_episode.png/pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/baseline_comparison/formation")
    parser.add_argument("--output", default="figures/baseline_longterm")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    perf_path = results_dir / "performance.csv"
    hist_path = results_dir / "memory_history.jsonl"

    if not perf_path.exists():
        print(f"ERROR: {perf_path} not found. Run the benchmark first.")
        sys.exit(1)

    perf = load_performance(perf_path)
    hist = load_history(hist_path) if hist_path.exists() else []

    print("Generating baseline long-term curves...")
    plot_reward_curves(perf, output_dir)
    plot_memory_size_curves(perf, output_dir)
    if hist:
        plot_contamination_curves(hist, perf, output_dir)
    else:
        print("  Skipping contamination plot (no memory_history.jsonl)")

    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
