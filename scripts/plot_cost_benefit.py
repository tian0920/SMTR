"""Cost-benefit curve plotting script.

Generates two figures:
  1. Cost vs Performance: interventions (x) vs final reward (y)
  2. Cost vs Memory Quality: interventions (x) vs quality score (y)

Quality score = transfer_gain + reuse - contamination (analysis metric)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def load_cost_results(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_scaling_results(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_cost_performance(
    cost_results: list[dict],
    scaling_results: list[dict],
    output_dir: Path,
) -> None:
    """Generate cost vs performance figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TCI Cost-Benefit Analysis", fontsize=14, fontweight="bold")

    # ── Panel 1: Method comparison (cost vs reward) ──
    ax = axes[0]
    method_labels = {
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "random_validation": "Random Validation",
        "smtr_tci": "SMTR-TCI",
    }
    colors = {
        "full_memory": "#95a5a6",
        "retrieval": "#f39c12",
        "random_validation": "#e74c3c",
        "smtr_tci": "#2ecc71",
    }

    for method in ["full_memory", "retrieval", "random_validation", "smtr_tci"]:
        rows = [r for r in cost_results if r["method"] == method]
        if not rows:
            continue
        interventions = np.mean([int(r["intervention_count"]) for r in rows])
        reward = np.mean([float(r["final_reward"]) for r in rows])
        reward_std = np.std([float(r["final_reward"]) for r in rows])
        ax.errorbar(interventions, reward, yerr=reward_std, fmt="o",
                    color=colors[method], markersize=10, capsize=5,
                    label=method_labels[method], linewidth=2)

    ax.set_xlabel("Number of Interventions")
    ax.set_ylabel("Final Reward (last 20 ep)")
    ax.set_title("Cost vs Performance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Coverage scaling ──
    ax = axes[1]
    coverages = sorted(set(float(r["coverage"]) for r in scaling_results))
    rewards = []
    interventions = []
    for cov in coverages:
        rows = [r for r in scaling_results if float(r["coverage"]) == cov]
        rewards.append(np.mean([float(r["final_reward"]) for r in rows]))
        interventions.append(np.mean([int(r["intervention_count"]) for r in rows]))

    ax.plot(interventions, rewards, "o-", color="#2ecc71", linewidth=2, markersize=8)
    for i, cov in enumerate(coverages):
        ax.annotate(f"{cov:.0%}", (interventions[i], rewards[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Number of Interventions")
    ax.set_ylabel("Final Reward")
    ax.set_title("Validation Coverage Scaling")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cost_performance_curve.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "cost_performance_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/cost_performance_curve.{{pdf,png}}")


def plot_cost_quality(
    cost_results: list[dict],
    output_dir: Path,
) -> None:
    """Generate cost vs memory quality figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Cost vs Memory Quality", fontsize=14, fontweight="bold")

    method_labels = {
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "random_validation": "Random Validation",
        "smtr_tci": "SMTR-TCI",
    }
    colors = {
        "full_memory": "#95a5a6",
        "retrieval": "#f39c12",
        "random_validation": "#e74c3c",
        "smtr_tci": "#2ecc71",
    }

    for method in ["full_memory", "retrieval", "random_validation", "smtr_tci"]:
        rows = [r for r in cost_results if r["method"] == method]
        if not rows:
            continue
        interventions = np.mean([int(r["intervention_count"]) for r in rows])
        reward = np.mean([float(r["final_reward"]) for r in rows])
        contam = np.mean([float(r["contamination_rate"]) for r in rows])
        # Quality = reward - contamination penalty
        quality = reward - contam
        ax.bar(method_labels[method], quality, color=colors[method], alpha=0.85)

    ax.set_ylabel("Knowledge Quality (reward − contamination)")
    ax.set_title("Memory Quality by Method")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "cost_quality.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "cost_quality.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/cost_quality.{{pdf,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/cost_analysis")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)

    cost_results = load_cost_results(results_dir / "cost_results.csv")
    scaling_results = []
    scaling_path = results_dir / "cost_scaling_results.csv"
    if scaling_path.exists():
        scaling_results = load_scaling_results(scaling_path)

    plot_cost_performance(cost_results, scaling_results, output_dir)
    plot_cost_quality(cost_results, output_dir)


if __name__ == "__main__":
    main()
