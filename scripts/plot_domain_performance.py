"""Plot MARBLE domain performance comparison (Task A3).

Figure: Normalized reward across domains.
- x: MARBLE domains (solo, small, medium, large, complex)
- y: normalized reward
- SMTR-TCI vs best baseline
- No color coding — use markers and legend only.

Output: figures/marble_domain_performance.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

DOMAIN_ORDER = ["solo", "small", "medium", "large", "complex"]
DOMAIN_LABELS = {
    "solo": "Solo\n(1-2)",
    "small": "Small\n(3)",
    "medium": "Medium\n(4-5)",
    "large": "Large\n(6)",
    "complex": "Complex\n(7+)",
}

METHOD_LABELS = {
    "smtr_tci": "SMTR-TCI",
    "best_baseline": "Best Baseline",
}


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = _PROJECT_ROOT / "results" / "marble" / "domain_analysis" / "domain_wise_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    rows = load_csv(csv_path)

    # Build domain -> method -> reward
    data: dict[str, dict[str, float]] = {}
    for r in rows:
        data.setdefault(r["domain"], {})[r["method"]] = float(r["mean_reward"])

    # Extract SMTR and best baseline per domain
    smtr_rewards: list[float] = []
    best_bl_rewards: list[float] = []
    x_labels: list[str] = []

    for domain in DOMAIN_ORDER:
        d_data = data.get(domain, {})
        smtr_val = d_data.get("smtr_tci", 0.0)
        baseline_methods = [m for m in d_data if m != "smtr_tci" and m != "no_memory"]
        best_bl = max((d_data.get(m, 0.0) for m in baseline_methods), default=0.0)

        smtr_rewards.append(smtr_val)
        best_bl_rewards.append(best_bl)
        x_labels.append(DOMAIN_LABELS.get(domain, domain))

    # Normalize rewards to [0, 1] range
    all_rewards = smtr_rewards + best_bl_rewards
    r_min = min(all_rewards)
    r_max = max(all_rewards)
    if r_max > r_min:
        smtr_norm = [(r - r_min) / (r_max - r_min) for r in smtr_rewards]
        bl_norm = [(r - r_min) / (r_max - r_min) for r in best_bl_rewards]
    else:
        smtr_norm = [0.5] * len(smtr_rewards)
        bl_norm = [0.5] * len(best_bl_rewards)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.35

    bars_bl = ax.bar(x - width / 2, bl_norm, width,
                      label="Best Baseline", hatch="//", color="white", edgecolor="black")
    bars_smtr = ax.bar(x + width / 2, smtr_norm, width,
                        label="SMTR-TCI", hatch="\\\\", color="lightgray", edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars_bl, best_bl_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_smtr, smtr_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("MARBLE Domain (Agent Count)")
    ax.set_ylabel("Normalized Reward")
    ax.set_title("SMTR-TCI vs Best Baseline Across MARBLE Domains")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(smtr_norm + bl_norm) + 0.2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = _PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "marble_domain_performance.pdf"
    png_path = output_dir / "marble_domain_performance.png"
    fig.savefig(pdf_path, dpi=150)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Written: {pdf_path}, {png_path}")


if __name__ == "__main__":
    main()
