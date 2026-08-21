"""Plot noise robustness figures (Task B3).

Figure 1: noise level vs reward (SMTR vs Random Validation)
Figure 2: noise level vs harmful retention

Output: figures/noise_robustness.pdf, figures/noise_robustness.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = _PROJECT_ROOT / "results" / "noise_robustness" / "noise_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        return

    summary = json.loads(summary_path.read_text())

    # Extract data
    noise_levels = sorted(set(s["noise_sigma"] for s in summary))
    smtr_data = {s["noise_sigma"]: s for s in summary if s["method"] == "smtr_tci"}
    rand_data = {s["noise_sigma"]: s for s in summary if s["method"] == "random_validation"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Figure 1: noise vs reward
    smtr_rewards = [smtr_data.get(s, {}).get("mean_reward", 0) for s in noise_levels]
    rand_rewards = [rand_data.get(s, {}).get("mean_reward", 0) for s in noise_levels]
    smtr_stds = [smtr_data.get(s, {}).get("std_reward", 0) for s in noise_levels]
    rand_stds = [rand_data.get(s, {}).get("std_reward", 0) for s in noise_levels]

    ax1.errorbar(noise_levels, smtr_rewards, yerr=smtr_stds,
                 marker="o", linestyle="-", label="SMTR-TCI", color="black", linewidth=2)
    ax1.errorbar(noise_levels, rand_rewards, yerr=rand_stds,
                 marker="s", linestyle="--", label="Random Validation", color="gray", linewidth=2)
    ax1.set_xlabel("Noise Level (σ)")
    ax1.set_ylabel("Method Reward")
    ax1.set_title("TCI Robustness Under Noisy Reward Observations")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Figure 2: noise vs harmful retention
    smtr_harmful = [smtr_data.get(s, {}).get("mean_harmful", 0) for s in noise_levels]
    rand_harmful = [rand_data.get(s, {}).get("mean_harmful", 0) for s in noise_levels]

    ax2.plot(noise_levels, smtr_harmful, marker="o", linestyle="-",
             label="SMTR-TCI", color="black", linewidth=2)
    ax2.plot(noise_levels, rand_harmful, marker="s", linestyle="--",
             label="Random Validation", color="gray", linewidth=2)
    ax2.set_xlabel("Noise Level (σ)")
    ax2.set_ylabel("Harmful Memory Retention")
    ax2.set_title("Harmful Retention Under Noise")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_dir = _PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "noise_robustness.pdf"
    png_path = output_dir / "noise_robustness.png"
    fig.savefig(pdf_path, dpi=150)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Written: {pdf_path}, {png_path}")


if __name__ == "__main__":
    main()
