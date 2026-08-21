"""MARBLE statistical analysis (Phase 9).

Computes:
  - Per-method mean, std over seeds
  - Bootstrap 95% CI
  - SMTR vs best baseline comparison

Output: results/marble/main/marble_significance.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap CI."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    rng = np.random.RandomState(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(float(np.mean(sample)))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_means, alpha * 100))
    upper = float(np.percentile(bootstrap_means, (1.0 - alpha) * 100))
    return mean, lower, upper


def main() -> None:
    csv_path = _PROJECT_ROOT / "results" / "marble" / "main" / "baseline_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run main experiment first.")
        return

    rows = load_csv(csv_path)

    methods = sorted(set(r["method"] for r in rows))
    results: list[dict] = []

    print("=== MARBLE Statistical Analysis ===")
    print(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'CI95 Low':>10} {'CI95 High':>10}")
    print("-" * 65)

    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        rewards = [float(r["method_reward"]) for r in m_rows]
        mean, ci_low, ci_high = bootstrap_ci(rewards)
        std = float(np.std(rewards))
        results.append({
            "method": method,
            "mean_reward": mean,
            "std_reward": std,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "n_groups": len(m_rows),
        })
        print(f"{method:<20} {mean:>10.4f} {std:>10.4f} {ci_low:>10.4f} {ci_high:>10.4f}")

    # SMTR vs best baseline comparison
    smtr_result = next((r for r in results if r["method"] == "smtr_tci"), None)
    baselines = [r for r in results if r["method"] != "smtr_tci"]
    best_baseline = max(baselines, key=lambda r: r["mean_reward"]) if baselines else None

    if smtr_result and best_baseline:
        delta = smtr_result["mean_reward"] - best_baseline["mean_reward"]
        pct = delta / best_baseline["mean_reward"] * 100 if best_baseline["mean_reward"] else 0
        print(f"\nSMTR-TCI vs best baseline ({best_baseline['method']}):")
        print(f"  Delta: {delta:+.4f} ({pct:+.1f}%)")
        print(f"  SMTR CI: [{smtr_result['ci95_lower']:.4f}, {smtr_result['ci95_upper']:.4f}]")

    # Write CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "main"
    csv_out = output_dir / "marble_significance.csv"
    if results:
        fieldnames = list(results[0].keys())
        with csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nWritten: {csv_out}")


if __name__ == "__main__":
    main()
