"""Cost metrics computation for TCI cost-benefit analysis.

Computes five cost-effectiveness metrics:

  1. Validation Cost (VC) = number_of_interventions
  2. Memory Validation Density (MVD) = validated / candidates
  3. Performance per Validation (PV) = final_reward / intervention_count
  4. Knowledge Gain per Cost (KGC) = (smtr_reward - baseline_reward) / cost
  5. Contamination Avoidance Efficiency (CAE) = harmful_reduction / cost
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def load_cost_results(results_dir: Path) -> list[dict]:
    """Load cost_results.csv."""
    csv_path = results_dir / "cost_results.csv"
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def compute_metrics(results: list[dict]) -> dict:
    """Compute cost metrics per method."""
    methods = ["full_memory", "retrieval", "random_validation", "smtr_tci"]
    metrics: dict[str, dict] = {}

    for method in methods:
        rows = [r for r in results if r["method"] == method]
        if not rows:
            continue

        # Aggregate across seeds
        interventions = [int(r["intervention_count"]) for r in rows]
        final_rewards = [float(r["final_reward"]) for r in rows]
        validated = [int(r["validated_memories"]) for r in rows]
        candidates = [int(r["candidate_memories_checked"]) for r in rows]
        contam_rates = [float(r["contamination_rate"]) for r in rows]

        mean_interventions = float(np.mean(interventions))
        mean_reward = float(np.mean(final_rewards))
        mean_validated = float(np.mean(validated))
        mean_candidates = float(np.mean(candidates))
        mean_contam = float(np.mean(contam_rates))

        metrics[method] = {
            "VC": mean_interventions,
            "MVD": mean_validated / mean_candidates if mean_candidates > 0 else 0.0,
            "PV": mean_reward / mean_interventions if mean_interventions > 0 else 0.0,
            "final_reward": mean_reward,
            "contamination_rate": mean_contam,
            "validated_memories": mean_validated,
            "intervention_count": mean_interventions,
        }

    # Cross-method metrics
    if "smtr_tci" in metrics and "full_memory" in metrics:
        smtr_cost = metrics["smtr_tci"]["intervention_count"]
        smtr_reward = metrics["smtr_tci"]["final_reward"]
        baseline_reward = metrics["full_memory"]["final_reward"]

        metrics["KGC"] = (
            (smtr_reward - baseline_reward) / smtr_cost
            if smtr_cost > 0 else 0.0
        )

        # Contamination avoidance efficiency
        smtr_contam = metrics["smtr_tci"]["contamination_rate"]
        full_contam = metrics["full_memory"]["contamination_rate"]
        metrics["CAE"] = (
            (full_contam - smtr_contam) / smtr_cost
            if smtr_cost > 0 else 0.0
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/cost_analysis")
    parser.add_argument("--output", default="results/cost_analysis")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_cost_results(results_dir)
    metrics = compute_metrics(results)

    # Save metrics
    metrics_path = output_dir / "cost_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Cost Metrics:")
    for method, m in metrics.items():
        if isinstance(m, dict):
            print(f"  {method}:")
            for k, v in m.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {method}: {m:.4f}")

    print(f"\nSaved: {metrics_path}")


if __name__ == "__main__":
    main()
