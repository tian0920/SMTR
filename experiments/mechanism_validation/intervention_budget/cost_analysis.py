"""Cost analysis: efficiency and shared control ablation.

Part 1 — Cost Efficiency
  Efficiency = RankingAccuracy / Cost
  For budget=0: cost is defined as epsilon (1e-6) to avoid division by zero.

  Expected: lower budgets have higher efficiency (ranking/cost ratio)
  because they achieve good performance with less intervention.

Part 2 — Shared Control Ablation
  Naive approach: each (memory, receiver) pair needs own control rollout.
    cost_naive = N_memory × N_receiver = 50 × 20 = 1000

  Shared control (SMTR): reuse control across memories per receiver.
    cost_shared = N_receiver = 20

  Reduction = 1 - cost_shared / cost_naive = 1 - 20/1000 = 0.98 (≥80%)

Outputs:
  artifacts/cost_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def main() -> None:
    config = _load_config()
    control_cfg = config["shared_control"]
    acceptance = config["acceptance"]

    n_memories = control_cfg["n_memories"]
    n_receivers = control_cfg["n_receivers"]

    print("=" * 60)
    print("Intervention Budget — Cost Analysis")
    print("=" * 60)

    # Load evaluation results.
    artifacts_dir = _THIS_DIR / "artifacts"
    with open(artifacts_dir / "budget_evaluation.json") as f:
        eval_results = json.load(f)

    # ── Part 1: Cost Efficiency ──
    print("\n  ── Cost Efficiency ──")
    efficiency_results = {}

    for ratio_key, res in eval_results.items():
        ranking = res["avg_ranking"]
        cost = res["cost"]

        # Avoid division by zero for budget=0.
        effective_cost = max(cost, 1e-6)
        efficiency = ranking / effective_cost

        efficiency_results[ratio_key] = {
            "ranking": ranking,
            "cost": cost,
            "efficiency": efficiency,
        }
        print(f"  Budget {float(ratio_key):.0%}: "
              f"ranking={ranking:.4f}, cost={cost:.2f}, "
              f"efficiency={efficiency:.4f}")

    # Find best non-full efficiency.
    full_efficiency = efficiency_results.get("1.00", {}).get(
        "efficiency", 0.0
    )
    best_partial = {
        k: v for k, v in efficiency_results.items()
        if float(k) < 1.0 and float(k) > 0.0
    }
    if best_partial:
        best_key = max(best_partial, key=lambda k: best_partial[k]["efficiency"])
        best_efficiency = best_partial[best_key]["efficiency"]
        efficiency_exceeds_full = best_efficiency > full_efficiency
        print(f"\n  Best partial efficiency: "
              f"budget={best_key}, efficiency={best_efficiency:.4f}")
        print(f"  Full efficiency: {full_efficiency:.4f}")
        print(f"  Partial > Full: {efficiency_exceeds_full}")
    else:
        best_efficiency = 0.0
        efficiency_exceeds_full = False

    # ── Part 2: Shared Control Ablation ──
    print("\n  ── Shared Control Ablation ──")

    naive_cost = n_memories * n_receivers  # Each pair needs own control.
    shared_cost = n_receivers              # Reuse control per receiver.
    reduction = 1.0 - shared_cost / naive_cost

    print(f"  N_memories: {n_memories}")
    print(f"  N_receivers: {n_receivers}")
    print(f"  Naive cost (N_m × N_r): {naive_cost}")
    print(f"  Shared control cost (N_r): {shared_cost}")
    print(f"  Cost reduction: {reduction:.4f} ({reduction:.1%})")

    # ── Acceptance check ──
    print("\n  ── Acceptance Criteria ──")

    budget_50_ranking = eval_results.get("0.50", {}).get("avg_ranking", 0.0)
    budget_25_ranking = eval_results.get("0.25", {}).get("avg_ranking", 0.0)

    checks = {
        "budget_50_ranking": {
            "description": "50% budget ranking ≥ 0.90",
            "value": budget_50_ranking,
            "threshold": acceptance["budget_50_ranking_min"],
            "passed": budget_50_ranking >= acceptance["budget_50_ranking_min"],
        },
        "budget_25_ranking": {
            "description": "25% budget ranking ≥ 0.80",
            "value": budget_25_ranking,
            "threshold": acceptance["budget_25_ranking_min"],
            "passed": budget_25_ranking >= acceptance["budget_25_ranking_min"],
        },
        "efficiency_exceeds_full": {
            "description": "At least one non-100% budget efficiency > Full",
            "value": round(best_efficiency, 4),
            "threshold": round(full_efficiency, 4),
            "passed": efficiency_exceeds_full,
        },
        "shared_control_reduction": {
            "description": f"Shared control reduction ≥ "
                           f"{acceptance['shared_control_reduction_min']:.0%}",
            "value": reduction,
            "threshold": acceptance["shared_control_reduction_min"],
            "passed": reduction >= acceptance["shared_control_reduction_min"],
        },
    }

    for name, check in checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['description']}: "
              f"{check['value']:.4f} (threshold: {check['threshold']})")

    all_passed = all(c["passed"] for c in checks.values())
    verdict = "PASS" if all_passed else "FAIL"
    print(f"\n  Verdict: {verdict}")

    # Save.
    output = {
        "efficiency": efficiency_results,
        "shared_control": {
            "naive_cost": naive_cost,
            "shared_cost": shared_cost,
            "reduction": reduction,
        },
        "acceptance_criteria": checks,
        "verdict": verdict,
    }

    out_path = artifacts_dir / "cost_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
