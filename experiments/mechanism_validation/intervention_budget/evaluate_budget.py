"""Evaluate models at each budget level.

For each (budget_ratio, seed) combination:
  1. Load trained model
  2. Predict on fixed test set
  3. Compute: Pearson, sign accuracy, pairwise ranking
  4. Record intervention cost

For the outcome-only baseline (budget=0):
  Model predicts Y_withhold, which has no correlation with tau_true.
  Expected ranking ≈ 0.5 (random).

Outputs:
  artifacts/budget_evaluation.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _pearson(preds: np.ndarray, true: np.ndarray) -> float:
    if np.std(preds) < 1e-8 or np.std(true) < 1e-8:
        return 0.0
    return float(np.corrcoef(preds, true)[0, 1])


def _sign_accuracy(preds: np.ndarray, true: np.ndarray) -> float:
    pred_signs = np.sign(preds)
    true_signs = np.sign(true)
    pred_signs[pred_signs == 0] = 1.0
    return float((pred_signs == true_signs).mean())


def _pairwise_ranking(
    preds: np.ndarray,
    true: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> float:
    """Pairwise ranking: P(τ̂_i > τ̂_j | τ_i > τ_j)."""
    n = len(preds)
    indices = rng.randint(0, n, size=(n_samples, 2))
    correct = 0
    total = 0

    for i, j in indices:
        if true[i] > true[j]:
            total += 1
            if preds[i] > preds[j]:
                correct += 1
        elif true[i] < true[j]:
            total += 1
            if preds[i] < preds[j]:
                correct += 1

    return correct / max(total, 1)


def main() -> None:
    config = _load_config()
    budget_cfg = config["budget"]
    eval_cfg = config["evaluation"]
    seed = config["environment"]["seed"]

    ratios = budget_cfg["ratios"]
    seeds = budget_cfg["seeds"]
    n_pairwise = eval_cfg["n_pairwise_samples"]

    print("=" * 60)
    print("Intervention Budget — Evaluation")
    print("=" * 60)

    # Load fixed test set.
    artifacts_dir = _THIS_DIR / "artifacts"
    test_data = np.load(artifacts_dir / "test_data.npz")
    X_test = test_data["combined_features"]   # (n, 32)
    tau_true = test_data["tau_true"]           # (n,)

    print(f"  Test samples: {len(tau_true)}")
    print(f"  τ distribution: +{(tau_true > 0).sum()}"
          f"/-{(tau_true < 0).sum()}")
    print(f"  Pairwise samples: {n_pairwise}\n")

    # Results storage.
    results: dict[str, dict] = {}

    for ratio in ratios:
        ratio_results = []

        for bseed in seeds:
            ckpt_name = (
                f"model_budget_{ratio:.2f}_seed_{bseed}.joblib"
            )
            model = joblib.load(artifacts_dir / ckpt_name)

            # Predict.
            preds = model.predict(X_test)

            # For outcome-only model (budget=0), predictions are
            # Y_withhold estimates. Convert to "tau" by centering:
            # tau_hat ≈ 2*(pred - 0.5) to get sign-like values.
            # This gives no ranking signal since Y_withhold ⊥ tau.
            if ratio <= 0.0:
                # Center predictions around 0 as pseudo-tau.
                preds = preds - preds.mean()

            rng = np.random.RandomState(seed)
            pearson = _pearson(preds, tau_true)
            sign_acc = _sign_accuracy(preds, tau_true)
            ranking = _pairwise_ranking(preds, tau_true, n_pairwise, rng)

            cost = ratio  # Relative cost = budget ratio.

            ratio_results.append({
                "seed": bseed,
                "pearson": pearson,
                "sign_accuracy": sign_acc,
                "ranking": ranking,
                "cost": cost,
            })

        # Average across seeds.
        avg_pearson = float(np.mean([r["pearson"] for r in ratio_results]))
        avg_sign = float(np.mean([r["sign_accuracy"] for r in ratio_results]))
        avg_ranking = float(np.mean([r["ranking"] for r in ratio_results]))
        std_pearson = float(np.std([r["pearson"] for r in ratio_results]))
        std_sign = float(np.std([r["sign_accuracy"] for r in ratio_results]))
        std_ranking = float(np.std([r["ranking"] for r in ratio_results]))

        ratio_key = f"{ratio:.2f}"
        results[ratio_key] = {
            "avg_pearson": avg_pearson,
            "avg_sign": avg_sign,
            "avg_ranking": avg_ranking,
            "std_pearson": std_pearson,
            "std_sign": std_sign,
            "std_ranking": std_ranking,
            "cost": ratio,
            "per_seed": ratio_results,
        }

        print(f"  Budget {ratio:.0%}: "
              f"Pearson={avg_pearson:.4f}±{std_pearson:.4f}, "
              f"Sign={avg_sign:.4f}±{std_sign:.4f}, "
              f"Ranking={avg_ranking:.4f}±{std_ranking:.4f}, "
              f"Cost={ratio:.2f}")

    # Save results.
    # Flatten for npz.
    save_dict = {}
    for ratio_key, res in results.items():
        for metric, val in res.items():
            if isinstance(val, list):
                for i, item in enumerate(val):
                    for k, v in item.items():
                        save_dict[f"{ratio_key}_per_seed_{i}_{k}"] = v
            else:
                save_dict[f"{ratio_key}_{metric}"] = val

    np.savez(artifacts_dir / "budget_evaluation.npz", **save_dict)

    # Also save as JSON for easier reading.
    import json

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

    with open(artifacts_dir / "budget_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)

    print(f"\n  Saved: artifacts/budget_evaluation.npz")
    print(f"  Saved: artifacts/budget_evaluation.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
