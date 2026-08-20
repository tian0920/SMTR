"""Evaluate SMTR probe signal on real MARBLE test data.

Metrics:
  1. Pairwise ranking: P(τ̂(m1) > τ̂(m2) | τ(m1) > τ(m2))
  2. Transfer identification accuracy: predicted label vs actual label
  3. Baseline comparison: random ranking + outcome-only probe

Outputs:
  - data/evaluation_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_paired_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _get_tau_from_record(record: dict) -> int:
    """Compute tau = Y_expose - Y_withhold from a paired record."""
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    return y_expose - y_withhold


def _pairwise_ranking(
    predictions: np.ndarray,
    true_values: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> float:
    """Compute pairwise ranking accuracy."""
    n = len(predictions)
    indices = rng.randint(0, n, size=(n_samples, 2))
    correct = 0
    total = 0

    for i, j in indices:
        if true_values[i] > true_values[j]:
            total += 1
            if predictions[i] > predictions[j]:
                correct += 1
        elif true_values[i] < true_values[j]:
            total += 1
            if predictions[i] < predictions[j]:
                correct += 1

    return correct / max(total, 1)


def _predict_with_critic(critic, records: list[dict], memory_pool_path: Path) -> np.ndarray:
    """Generate tau scores using the trained critic."""
    from smtr.router.transfer_features import build_training_data_from_records

    eval_data = build_training_data_from_records(records, memory_pool_path)
    if not eval_data:
        return np.zeros(len(records))

    inputs = [item for item, _, _ in eval_data]
    predictions = critic.predict_batch(inputs)
    return np.array([p.tau_hat for p in predictions])


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    print("=" * 60)
    print("MARBLE Feasibility Test — Signal Evaluation")
    print("=" * 60)

    # Load test records
    test_path = _PROJECT_ROOT / data_cfg["test_records_path"]
    memory_pool_path = _PROJECT_ROOT / data_cfg["memory_pool_path"]
    print(f"\n  Loading test records: {test_path}")
    test_records = _load_paired_records(test_path)
    valid_records = [r for r in test_records if r.get("valid", False)]
    print(f"  Total test records: {len(test_records)}")
    print(f"  Valid test records: {len(valid_records)}")

    # Get ground truth tau values
    tau_true = np.array([_get_tau_from_record(r) for r in valid_records])
    print(f"  τ distribution: +{(tau_true > 0).sum()}, "
          f"{(tau_true < 0).sum()}, 0={(tau_true == 0).sum()}")

    # Load SMTR probe
    probe_path = _THIS_DIR / "data" / "smtr_probe.joblib"
    print(f"\n  Loading SMTR probe: {probe_path}")
    from smtr.router.transfer_critic import FourOutcomeTransferCritic
    smtr_probe = FourOutcomeTransferCritic.load(probe_path)

    # SMTR probe predictions (use build_training_data to get aligned records)
    print("  Computing SMTR probe predictions...")
    from smtr.router.transfer_features import build_training_data_from_records
    eval_data = build_training_data_from_records(valid_records, memory_pool_path)
    eval_inputs = [item for item, _, _ in eval_data]
    eval_records = [rec for _, _, rec in eval_data]
    tau_aligned = np.array([_get_tau_from_record(r) for r in eval_records])

    # Predict tau scores using predict_batch → TransferPrediction.tau_hat
    predictions_raw = smtr_probe.predict_batch(eval_inputs)
    smtr_preds = np.array([p.tau_hat for p in predictions_raw])

    # Evaluate SMTR probe (use tau_aligned which matches smtr_preds)
    rng = np.random.RandomState(eval_cfg["seed"])
    smtr_ranking = _pairwise_ranking(
        smtr_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng
    )
    print(f"  SMTR pairwise ranking: {smtr_ranking:.4f}")

    # Random baseline
    print("\n  Computing random baseline...")
    random_preds = rng.randn(len(tau_aligned))
    random_ranking = _pairwise_ranking(
        random_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng
    )
    print(f"  Random pairwise ranking: {random_ranking:.4f}")

    # Outcome-only baseline: predict P(Y_expose) without using Y_withhold
    print("\n  Computing outcome-only baseline...")
    outcome_preds = np.array([
        1.0 if r.get("share", {}).get("team_success") else 0.0
        for r in eval_records
    ])
    outcome_ranking = _pairwise_ranking(
        outcome_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng
    )
    print(f"  Outcome-only pairwise ranking: {outcome_ranking:.4f}")

    # Transfer identification accuracy
    print("\n  Computing transfer identification accuracy...")
    _LABEL_MAP = {
        "q00_neutral_failure": "neutral_failure",
        "q01_negative_transfer": "negative_transfer",
        "q10_positive_transfer": "positive_transfer",
        "q11_neutral_success": "neutral_success",
    }
    smtr_labels = []
    for pred in predictions_raw:
        probs = {
            "neutral_failure": pred.q00_neutral_failure,
            "negative_transfer": pred.q01_negative_transfer,
            "positive_transfer": pred.q10_positive_transfer,
            "neutral_success": pred.q11_neutral_success,
        }
        smtr_labels.append(max(probs, key=probs.get))

    true_labels = [r.get("label", "unknown") for r in eval_records]
    correct = sum(1 for p, t in zip(smtr_labels, true_labels) if p == t)
    smtr_identification_acc = correct / len(true_labels) if true_labels else 0.0
    print(f"  SMTR transfer identification accuracy: {smtr_identification_acc:.4f}")

    # Save results
    results = {
        "test_records": len(test_records),
        "valid_records": len(valid_records),
        "tau_distribution": {
            "positive": int((tau_true > 0).sum()),
            "negative": int((tau_true < 0).sum()),
            "neutral": int((tau_true == 0).sum()),
        },
        "smtr_probe": {
            "pairwise_ranking": round(smtr_ranking, 4),
            "identification_accuracy": round(smtr_identification_acc, 4),
        },
        "random_baseline": {
            "pairwise_ranking": round(random_ranking, 4),
        },
        "outcome_only_baseline": {
            "pairwise_ranking": round(outcome_ranking, 4),
        },
        "improvement": {
            "vs_random": round(smtr_ranking - random_ranking, 4),
            "vs_outcome_only": round(smtr_ranking - outcome_ranking, 4),
        },
    }

    results_path = _THIS_DIR / "data" / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Summary
    print("\n  Evaluation Summary:")
    print(f"    SMTR probe ranking:     {smtr_ranking:.4f}")
    print(f"    Random baseline:       {random_ranking:.4f}")
    print(f"    Outcome-only baseline: {outcome_ranking:.4f}")
    print(f"    SMTR vs random:        +{smtr_ranking - random_ranking:.4f}")
    print(f"    SMTR vs outcome-only:  +{smtr_ranking - outcome_ranking:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
