"""Train SMTR critic probe on balanced MARBLE intervention data.

Two training stages:
  1. FourOutcomeTransferCritic with class-weighted sample_weights
  2. Sign classifier z = sign(τ) ∈ {-1, 0, +1} (auxiliary head)

Uses the balanced training set produced by collect_interventions.py.

Outputs:
  - data/smtr_probe.joblib        (trained critic checkpoint)
  - data/sign_classifier.joblib   (3-class sign classifier)
  - data/probe_metrics.json       (training metrics)
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


def _get_tau(record: dict) -> int:
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    return y_expose - y_withhold


def _build_class_weighted_sample_weights(
    records: list[dict],
    class_weights: dict[str, float],
) -> np.ndarray:
    """Compute per-record sample weights combining edge-equal and class weights.

    Edge-equal weight: 1/n_e per treatment edge (same as SMTR).
    Class weight: multiplier based on the record's transfer label.
    """
    from smtr.counterfactual.edge_keys import (
        treatment_edge_key,
    )

    edge_counts = Counter(treatment_edge_key(rec) for rec in records)
    weights = []
    for rec in records:
        edge_w = 1.0 / edge_counts[treatment_edge_key(rec)]
        label = rec.get("label", "neutral_failure")
        class_w = class_weights.get(label, 1.0)
        weights.append(edge_w * class_w)

    return np.asarray(weights, dtype=float)


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    probe_cfg = config["probe"]

    print("=" * 60)
    print("MARBLE Feasibility Test — SMTR Probe Training (Balanced)")
    print("=" * 60)

    # ── Import SMTR internals ──
    from smtr.router.transfer_critic import FourOutcomeTransferCritic
    from smtr.router.transfer_features import build_training_data_from_records
    from smtr.counterfactual.edge_keys import group_records_by_control_family

    # ── Paths ──
    balanced_train_path = _THIS_DIR / "data" / "balanced_train.jsonl"
    balanced_val_path = _THIS_DIR / "data" / "balanced_validation.jsonl"
    memory_pool_path = _PROJECT_ROOT / data_cfg["memory_pool_path"]
    output_path = _THIS_DIR / "data" / "smtr_probe.joblib"
    sign_clf_path = _THIS_DIR / "data" / "sign_classifier.joblib"

    # TCI supervision (optional)
    tci_contrasts_path_raw = data_cfg.get("tci_contrasts_path")
    tci_perturbations_manifest_raw = data_cfg.get("tci_perturbations_manifest_path")
    tci_contrasts_path = (
        _PROJECT_ROOT / tci_contrasts_path_raw if tci_contrasts_path_raw else None
    )
    tci_perturbations_manifest_path = (
        _PROJECT_ROOT / tci_perturbations_manifest_raw
        if tci_perturbations_manifest_raw
        else None
    )

    print(f"\n  Balanced train: {balanced_train_path}")
    print(f"  Balanced val: {balanced_val_path}")
    print(f"  Memory pool: {memory_pool_path}")
    if tci_contrasts_path:
        print(f"  TCI contrasts: {tci_contrasts_path}")

    # ── Load balanced training records ──
    train_records = []
    for line in balanced_train_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            train_records.append(json.loads(line))
    train_valid = [r for r in train_records if r.get("valid", False)]
    print(f"\n  Training records: {len(train_valid)}")

    # ── Build training data ──
    print("\n  Building training features...")
    train_data = build_training_data_from_records(train_valid, memory_pool_path)
    if not train_data:
        print("  ERROR: no training data after feature construction")
        sys.exit(1)

    inputs = [item for item, _, _ in train_data]
    labels = [label for _, label, _ in train_data]
    records = [rec for _, _, rec in train_data]

    print(f"  Training examples: {len(inputs)}")
    print(f"  Label distribution: {dict(Counter(labels))}")

    # ── Compute class-weighted sample weights ──
    class_weights = probe_cfg.get("class_weights", {})
    print(f"\n  Class weights: {class_weights}")
    sample_weights = _build_class_weighted_sample_weights(records, class_weights)
    print(f"  Sample weights: min={sample_weights.min():.4f}, "
          f"max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")

    # ── Bootstrap clusters ──
    bootstrap_clusters = group_records_by_control_family(records)

    # ── Build TCI inputs (optional) ──
    tci_inputs = None
    if tci_contrasts_path and tci_perturbations_manifest_path:
        from smtr.marble.training import _build_tci_inputs_for_critic
        tci_inputs = _build_tci_inputs_for_critic(
            tci_contrasts_path=tci_contrasts_path,
            perturbations_manifest_path=tci_perturbations_manifest_path,
            paired_records_path=balanced_train_path,
            memory_pool_path=memory_pool_path,
        )
        if tci_inputs:
            print(f"  TCI contrast pairs: {len(tci_inputs)}")

    # ── Train FourOutcomeTransferCritic ──
    print(f"\n  Training critic probe:")
    print(f"    n_features: {probe_cfg['n_features']}")
    print(f"    feature_block: {probe_cfg['feature_block']}")
    print(f"    critic_mode: {probe_cfg['critic_mode']}")
    print(f"    seed: {probe_cfg['seed']}")

    critic = FourOutcomeTransferCritic(
        n_features=probe_cfg["n_features"],
        n_bootstrap=probe_cfg.get("n_bootstrap", 31),
        feature_block=probe_cfg["feature_block"],
        critic_mode=probe_cfg["critic_mode"],
        seed=probe_cfg["seed"],
    )

    critic.fit(
        inputs,
        labels,
        records=records,
        coverage_mode="pilot",
        sample_weights=sample_weights,
        bootstrap_clusters=bootstrap_clusters,
        tci_inputs=tci_inputs,
        tci_alpha=probe_cfg.get("tci_alpha", 1.0),
    )

    # Save critic
    critic.save(output_path)
    print(f"\n  Saved critic: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # ── Train sign classifier z = sign(τ) ──
    sign_clf_used = probe_cfg.get("sign_classifier", False)
    sign_metrics = {}
    if sign_clf_used:
        print("\n  Training sign classifier z = sign(τ)...")
        from sklearn.linear_model import LogisticRegression

        encoder = critic.encoder
        # encode_batch returns a sparse matrix; convert to dense
        X_train_sparse = encoder.encode_batch(inputs)
        X_train = X_train_sparse.toarray() if hasattr(X_train_sparse, 'toarray') else np.asarray(X_train_sparse)
        z_labels = np.array([np.sign(_get_tau(r)) for r in records])
        # Remap: -1→0, 0→1, +1→2 for sklearn
        z_mapped = z_labels + 1  # {-1,0,1} → {0,1,2}
        z_class_names = ["negative", "neutral", "positive"]

        # Use class_weight=balanced for the sign classifier
        sign_clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=probe_cfg["seed"],
            C=1.0,
        )
        sign_clf.fit(X_train, z_mapped)

        # Evaluate on training data
        z_pred = sign_clf.predict(X_train)
        sign_acc = (z_pred == z_mapped).mean()
        sign_metrics = {
            "train_accuracy": round(float(sign_acc), 4),
            "class_labels": z_class_names,
            "label_distribution": {
                "negative": int((z_mapped == 0).sum()),
                "neutral": int((z_mapped == 1).sum()),
                "positive": int((z_mapped == 2).sum()),
            },
        }
        print(f"    Train accuracy: {sign_acc:.4f}")
        print(f"    Distribution: {sign_metrics['label_distribution']}")

        joblib.dump(sign_clf, sign_clf_path)
        print(f"  Saved sign classifier: {sign_clf_path}")

    # ── Validation metrics ──
    val_records = []
    if balanced_val_path.exists():
        for line in balanced_val_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                val_records.append(json.loads(line))
    val_valid = [r for r in val_records if r.get("valid", False)]

    val_metrics = {}
    if val_valid:
        val_data = build_training_data_from_records(val_valid, memory_pool_path)
        val_inputs = [item for item, _, _ in val_data]
        val_preds = critic.predict_batch(val_inputs)
        val_true_labels = [r.get("label", "unknown") for _, _, r in val_data]

        # Classification accuracy
        correct = 0
        for pred, true_label in zip(val_preds, val_true_labels):
            probs = {
                "neutral_failure": pred.q00_neutral_failure,
                "negative_transfer": pred.q01_negative_transfer,
                "positive_transfer": pred.q10_positive_transfer,
                "neutral_success": pred.q11_neutral_success,
            }
            pred_label = max(probs, key=probs.get)
            if pred_label == true_label:
                correct += 1
        val_acc = correct / len(val_true_labels) if val_true_labels else 0.0

        # Prediction distribution
        taus = [p.tau_hat for p in val_preds]
        val_metrics = {
            "validation_records": len(val_valid),
            "validation_accuracy": round(val_acc, 4),
            "tau_pred_mean": round(float(np.mean(taus)), 4),
            "tau_pred_std": round(float(np.std(taus)), 4),
            "tau_pred_min": round(float(np.min(taus)), 4),
            "tau_pred_max": round(float(np.max(taus)), 4),
            "unique_tau_values": len(set(round(t, 6) for t in taus)),
        }
        print(f"\n  Validation: {val_acc:.4f} accuracy, "
              f"τ pred std={np.std(taus):.4f}")

    # ── Save all metrics ──
    metrics = {
        "train_records": len(records),
        "label_distribution": dict(Counter(labels)),
        "class_weights": class_weights,
        "sample_weights_stats": {
            "min": round(float(sample_weights.min()), 4),
            "max": round(float(sample_weights.max()), 4),
            "mean": round(float(sample_weights.mean()), 4),
        },
        "n_features": probe_cfg["n_features"],
        "feature_block": probe_cfg["feature_block"],
        "critic_mode": probe_cfg["critic_mode"],
        "seed": probe_cfg["seed"],
        "tci_distillation_n_examples": len(tci_inputs) if tci_inputs else 0,
        "sign_classifier": sign_metrics,
        **val_metrics,
    }

    metrics_path = _THIS_DIR / "data" / "probe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Saved metrics: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
