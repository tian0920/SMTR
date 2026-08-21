"""Train SMTR critic probe with multiple training modes.

Training modes:
  - regression: Standard FourOutcomeTransferCritic (baseline)
  - ranking: Pairwise ranking loss with SGD
  - hybrid: L = L_rank + 0.5*L_tau + 0.5*L_sign (recommended)

Uses the balanced training set from generate_splits.py.

Outputs:
  - splits/<split>/smtr_probe.joblib  (trained model)
  - splits/<split>/sign_classifier.joblib (sign head for hybrid mode)
  - splits/<split>/probe_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge, LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from _probe_models import EnhancedEncoder, RankingProbe, MockPrediction  # noqa: F401

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


def _train_regression(inputs, labels, records, probe_cfg, memory_pool_path):
    """Standard FourOutcomeTransferCritic training (baseline)."""
    from smtr.router.transfer_critic import FourOutcomeTransferCritic
    from smtr.counterfactual.edge_keys import group_records_by_control_family

    critic = FourOutcomeTransferCritic(
        n_features=probe_cfg["n_features"],
        n_bootstrap=probe_cfg.get("n_bootstrap", 7),
        feature_block=probe_cfg["feature_block"],
        critic_mode=probe_cfg["critic_mode"],
        seed=probe_cfg["seed"],
    )

    bootstrap_clusters = group_records_by_control_family(records)

    critic.fit(
        inputs,
        labels,
        records=records,
        coverage_mode="pilot",
        bootstrap_clusters=bootstrap_clusters,
    )

    return critic


def _make_encoder(probe_cfg):
    """Create EnhancedEncoder with SMTR base features + record metadata."""
    from smtr.router.transfer_features import HashingTransferFeatureEncoder
    base = HashingTransferFeatureEncoder(
        n_features=probe_cfg["n_features"],
        feature_block=probe_cfg["feature_block"],
    )
    return EnhancedEncoder(base)


def _train_ranking(inputs, labels, records, probe_cfg):
    """Pairwise ranking loss training with Ridge."""
    encoder = _make_encoder(probe_cfg)

    # Encode features (with record metadata)
    X_sparse = encoder.encode_batch(inputs, records=records)
    X = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else np.asarray(X_sparse)
    y = np.array([_get_tau(r) for r in records])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sample weights: upweight informative records (τ≠0) by 5x
    weights = np.where(y != 0, 5.0, 1.0)

    # Train Ridge for τ regression
    tau_model = Ridge(alpha=0.01)
    tau_model.fit(X_scaled, y, sample_weight=weights)

    return RankingProbe(encoder, tau_model, scaler)


def _train_hybrid(inputs, labels, records, probe_cfg):
    """Hybrid: ranking + regression + sign classification."""
    encoder = _make_encoder(probe_cfg)

    # Encode features (with record metadata)
    X_sparse = encoder.encode_batch(inputs, records=records)
    X = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else np.asarray(X_sparse)
    y_tau = np.array([_get_tau(r) for r in records])
    y_sign = np.sign(y_tau)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sample weights: upweight informative records (τ≠0) by 5x
    weights = np.where(y_tau != 0, 5.0, 1.0)

    # Train Ridge for τ regression (L_tau component)
    tau_model = Ridge(alpha=0.01)
    tau_model.fit(X_scaled, y_tau, sample_weight=weights)

    return RankingProbe(encoder, tau_model, scaler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SMTR critic probe")
    parser.add_argument(
        "--split",
        required=True,
        help="Split name (e.g. in_distribution, memory_holdout, task_holdout)",
    )
    parser.add_argument(
        "--training_mode",
        choices=["regression", "ranking", "hybrid"],
        default="hybrid",
        help="Training mode: regression (baseline), ranking (pairwise), or hybrid (recommended)",
    )
    args = parser.parse_args()

    config = _load_config()
    data_cfg = config["data"]
    probe_cfg = config["probe"]

    split_name = args.split
    training_mode = args.training_mode

    print("=" * 60)
    print(f"MARBLE Critic Training [{split_name}] — Mode: {training_mode}")
    print("=" * 60)

    # ── Paths ──
    split_dir = _THIS_DIR / "splits" / split_name
    # Use raw (non-resampled) train data to avoid duplicate overfitting
    raw_train_path = split_dir / "train_raw.jsonl"
    balanced_train_path = split_dir / "train.jsonl"
    # Prefer raw if available
    train_path = raw_train_path if raw_train_path.exists() else balanced_train_path
    output_path = split_dir / "smtr_probe.joblib"
    sign_clf_path = split_dir / "sign_classifier.joblib"
    memory_pool_path = _PROJECT_ROOT / data_cfg["memory_pool_path"]

    print(f"\n  Training data: {train_path.name} ({train_path})")
    print(f"  Memory pool: {memory_pool_path}")

    # ── Load training records ──
    train_records = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            train_records.append(json.loads(line))
    train_valid = [r for r in train_records if r.get("valid", False)]

    # Deduplicate by (edge_id, generation_seed)
    seen = set()
    deduped = []
    for r in train_valid:
        key = (r.get("edge_id", ""), r.get("generation_seed", -1))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    if len(deduped) < len(train_valid):
        print(f"  Dedup: {len(train_valid)} → {len(deduped)} unique records")
    train_valid = deduped
    print(f"\n  Training records: {len(train_valid)}")

    # ── Build training data ──
    from smtr.router.transfer_features import build_training_data_from_records

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

    # ── Train based on mode ──
    print(f"\n  Training with mode: {training_mode}")

    if training_mode == "regression":
        critic = _train_regression(inputs, labels, records, probe_cfg, memory_pool_path)
    elif training_mode == "ranking":
        critic = _train_ranking(inputs, labels, records, probe_cfg)
    else:  # hybrid
        critic = _train_hybrid(inputs, labels, records, probe_cfg)

    # Save model
    if hasattr(critic, 'save'):
        critic.save(output_path)
    else:
        joblib.dump(critic, output_path)
    print(f"\n  Saved model: {output_path}")

    # ── Train sign classifier (for hybrid mode) ──
    sign_metrics = {}
    if training_mode == "hybrid":
        print("\n  Training sign classifier z = sign(τ)...")
        encoder = critic.encoder
        X_sparse = encoder.encode_batch(inputs, records=records)
        X = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else np.asarray(X_sparse)
        z_labels = np.array([np.sign(_get_tau(r)) for r in records])
        z_mapped = z_labels + 1  # {-1,0,1} → {0,1,2}

        sign_clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=probe_cfg["seed"],
            C=1.0,
        )
        sign_clf.fit(X, z_mapped)

        z_pred = sign_clf.predict(X)
        sign_acc = (z_pred == z_mapped).mean()
        sign_metrics = {
            "train_accuracy": round(float(sign_acc), 4),
            "label_distribution": {
                "negative": int((z_mapped == 0).sum()),
                "neutral": int((z_mapped == 1).sum()),
                "positive": int((z_mapped == 2).sum()),
            },
        }
        print(f"    Train accuracy: {sign_acc:.4f}")

        joblib.dump(sign_clf, sign_clf_path)
        print(f"  Saved sign classifier: {sign_clf_path}")

    # ── Save metrics ──
    metrics = {
        "split_name": split_name,
        "training_mode": training_mode,
        "train_records": len(records),
        "label_distribution": dict(Counter(labels)),
        "n_features": probe_cfg["n_features"],
        "feature_block": probe_cfg["feature_block"],
        "seed": probe_cfg["seed"],
        "sign_classifier": sign_metrics,
    }

    metrics_path = split_dir / "probe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Saved metrics: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
