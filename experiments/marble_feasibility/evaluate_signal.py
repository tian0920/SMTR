"""Evaluate SMTR probe signal with informative-pair ranking.

Metrics:
  1. Informative ranking: pairwise accuracy on pairs where τ(m1) ≠ τ(m2)
     (excludes τ=0 vs τ=0 — only measures discrimination ability)
  2. Full ranking: standard pairwise (for comparison)
  3. Sign classification: z = sign(τ) accuracy
  4. Transfer identification: 4-class accuracy
  5. Baselines: random + outcome-only

Diagnostic outputs:
  - Label distribution (positive/negative/neutral %)
  - Prediction distribution (mean, std, unique values)
  - Ranking only on informative pairs

Outputs:
  - data/evaluation_results.json
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
    """Compute τ = Y_expose - Y_withhold from a paired record."""
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    return y_expose - y_withhold


def _pairwise_ranking(
    predictions: np.ndarray,
    true_values: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
    *,
    informative_only: bool = False,
) -> tuple[float, int]:
    """Compute pairwise ranking accuracy.

    If informative_only=True, sample only from records where |τ|>0
    and only count pairs where τ(i) ≠ τ(j).
    Returns (accuracy, n_informative_pairs).
    """
    n = len(predictions)

    if informative_only:
        # Only sample from informative records (|τ| > 0)
        info_idx = np.where(true_values != 0)[0]
        if len(info_idx) < 2:
            return 0.5, 0
        pairs = rng.choice(info_idx, size=(n_samples, 2), replace=True)
    else:
        pairs = rng.randint(0, n, size=(n_samples, 2))

    correct = 0
    total = 0

    for i, j in pairs:
        if true_values[i] != true_values[j]:
            total += 1
            if true_values[i] > true_values[j]:
                if predictions[i] > predictions[j]:
                    correct += 1
            elif true_values[i] < true_values[j]:
                if predictions[i] < predictions[j]:
                    correct += 1

    return correct / max(total, 1), total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SMTR probe signal")
    parser.add_argument(
        "--split",
        default=None,
        help="Split name (e.g. in_distribution, memory_holdout, task_holdout). "
             "If provided, reads from splits/<split>/test.jsonl and loads probe "
             "from splits/<split>/smtr_probe.joblib.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Training mode name. If provided, saves results to "
             "splits/<split>/eval_<mode>.json instead of evaluation_results.json.",
    )
    args = parser.parse_args()

    config = _load_config()
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    split_name = args.split
    training_mode = args.mode

    print("=" * 60)
    if split_name:
        print(f"MARBLE Feasibility — Signal Evaluation [{split_name}]")
    else:
        print("MARBLE Feasibility Test — Signal Evaluation (Informative)")
    print("=" * 60)

    # ── Paths (split-aware) ──
    if split_name:
        split_dir = _THIS_DIR / "splits" / split_name
        test_path = split_dir / "test.jsonl"
        probe_path = split_dir / "smtr_probe.joblib"
        sign_clf_path = split_dir / "sign_classifier.joblib"
        if training_mode:
            results_path = split_dir / f"eval_{training_mode}.json"
        else:
            results_path = split_dir / "evaluation_results.json"
    else:
        test_path = _PROJECT_ROOT / data_cfg["test_records_path"]
        probe_path = _THIS_DIR / "data" / "smtr_probe.joblib"
        sign_clf_path = _THIS_DIR / "data" / "sign_classifier.joblib"
        results_path = _THIS_DIR / "data" / "evaluation_results.json"

    memory_pool_path = _PROJECT_ROOT / data_cfg["memory_pool_path"]
    # ── Load test records ──
    print(f"\n  Loading test records: {test_path}")
    test_records = _load_paired_records(test_path)
    valid_records = [r for r in test_records if r.get("valid", False)]
    print(f"  Total test records: {len(test_records)}")
    print(f"  Valid test records: {len(valid_records)}")

    tau_true = np.array([_get_tau_from_record(r) for r in valid_records])
    n_pos = int((tau_true > 0).sum())
    n_neg = int((tau_true < 0).sum())
    n_neu = int((tau_true == 0).sum())
    print(f"  τ distribution: +{n_pos}, -{n_neg}, 0={n_neu}")

    # ── Load SMTR probe ──
    print(f"\n  Loading SMTR probe: {probe_path}")
    from smtr.router.transfer_critic import FourOutcomeTransferCritic
    from smtr.router.transfer_features import build_training_data_from_records

    # Try RankingProbe first, fall back to FourOutcomeTransferCritic
    # Import shared classes so joblib can unpickle them
    from _probe_models import EnhancedEncoder, RankingProbe  # noqa: F401
    probe_data = joblib.load(probe_path)
    if isinstance(probe_data, dict) and probe_data.get('type') == 'ranking_probe':
        # Load RankingProbe wrapper
        _encoder = probe_data['encoder']
        _tau_model = probe_data['tau_model']
        _scaler = probe_data.get('scaler')

        class _LoadedRankingProbe:
            def __init__(self):
                self.encoder = _encoder
                self.tau_model = _tau_model
                self.scaler = _scaler
            def predict_batch(self, inputs, records=None):
                X_sparse = self.encoder.encode_batch(inputs, records=records)
                X = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else np.asarray(X_sparse)
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                tau_hats = self.tau_model.predict(X)
                predictions = []
                for tau in tau_hats:
                    if tau > 0:
                        pred = type('P', (), {
                            'q00_neutral_failure': 0.1, 'q01_negative_transfer': 0.05,
                            'q10_positive_transfer': 0.7, 'q11_neutral_success': 0.15,
                            'tau_hat': float(tau)
                        })()
                    elif tau < 0:
                        pred = type('P', (), {
                            'q00_neutral_failure': 0.15, 'q01_negative_transfer': 0.7,
                            'q10_positive_transfer': 0.05, 'q11_neutral_success': 0.1,
                            'tau_hat': float(tau)
                        })()
                    else:
                        pred = type('P', (), {
                            'q00_neutral_failure': 0.4, 'q01_negative_transfer': 0.1,
                            'q10_positive_transfer': 0.1, 'q11_neutral_success': 0.4,
                            'tau_hat': float(tau)
                        })()
                    predictions.append(pred)
                return predictions

        smtr_probe = _LoadedRankingProbe()
        encoder = smtr_probe.encoder
        _is_ranking_probe = True
    else:
        smtr_probe = FourOutcomeTransferCritic.load(probe_path)
        encoder = smtr_probe.encoder
        _is_ranking_probe = False

    # ── SMTR predictions ──
    print("  Computing SMTR probe predictions...")
    eval_data = build_training_data_from_records(valid_records, memory_pool_path)
    eval_inputs = [item for item, _, _ in eval_data]
    eval_records = [rec for _, _, rec in eval_data]
    tau_aligned = np.array([_get_tau_from_record(r) for r in eval_records])

    # Pass records for EnhancedEncoder (ranking probe)
    if _is_ranking_probe:
        predictions_raw = smtr_probe.predict_batch(eval_inputs, records=eval_records)
    else:
        predictions_raw = smtr_probe.predict_batch(eval_inputs)
    smtr_preds = np.array([p.tau_hat for p in predictions_raw])

    # ── Diagnostic: prediction distribution ──
    pred_std = float(np.std(smtr_preds))
    pred_mean = float(np.mean(smtr_preds))
    pred_min = float(np.min(smtr_preds))
    pred_max = float(np.max(smtr_preds))
    unique_vals = len(set(round(t, 6) for t in smtr_preds))

    print(f"\n  Prediction distribution:")
    print(f"    mean={pred_mean:.4f}, std={pred_std:.4f}")
    print(f"    min={pred_min:.4f}, max={pred_max:.4f}")
    print(f"    unique values: {unique_vals}")

    # ── Label distribution ──
    true_labels = [r.get("label", "unknown") for r in eval_records]
    label_dist = Counter(true_labels)
    print(f"\n  Label distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"    {label}: {count} ({count/len(true_labels):.1%})")

    # ── Informative-pair ranking ──
    rng = np.random.RandomState(eval_cfg["seed"])
    informative_only = eval_cfg.get("informative_only_ranking", True)

    print(f"\n  Computing {'informative-only' if informative_only else 'full'} ranking...")
    smtr_ranking, n_info_pairs = _pairwise_ranking(
        smtr_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng,
        informative_only=informative_only,
    )
    print(f"  SMTR ranking: {smtr_ranking:.4f} ({n_info_pairs} informative pairs)")

    # Random baseline (informative)
    print("\n  Computing random baseline...")
    random_preds = rng.randn(len(tau_aligned))
    random_ranking, _ = _pairwise_ranking(
        random_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng,
        informative_only=informative_only,
    )
    print(f"  Random ranking: {random_ranking:.4f}")

    # Outcome-only baseline
    print("\n  Computing outcome-only baseline...")
    outcome_preds = np.array([
        1.0 if r.get("share", {}).get("team_success") else 0.0
        for r in eval_records
    ])
    outcome_ranking, _ = _pairwise_ranking(
        outcome_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng,
        informative_only=informative_only,
    )
    print(f"  Outcome-only ranking: {outcome_ranking:.4f}")

    # Outcome-only full ranking (for SMTR comparison)
    outcome_full_ranking, _ = _pairwise_ranking(
        outcome_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng,
        informative_only=False,
    )
    print(f"  Outcome-only full ranking: {outcome_full_ranking:.4f}")

    # ── Full ranking (for reference) ──
    print("\n  Computing full ranking (reference)...")
    smtr_full_ranking, n_full_pairs = _pairwise_ranking(
        smtr_preds, tau_aligned, eval_cfg["n_pairwise_samples"], rng,
        informative_only=False,
    )
    print(f"  SMTR full ranking: {smtr_full_ranking:.4f} ({n_full_pairs} total informative pairs)")

    # ── Sign classifier evaluation ──
    sign_accuracy = None
    sign_pred_dist = {}
    if sign_clf_path.exists():
        print("\n  Evaluating sign classifier...")
        sign_clf = joblib.load(sign_clf_path)
        if _is_ranking_probe:
            X_test_sparse = encoder.encode_batch(eval_inputs, records=eval_records)
        else:
            X_test_sparse = encoder.encode_batch(eval_inputs)
        X_test = X_test_sparse.toarray() if hasattr(X_test_sparse, 'toarray') else np.asarray(X_test_sparse)
        z_pred = sign_clf.predict(X_test)
        z_true = np.array([np.sign(t) + 1 for t in tau_aligned])  # {-1,0,1}→{0,1,2}
        sign_accuracy = float((z_pred == z_true).mean())

        z_names = ["negative", "neutral", "positive"]
        sign_pred_dist = {z_names[k]: int((z_pred == k).sum()) for k in range(3)}
        print(f"  Sign classifier accuracy: {sign_accuracy:.4f}")
        print(f"  Sign predictions: {sign_pred_dist}")

    # ── Transfer identification accuracy ──
    print("\n  Computing transfer identification accuracy...")
    smtr_labels = []
    for pred in predictions_raw:
        probs = {
            "neutral_failure": pred.q00_neutral_failure,
            "negative_transfer": pred.q01_negative_transfer,
            "positive_transfer": pred.q10_positive_transfer,
            "neutral_success": pred.q11_neutral_success,
        }
        smtr_labels.append(max(probs, key=probs.get))

    correct = sum(1 for p, t in zip(smtr_labels, true_labels) if p == t)
    smtr_identification_acc = correct / len(true_labels) if true_labels else 0.0
    print(f"  SMTR 4-class identification accuracy: {smtr_identification_acc:.4f}")

    # ── Tau correlation (tertiary metric) ──
    informative_mask = tau_aligned != 0
    if informative_mask.sum() > 2:
        tau_corr = float(np.corrcoef(
            smtr_preds[informative_mask],
            tau_aligned[informative_mask],
        )[0, 1])
    else:
        tau_corr = 0.0
    print(f"\n  Tau correlation (informative): {tau_corr:.4f}")

    results = {
        "split_name": split_name,
        "test_records": len(test_records),
        "valid_records": len(valid_records),
        "tau_distribution": {
            "positive": n_pos,
            "negative": n_neg,
            "neutral": n_neu,
        },
        "prediction_distribution": {
            "mean": round(pred_mean, 4),
            "std": round(pred_std, 4),
            "min": round(pred_min, 4),
            "max": round(pred_max, 4),
            "unique_values": unique_vals,
        },
        "label_distribution": dict(label_dist),
        "smtr_probe": {
            "informative_ranking": round(smtr_ranking, 4),
            "full_ranking": round(smtr_full_ranking, 4),
            "identification_accuracy": round(smtr_identification_acc, 4),
            "tau_correlation": round(tau_corr, 4),
            "n_informative_pairs": n_info_pairs,
        },
        "random_baseline": {
            "pairwise_ranking": round(random_ranking, 4),
        },
        "outcome_only_baseline": {
            "pairwise_ranking": round(outcome_ranking, 4),
            "full_ranking": round(outcome_full_ranking, 4),
        },
        "sign_classifier": {
            "accuracy": round(sign_accuracy, 4) if sign_accuracy is not None else None,
            "prediction_distribution": sign_pred_dist,
        },
        "improvement": {
            "vs_random": round(smtr_ranking - random_ranking, 4),
            "vs_outcome_only": round(smtr_ranking - outcome_ranking, 4),
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # ── Summary ──
    print("\n  Evaluation Summary:")
    print(f"    SMTR informative ranking: {smtr_ranking:.4f}")
    print(f"    Random baseline:          {random_ranking:.4f}")
    print(f"    Outcome-only baseline:    {outcome_ranking:.4f}")
    print(f"    SMTR vs random:           {smtr_ranking - random_ranking:+.4f}")
    print(f"    SMTR vs outcome-only:     {smtr_ranking - outcome_ranking:+.4f}")
    print(f"    τ pred std:               {pred_std:.4f}")
    if sign_accuracy is not None:
        print(f"    Sign classifier acc:      {sign_accuracy:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
