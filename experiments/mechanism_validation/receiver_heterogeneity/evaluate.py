"""Evaluate Global vs SMTR models on receiver heterogeneity test.

Metrics:
  1. Pearson correlation: corr(τ̂, τ_true)
  2. Sign accuracy: P(sign(τ̂) == sign(τ_true))
  3. Pairwise ranking: P(τ̂_i > τ̂_j | τ_i > τ_j)
  4. Receiver permutation test: performance drop when receiver is shuffled

Outputs:
  artifacts/evaluation_results.npz
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

from train_global import _load_config


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Run model inference."""
    return model.predict(X)


def _pearson(preds: np.ndarray, true: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if np.std(preds) < 1e-8 or np.std(true) < 1e-8:
        return 0.0
    return float(np.corrcoef(preds, true)[0, 1])


def _sign_accuracy(preds: np.ndarray, true: np.ndarray) -> float:
    """Compute sign prediction accuracy."""
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
    """Compute pairwise ranking accuracy.

    For randomly sampled pairs (i,j), check if the ordering of
    predictions matches the ordering of ground truth.
    """
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


def _permutation_test(
    model,
    mem_features: np.ndarray,
    recv_features: np.ndarray,
    tau_true: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict:
    """Run receiver permutation test.

    Shuffle receiver features while keeping memory features fixed.
    Measure the drop in SMTR model performance.
    """
    rng = np.random.RandomState(seed)

    # Normal performance.
    combined = np.concatenate([mem_features, recv_features], axis=1)
    normal_preds = _predict(model, combined)
    normal_corr = _pearson(normal_preds, tau_true)
    normal_sign = _sign_accuracy(normal_preds, tau_true)

    # Shuffled performance.
    corr_drops = []
    sign_drops = []
    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(recv_features))
        shuffled_recv = recv_features[perm_idx]
        shuffled_combined = np.concatenate(
            [mem_features, shuffled_recv], axis=1,
        )
        shuffled_preds = _predict(model, shuffled_combined)
        shuffled_corr = _pearson(shuffled_preds, tau_true)
        shuffled_sign = _sign_accuracy(shuffled_preds, tau_true)
        corr_drops.append(normal_corr - shuffled_corr)
        sign_drops.append(normal_sign - shuffled_sign)

    return {
        "normal_pearson": normal_corr,
        "normal_sign": normal_sign,
        "shuffled_pearson_mean": float(np.mean([
            normal_corr - d for d in corr_drops
        ])),
        "shuffled_sign_mean": float(np.mean([
            normal_sign - d for d in sign_drops
        ])),
        "pearson_drop_mean": float(np.mean(corr_drops)),
        "pearson_drop_std": float(np.std(corr_drops)),
        "sign_drop_mean": float(np.mean(sign_drops)),
        "sign_drop_std": float(np.std(sign_drops)),
    }


def main() -> None:
    config = _load_config()
    eval_cfg = config["evaluation"]
    seed = config["environment"]["seed"]
    rng = np.random.RandomState(seed)

    print("=" * 60)
    print("Receiver Heterogeneity Stress Test — Evaluation")
    print("=" * 60)

    # Load test data.
    artifacts_dir = _THIS_DIR / "artifacts"
    test_data = np.load(artifacts_dir / "test_data.npz")

    mem_test = test_data["mem_features"]          # (n, 16)
    recv_test = test_data["recv_features"]        # (n, 16)
    combined_test = test_data["combined_features"]  # (n, 32)
    tau_true = test_data["tau_true"]              # (n,)

    print(f"  Test samples: {len(tau_true)}")
    print(f"  τ distribution: "
          f"+{(tau_true > 0).sum()}/-{(tau_true < 0).sum()}")

    # Load models.
    global_model = joblib.load(artifacts_dir / "global_model.joblib")
    recv_model = joblib.load(artifacts_dir / "receiver_model.joblib")

    # Evaluate Global model.
    print("\n  Evaluating Global model...")
    global_preds = _predict(global_model, mem_test)
    global_pearson = _pearson(global_preds, tau_true)
    global_sign = _sign_accuracy(global_preds, tau_true)
    global_ranking = _pairwise_ranking(
        global_preds, tau_true,
        eval_cfg["n_pairwise_samples"], rng,
    )
    print(f"    Pearson:  {global_pearson:.4f}")
    print(f"    Sign:     {global_sign:.4f}")
    print(f"    Ranking:  {global_ranking:.4f}")

    # Evaluate SMTR model.
    print("\n  Evaluating SMTR model...")
    smtr_preds = _predict(recv_model, combined_test)
    smtr_pearson = _pearson(smtr_preds, tau_true)
    smtr_sign = _sign_accuracy(smtr_preds, tau_true)
    smtr_ranking = _pairwise_ranking(
        smtr_preds, tau_true,
        eval_cfg["n_pairwise_samples"], rng,
    )
    print(f"    Pearson:  {smtr_pearson:.4f}")
    print(f"    Sign:     {smtr_sign:.4f}")
    print(f"    Ranking:  {smtr_ranking:.4f}")

    # Improvement.
    pearson_imp = smtr_pearson - global_pearson
    sign_imp = smtr_sign - global_sign
    ranking_imp = smtr_ranking - global_ranking
    print(f"\n  SMTR improvement over Global:")
    print(f"    Pearson:  +{pearson_imp:.4f}")
    print(f"    Sign:     +{sign_imp:.4f}")
    print(f"    Ranking:  +{ranking_imp:.4f}")

    # Receiver permutation test.
    print(f"\n  Running receiver permutation test "
          f"({eval_cfg['n_permutations']} permutations)...")
    perm_results = _permutation_test(
        recv_model,
        mem_test,
        recv_test,
        tau_true,
        eval_cfg["n_permutations"],
        seed,
    )
    print(f"    Normal Pearson:    {perm_results['normal_pearson']:.4f}")
    print(f"    Shuffled Pearson: {perm_results['shuffled_pearson_mean']:.4f}"
          f" ± {perm_results['pearson_drop_std']:.4f}")
    print(f"    Pearson drop:     {perm_results['pearson_drop_mean']:.4f}")
    print(f"    Normal Sign:      {perm_results['normal_sign']:.4f}")
    print(f"    Shuffled Sign:    {perm_results['shuffled_sign_mean']:.4f}"
          f" ± {perm_results['sign_drop_std']:.4f}")
    print(f"    Sign drop:        {perm_results['sign_drop_mean']:.4f}")

    # Save results.
    out_path = artifacts_dir / "evaluation_results.npz"
    np.savez(
        out_path,
        tau_true=tau_true,
        global_preds=global_preds,
        smtr_preds=smtr_preds,
        global_pearson=global_pearson,
        global_sign=global_sign,
        global_ranking=global_ranking,
        smtr_pearson=smtr_pearson,
        smtr_sign=smtr_sign,
        smtr_ranking=smtr_ranking,
        pearson_imp=pearson_imp,
        sign_imp=sign_imp,
        ranking_imp=ranking_imp,
        perm_normal_pearson=perm_results["normal_pearson"],
        perm_shuffled_pearson=perm_results["shuffled_pearson_mean"],
        perm_pearson_drop=perm_results["pearson_drop_mean"],
        perm_pearson_drop_std=perm_results["pearson_drop_std"],
        perm_normal_sign=perm_results["normal_sign"],
        perm_shuffled_sign=perm_results["shuffled_sign_mean"],
        perm_sign_drop=perm_results["sign_drop_mean"],
        perm_sign_drop_std=perm_results["sign_drop_std"],
    )
    print(f"\n  Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
