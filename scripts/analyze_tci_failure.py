#!/usr/bin/env python3
"""TCI failure mode analysis (Task 6).

Analyzes prediction failures in TCI models:
  - False positive: predicted direction > 0, truth <= 0
  - False negative: predicted direction <= 0, truth > 0
  - Neutral confusion: truth == 0 (model struggles with neutral cases)

This script loads a trained critic and TCI pairs,
then categorizes prediction errors per intervention contrast.

Output:
  artifacts/tci_failure_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"

TRAIN = PAIRED / "train" / "paired_records.jsonl"
POOL = ROOT / "real_data" / "database_v1" / "memory_pool.jsonl"

CONTRASTS = INTERV / "intervention_contrasts.jsonl"
PERTURB = INTERV / "perturbations.json"

CRITIC_PATH = Path("outputs/tci_smtr_c_full.joblib")
OUTPUT_PATH = Path("artifacts/tci_failure_analysis.json")


def _load_contrasts(path: Path):
    """Load intervention contrasts."""
    from smtr.intervention.intervention_contrast import InterventionContrast
    contrasts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            contrasts.append(InterventionContrast.from_dict(json.loads(line)))
    return contrasts


def main() -> None:
    from smtr.marble.training import _build_tci_inputs_for_critic
    from smtr.router.transfer_critic import FourOutcomeTransferCritic

    # Load critic.
    print(f"Loading critic from {CRITIC_PATH}...")
    critic = FourOutcomeTransferCritic.load(CRITIC_PATH)
    print(f"  Mode: {critic.training_mode}")
    print(f"  Observational examples: {critic.n_observational_examples}")
    print(f"  TCI examples: {critic.n_tci_examples}")

    # Load contrasts for ground truth.
    print(f"\nLoading contrasts from {CONTRASTS}...")
    contrasts = _load_contrasts(CONTRASTS)
    print(f"  Contrasts: {len(contrasts)}")

    # Build TCI tuples with full CandidateExposureInput objects.
    print("Building TCI tuples...")
    tci_tuples = _build_tci_inputs_for_critic(
        tci_contrasts_path=CONTRASTS,
        perturbations_manifest_path=PERTURB,
        paired_records_path=TRAIN,
        memory_pool_path=POOL,
    )
    print(f"  TCI pairs: {len(tci_tuples)}")

    # Analyze failures.
    print("\nAnalyzing failures...")
    false_positive_examples = []
    false_negative_examples = []
    neutral_confusion_examples = []

    by_factor = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "false_positive": 0, "false_negative": 0, "neutral_confusion": 0,
    })

    n_pairs = len(tci_tuples)
    all_pred_margins = []
    all_true_dirs = []

    for i, (inp_orig, inp_pert, direction, ct) in enumerate(tci_tuples):
        # Get critic predictions.
        pred_orig = critic.predict(inp_orig)
        pred_pert = critic.predict(inp_pert)

        # Transfer utility: s_theta(m) = q10 - q01.
        s_orig = pred_orig.q10_positive_transfer - pred_orig.q01_negative_transfer
        s_pert = pred_pert.q10_positive_transfer - pred_pert.q01_negative_transfer

        # Predicted direction: sign of (s_orig - s_pert).
        pred_margin = s_orig - s_pert
        pred_dir = np.sign(pred_margin)
        true_dir = float(direction)

        all_pred_margins.append(pred_margin)
        all_true_dirs.append(true_dir)

        # Map TCI tuple index back to contrast if possible.
        perturbation_type = ct  # contrast_type from TCI tuple

        # Categorize errors.
        factor_stats = by_factor[perturbation_type]
        factor_stats["total"] += 1

        if true_dir != 0 and pred_dir == true_dir:
            factor_stats["correct"] += 1
        elif pred_dir > 0 and true_dir <= 0:
            # False positive: predicted positive, truth non-positive.
            factor_stats["false_positive"] += 1
            if len(false_positive_examples) < 10:
                false_positive_examples.append({
                    "index": i,
                    "perturbation_type": perturbation_type,
                    "predicted_direction": int(pred_dir),
                    "true_direction": int(true_dir),
                    "pred_margin": round(float(pred_margin), 4),
                })
        elif pred_dir <= 0 and true_dir > 0:
            # False negative: predicted non-positive, truth positive.
            factor_stats["false_negative"] += 1
            if len(false_negative_examples) < 10:
                false_negative_examples.append({
                    "index": i,
                    "perturbation_type": perturbation_type,
                    "predicted_direction": int(pred_dir),
                    "true_direction": int(true_dir),
                    "pred_margin": round(float(pred_margin), 4),
                })
        elif true_dir == 0:
            # Neutral confusion: truth is neutral.
            factor_stats["neutral_confusion"] += 1
            if len(neutral_confusion_examples) < 10:
                neutral_confusion_examples.append({
                    "index": i,
                    "perturbation_type": perturbation_type,
                    "predicted_direction": int(pred_dir),
                    "true_direction": int(true_dir),
                    "pred_margin": round(float(pred_margin), 4),
                })

    # Compute overall accuracy (on non-neutral pairs).
    all_true_dirs_arr = np.array(all_true_dirs)
    all_pred_margins_arr = np.array(all_pred_margins)

    nonzero_mask = all_true_dirs_arr != 0
    if nonzero_mask.sum() > 0:
        pred_dirs_nz = np.sign(all_pred_margins_arr[nonzero_mask])
        true_dirs_nz = all_true_dirs_arr[nonzero_mask]
        correct = (pred_dirs_nz == true_dirs_nz)
        accuracy = float(correct.mean())
        n_nonzero = int(nonzero_mask.sum())
    else:
        accuracy = 0.0
        n_nonzero = 0

    # Neutral statistics.
    n_neutral = int((all_true_dirs_arr == 0).sum())

    # Build factor analysis.
    factor_analysis = {}
    for factor, stats in sorted(by_factor.items()):
        total = stats["total"]
        correct = stats["correct"]
        factor_analysis[factor] = {
            "n_pairs": total,
            "n_correct": correct,
            "accuracy": round(correct / max(1, total), 4),
            "false_positive": stats["false_positive"],
            "false_negative": stats["false_negative"],
            "neutral_confusion": stats["neutral_confusion"],
        }

    # Combine results.
    results = {
        "overall": {
            "n_pairs": n_pairs,
            "n_nonzero_pairs": n_nonzero,
            "n_neutral_pairs": n_neutral,
            "accuracy": round(accuracy, 4),
            "false_positive": {
                "count": len([s for s in by_factor.values()
                              for _ in range(s["false_positive"])]),
                "examples": false_positive_examples,
            },
            "false_negative": {
                "count": len([s for s in by_factor.values()
                              for _ in range(s["false_negative"])]),
                "examples": false_negative_examples,
            },
            "neutral_confusion": {
                "count": len([s for s in by_factor.values()
                              for _ in range(s["neutral_confusion"])]),
                "examples": neutral_confusion_examples,
            },
        },
        "by_factor": factor_analysis,
        "critic_info": {
            "training_mode": critic.training_mode,
            "n_observational": critic.n_observational_examples,
            "n_tci": critic.n_tci_examples,
        },
    }

    # Save results.
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary.
    print(f"\n{'='*60}")
    print("Failure Analysis Summary")
    print(f"{'='*60}")
    print(f"Total TCI pairs: {n_pairs}")
    print(f"Non-zero pairs: {n_nonzero}")
    print(f"Neutral pairs: {n_neutral}")
    print(f"Accuracy (non-zero): {accuracy:.4f}")

    fp_count = sum(s["false_positive"] for s in by_factor.values())
    fn_count = sum(s["false_negative"] for s in by_factor.values())
    nc_count = sum(s["neutral_confusion"] for s in by_factor.values())
    print(f"\nFalse positives: {fp_count}")
    print(f"False negatives: {fn_count}")
    print(f"Neutral confusion: {nc_count}")

    print(f"\nBy Factor:")
    for factor, stats in sorted(factor_analysis.items()):
        print(f"  {factor:30s}: n={stats['n_pairs']:3d}, "
              f"acc={stats['accuracy']:.3f}, "
              f"fp={stats['false_positive']:2d}, "
              f"fn={stats['false_negative']:2d}, "
              f"neutral={stats['neutral_confusion']:2d}")


if __name__ == "__main__":
    main()
