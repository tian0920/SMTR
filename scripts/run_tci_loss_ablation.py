#!/usr/bin/env python3
"""TCI loss contribution ablation (Task 4).

Tests the contribution of each loss component:
  Model A: L_obs only (baseline)
  Model B: L_obs + L_rank (TCI ranking)
  Model C: L_obs + L_tau (TCI value)
  Model D: L_obs + L_rank + L_tau (full TCI)

All models use identical:
  - Dataset (observational + intervention)
  - Split (train/val/test)
  - Seed (7)
  - Features (HashingTransferFeatureEncoder, n_features=512)

Metrics:
  - pairwise_accuracy: P(predicted direction matches true direction)
  - utility_correlation: Spearman correlation between predicted utility
    and true contrast direction

Output:
  artifacts/tci_loss_ablation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from smtr.intervention.intervention_contrast import InterventionContrast
from smtr.marble.training import (
    _build_tci_inputs_for_critic,
    train_critic,
)
from smtr.router.tci_effect_builder import build_tci_effect_examples
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import HashingTransferFeatureEncoder


ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = PAIRED / "train" / "paired_records.jsonl"
VAL = PAIRED / "validation" / "paired_records.jsonl"
TEST = PAIRED / "test" / "paired_records.jsonl"
POOL = ROOT / "real_data" / "database_v1" / "memory_pool.jsonl"

CONTRASTS = INTERV / "intervention_contrasts.jsonl"
PERTURB = INTERV / "perturbations.json"


def _load_contrasts(path: Path) -> list[InterventionContrast]:
    """Load intervention contrasts."""
    contrasts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            contrasts.append(InterventionContrast.from_dict(json.loads(line)))
    return contrasts


def _evaluate_pairwise(
    critic: FourOutcomeTransferCritic,
    tci_tuples: list,
) -> dict:
    """Evaluate pairwise ranking and utility correlation.

    Parameters
    ----------
    critic : FourOutcomeTransferCritic
        Trained critic.
    tci_tuples : list
        List of (input_orig, input_pert, direction, contrast_type).

    Returns
    -------
    dict
        Evaluation metrics.
    """
    if not tci_tuples:
        return {
            "pairwise_accuracy": 0.0,
            "utility_correlation": 0.0,
            "n_pairs": 0,
        }

    margins = []
    true_dirs = []

    for (inp_orig, inp_pert, direction, ct) in tci_tuples:
        pred_orig = critic.predict(inp_orig)
        pred_pert = critic.predict(inp_pert)
        s_orig = pred_orig.q10_positive_transfer - pred_orig.q01_negative_transfer
        s_pert = pred_pert.q10_positive_transfer - pred_pert.q01_negative_transfer
        margins.append(s_orig - s_pert)
        true_dirs.append(float(direction))

    margins_arr = np.array(margins)
    dirs_arr = np.array(true_dirs)

    # Pairwise accuracy on non-neutral pairs.
    nonzero_mask = dirs_arr != 0
    if nonzero_mask.sum() > 0:
        pred_dirs = np.sign(margins_arr[nonzero_mask])
        true_dirs_nz = dirs_arr[nonzero_mask]
        correct = (pred_dirs == true_dirs_nz)
        pairwise_accuracy = float(correct.mean())

        # Spearman correlation.
        corr, _ = spearmanr(margins_arr[nonzero_mask], true_dirs_nz)
        utility_correlation = float(corr) if not np.isnan(corr) else 0.0
    else:
        pairwise_accuracy = 0.0
        utility_correlation = 0.0

    return {
        "pairwise_accuracy": round(pairwise_accuracy, 4),
        "utility_correlation": round(utility_correlation, 4),
        "n_pairs": len(tci_tuples),
        "n_nonzero": int(nonzero_mask.sum()),
    }


def main() -> None:
    t0 = time.time()
    print("TCI Loss Contribution Ablation")
    print("=" * 60)

    # Load contrasts and build TCI tuples.
    contrasts = _load_contrasts(CONTRASTS)
    print(f"Contrasts: {len(contrasts)}")

    tci_tuples = _build_tci_inputs_for_critic(
        tci_contrasts_path=CONTRASTS,
        perturbations_manifest_path=PERTURB,
        paired_records_path=TRAIN,
        memory_pool_path=POOL,
    )
    print(f"TCI pairs: {len(tci_tuples)}")

    # Build effect batch for Model C and D.
    encoder = HashingTransferFeatureEncoder(n_features=512, feature_block="full")
    effect_batch = build_tci_effect_examples(
        contrasts, encoder, tci_inputs=tci_tuples
    )
    print(f"Effect examples: {effect_batch.n_examples}")

    common = dict(
        train_records_path=TRAIN,
        validation_records_path=VAL,
        test_records_path=TEST,
        memory_pool_path=POOL,
        seed=7,
        n_bootstrap=11,
        n_features=512,
        feature_block="full",
        coverage_mode="pilot",
    )

    # ---- Model A: L_obs only (baseline) ----
    print(f"\n{'='*60}")
    print("Model A: L_obs (baseline)")
    print(f"{'='*60}")
    out_a = OUT_DIR / "tci_smtr_a_obs.joblib"
    ta = time.time()
    train_critic(output_path=out_a, critic_mode="flat", **common)
    wall_a = round(time.time() - ta, 1)
    cA = FourOutcomeTransferCritic.load(out_a)
    eval_a = _evaluate_pairwise(cA, tci_tuples)
    eval_a["wall_seconds"] = wall_a
    print(f"  Pairwise accuracy: {eval_a['pairwise_accuracy']:.4f}")
    print(f"  Utility correlation: {eval_a['utility_correlation']:.4f}")
    print(f"  Wall time: {wall_a}s")

    # ---- Model B: L_obs + L_rank (TCI ranking) ----
    print(f"\n{'='*60}")
    print("Model B: L_obs + L_rank")
    print(f"{'='*60}")
    out_b = OUT_DIR / "tci_smtr_b_rank.joblib"
    tb = time.time()
    train_critic(
        output_path=out_b,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        **common,
    )
    wall_b = round(time.time() - tb, 1)
    cB = FourOutcomeTransferCritic.load(out_b)
    eval_b = _evaluate_pairwise(cB, tci_tuples)
    eval_b["wall_seconds"] = wall_b
    print(f"  Pairwise accuracy: {eval_b['pairwise_accuracy']:.4f}")
    print(f"  Utility correlation: {eval_b['utility_correlation']:.4f}")
    print(f"  Wall time: {wall_b}s")

    # ---- Model C: L_obs + L_tau (TCI value only, no rank) ----
    # Note: In the unified critic, L_tau is the effect supervision.
    # To isolate L_tau without L_rank, we train observational + effect batch
    # but without TCI rank pairs.
    print(f"\n{'='*60}")
    print("Model C: L_obs + L_tau (value only)")
    print(f"{'='*60}")
    out_c = OUT_DIR / "tci_smtr_c_value.joblib"
    tc = time.time()
    # Train observational + effect (no rank).
    cC = FourOutcomeTransferCritic(
        n_features=512, n_bootstrap=11, seed=7, critic_mode="flat"
    )
    from smtr.marble.paired_outcomes import paired_record_label
    from smtr.router.transfer_features import load_paired_records_with_metadata
    train_data = load_paired_records_with_metadata(TRAIN, POOL)
    obs_inputs = [item for item, _, _ in train_data]
    obs_labels = [paired_record_label(rec) for _, _, rec in train_data]
    cC.fit(
        obs_inputs, obs_labels,
        coverage_mode="pilot",
        tci_effect_batch=effect_batch,
    )
    wall_c = round(time.time() - tc, 1)
    eval_c = _evaluate_pairwise(cC, tci_tuples)
    eval_c["wall_seconds"] = wall_c
    print(f"  Pairwise accuracy: {eval_c['pairwise_accuracy']:.4f}")
    print(f"  Utility correlation: {eval_c['utility_correlation']:.4f}")
    print(f"  Wall time: {wall_c}s")

    # ---- Model D: L_obs + L_rank + L_tau (full TCI) ----
    print(f"\n{'='*60}")
    print("Model D: L_obs + L_rank + L_tau (full)")
    print(f"{'='*60}")
    out_d = OUT_DIR / "tci_smtr_c_full.joblib"
    td = time.time()
    train_critic(
        output_path=out_d,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        tci_effect_batch=effect_batch,
        **common,
    )
    wall_d = round(time.time() - td, 1)
    cD = FourOutcomeTransferCritic.load(out_d)
    eval_d = _evaluate_pairwise(cD, tci_tuples)
    eval_d["wall_seconds"] = wall_d
    print(f"  Pairwise accuracy: {eval_d['pairwise_accuracy']:.4f}")
    print(f"  Utility correlation: {eval_d['utility_correlation']:.4f}")
    print(f"  Wall time: {wall_d}s")

    # ---- Save results ----
    results = {
        "obs": eval_a,
        "rank": eval_b,
        "value": eval_c,
        "full": eval_d,
    }

    output_path = Path("artifacts/tci_loss_ablation.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_wall = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Total wall time: {total_wall}s")

    # Print summary table.
    print(f"\n{'='*60}")
    print("Loss Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Model':12s} {'Pairwise Acc':>14s} {'Utility Corr':>14s} {'Delta Acc':>10s}")
    print(f"{'-'*50}")

    obs_acc = eval_a["pairwise_accuracy"]
    for name, metrics in [("obs", eval_a), ("rank", eval_b),
                          ("value", eval_c), ("full", eval_d)]:
        acc = metrics["pairwise_accuracy"]
        corr = metrics["utility_correlation"]
        delta = acc - obs_acc
        print(f"  {name:8s} {acc:14.4f} {corr:14.4f} {delta:+10.4f}")


if __name__ == "__main__":
    main()
