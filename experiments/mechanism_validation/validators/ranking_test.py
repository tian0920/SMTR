"""Rank loss necessity validator (Task 4).

Validates that the ranking objective (L_rank) is necessary for
learning pairwise transfer ordering.

Four models compared:
  A: L_obs only
  B: L_obs + L_tau (effect only, no rank)
  C: L_obs + L_rank (rank only, no effect)
  D: L_obs + L_rank + L_tau (full)

Acceptance: rank > obs (L_obs + L_rank > L_obs)
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .base import BaseValidator, ValidationResult


class RankLossValidator(BaseValidator):
    """Validate that ranking objective is necessary."""
    
    def validate(self) -> ValidationResult:
        """Run rank loss ablation.
        
        Trains four models with different loss configurations and
        compares pairwise accuracy.
        """
        t0 = time.time()
        
        from smtr.marble.training import train_critic
        from smtr.marble.paired_outcomes import paired_record_label
        from smtr.router.tci_effect_builder import build_tci_effect_examples
        from smtr.router.transfer_critic import FourOutcomeTransferCritic
        from smtr.router.transfer_features import (
            HashingTransferFeatureEncoder,
            load_paired_records_with_metadata,
        )
        from smtr.intervention.intervention_contrast import InterventionContrast
        
        tci_tuples = self._load_tci_tuples()
        common = self._get_common_train_kwargs()
        
        out_dir = self.project_root / "outputs" / "mechanism_validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load contrasts and build effect batch.
        contrasts = []
        for line in self._get_contrasts_path().read_text().splitlines():
            if line.strip():
                contrasts.append(InterventionContrast.from_dict(
                    json.loads(line)
                ))
        
        encoder = HashingTransferFeatureEncoder(
            n_features=common["n_features"],
            feature_block=common.get("feature_block", "full"),
        )
        effect_batch = build_tci_effect_examples(
            contrasts, encoder, tci_inputs=tci_tuples,
        )
        
        results = {}
        
        # ---- Model A: L_obs only ----
        print("  Training Model A: L_obs only...")
        out_a = out_dir / "rank_obs.joblib"
        train_critic(output_path=out_a, critic_mode="flat", **common)
        cA = FourOutcomeTransferCritic.load(out_a)
        results["L_obs"] = self._evaluate_pairwise(cA, tci_tuples)
        
        # ---- Model B: L_obs + L_tau (effect only) ----
        print("  Training Model B: L_obs + L_tau...")
        out_b = out_dir / "rank_obs_tau.joblib"
        cB = FourOutcomeTransferCritic(
            n_features=common["n_features"],
            n_bootstrap=common["n_bootstrap"],
            seed=common["seed"],
            critic_mode="flat",
        )
        train_data = load_paired_records_with_metadata(
            common["train_records_path"], common["memory_pool_path"]
        )
        obs_inputs = [item for item, _, _ in train_data]
        obs_labels = [paired_record_label(rec) for _, _, rec in train_data]
        cB.fit(
            obs_inputs, obs_labels,
            coverage_mode=common["coverage_mode"],
            tci_effect_batch=effect_batch,
        )
        results["L_obs_tau"] = self._evaluate_pairwise(cB, tci_tuples)
        
        # ---- Model C: L_obs + L_rank ----
        print("  Training Model C: L_obs + L_rank...")
        out_c = out_dir / "rank_obs_rank.joblib"
        train_critic(
            output_path=out_c,
            critic_mode="flat",
            tci_contrasts_path=self._get_contrasts_path(),
            tci_perturbations_manifest_path=self._get_perturbations_path(),
            tci_paired_records_path=self._get_train_path(),
            **common,
        )
        cC = FourOutcomeTransferCritic.load(out_c)
        results["L_obs_rank"] = self._evaluate_pairwise(cC, tci_tuples)
        
        # ---- Model D: Full (L_obs + L_rank + L_tau) ----
        print("  Training Model D: Full (L_obs + L_rank + L_tau)...")
        out_d = out_dir / "rank_full.joblib"
        train_critic(
            output_path=out_d,
            critic_mode="flat",
            tci_contrasts_path=self._get_contrasts_path(),
            tci_perturbations_manifest_path=self._get_perturbations_path(),
            tci_paired_records_path=self._get_train_path(),
            tci_effect_batch=effect_batch,
            **common,
        )
        cD = FourOutcomeTransferCritic.load(out_d)
        results["full"] = self._evaluate_pairwise(cD, tci_tuples)
        
        # ---- Evaluate acceptance criteria ----
        acc_obs = results["L_obs"]["pairwise_accuracy"]
        acc_rank = results["L_obs_rank"]["pairwise_accuracy"]
        
        passed = acc_rank > acc_obs
        
        duration = time.time() - t0
        
        metrics = {
            model: {
                "pairwise_accuracy": r["pairwise_accuracy"],
                "pairwise_margin": r.get("pairwise_margin", 0.0),
                "n_pairs": r.get("n_pairs", 0),
            }
            for model, r in results.items()
        }
        
        message = (
            f"L_obs={acc_obs:.4f}, L_obs+L_rank={acc_rank:.4f}. "
            f"Rank > Obs: {passed}"
        )
        
        return ValidationResult(
            name="rank_loss",
            passed=passed,
            metrics=metrics,
            message=message,
            duration_seconds=duration,
        )
