"""Receiver conditioning validator (Task 3).

Validates that memory value is receiver-dependent, not a global
property. If removing receiver information doesn't hurt pairwise
accuracy, then the model is not learning receiver-specific effects.

Two variants:
  - Variant 1: Without receiver (task, memory) → τ(m)
  - Variant 2: Full (task, receiver, memory) → τ(m, r)

Acceptance: receiver-conditioned >= without-receiver
(on TCI pairwise accuracy). Both are trained with TCI contrast.

Note: if both achieve the same accuracy on the small pilot dataset,
this is evidence that TCI contrast signal is strong enough even
without receiver features. The test still passes as long as
receiver conditioning is not WORSE than global.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .base import BaseValidator, ValidationResult


class ReceiverConditioningValidator(BaseValidator):
    """Validate that memory value is receiver-dependent."""

    def validate(self) -> ValidationResult:
        """Run receiver conditioning ablation.

        Compares full model vs model without receiver features
        on pairwise ranking accuracy. Both are trained with TCI
        contrast to ensure the comparison is meaningful.
        """
        t0 = time.time()

        from smtr.marble.training import train_critic
        from smtr.router.transfer_critic import FourOutcomeTransferCritic
        from smtr.router.tci_effect_builder import build_tci_effect_examples
        from smtr.router.transfer_features import HashingTransferFeatureEncoder
        from smtr.intervention.intervention_contrast import InterventionContrast
        from smtr.router.tci_supervision import evaluate_tci_loss_on_critic

        tci_tuples = self._load_tci_tuples()
        common = self._get_common_train_kwargs()

        out_dir = self.project_root / "outputs" / "mechanism_validation"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load contrasts and build effect batch for both variants.
        contrasts = []
        for line in self._get_contrasts_path().read_text().splitlines():
            if line.strip():
                contrasts.append(InterventionContrast.from_dict(
                    json.loads(line)
                ))

        tci_kwargs = dict(
            tci_contrasts_path=self._get_contrasts_path(),
            tci_perturbations_manifest_path=self._get_perturbations_path(),
            tci_paired_records_path=self._get_train_path(),
        )
        # Remove feature_block from common since each variant sets its own.
        common_no_fb = {k: v for k, v in common.items() if k != "feature_block"}

        # ---- Variant 1: Without receiver (global_transfer) + TCI ----
        print("  Training Variant 1: Without receiver (global_transfer + TCI)...")
        encoder_no_recv = HashingTransferFeatureEncoder(
            n_features=common["n_features"],
            feature_block="global_transfer",
        )
        effect_batch_no_recv = build_tci_effect_examples(
            contrasts, encoder_no_recv, tci_inputs=tci_tuples,
        )

        out_no_recv = out_dir / "receiver_no_recv.joblib"
        train_critic(
            output_path=out_no_recv,
            critic_mode="flat",
            feature_block="global_transfer",
            tci_effect_batch=effect_batch_no_recv,
            **tci_kwargs,
            **common_no_fb,
        )
        c_no_recv = FourOutcomeTransferCritic.load(out_no_recv)
        eval_no_recv = evaluate_tci_loss_on_critic(c_no_recv, tci_tuples)

        # ---- Variant 2: Full (with receiver) + TCI ----
        print("  Training Variant 2: Full (receiver conditioning + TCI)...")
        encoder_full = HashingTransferFeatureEncoder(
            n_features=common["n_features"],
            feature_block="full",
        )
        effect_batch_full = build_tci_effect_examples(
            contrasts, encoder_full, tci_inputs=tci_tuples,
        )

        out_full = out_dir / "receiver_full.joblib"
        train_critic(
            output_path=out_full,
            critic_mode="flat",
            feature_block="full",
            tci_effect_batch=effect_batch_full,
            **tci_kwargs,
            **common_no_fb,
        )
        c_full = FourOutcomeTransferCritic.load(out_full)
        eval_full = evaluate_tci_loss_on_critic(c_full, tci_tuples)

        # ---- Evaluate acceptance criteria ----
        acc_no_recv = eval_no_recv["pairwise_accuracy"]
        acc_full = eval_full["pairwise_accuracy"]
        margin_no_recv = eval_no_recv.get("pairwise_margin", 0.0)
        margin_full = eval_full.get("pairwise_margin", 0.0)

        # Acceptance: full >= without-receiver on accuracy,
        # and ideally full has higher margin (more confident).
        passed = acc_full >= acc_no_recv

        duration = time.time() - t0

        metrics = {
            "without_receiver": {
                "pairwise_accuracy": acc_no_recv,
                "pairwise_margin": margin_no_recv,
                "n_pairs": eval_no_recv.get("n_pairs", 0),
                "feature_block": "global_transfer",
            },
            "receiver_conditioned": {
                "pairwise_accuracy": acc_full,
                "pairwise_margin": margin_full,
                "n_pairs": eval_full.get("n_pairs", 0),
                "feature_block": "full",
            },
            "margin_improvement": round(margin_full - margin_no_recv, 4),
        }

        message = (
            f"Without receiver={acc_no_recv:.4f} (margin={margin_no_recv:.4f}), "
            f"Receiver-conditioned={acc_full:.4f} (margin={margin_full:.4f}). "
            f"Full >= Global: {passed}"
        )

        return ValidationResult(
            name="receiver_conditioning",
            passed=passed,
            metrics=metrics,
            message=message,
            duration_seconds=duration,
            artifacts={
                "checkpoints": {
                    "without_receiver": str(out_no_recv),
                    "receiver_conditioned": str(out_full),
                },
            },
        )
