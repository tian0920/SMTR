"""Optional gate classes for the retained SMTR ablations."""

from __future__ import annotations

from dataclasses import dataclass

from smtr.router.gate_protocol import GateDecision, TransferPointEstimate


@dataclass(frozen=True)
class EffectOnlyGate:
    """Ablation gate that ignores negative-transfer risk."""

    gate_name: str = "effect_only_smtr"

    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
        effect_passed = estimate.tau_mean > 0.0
        return GateDecision(
            share=effect_passed,
            reason="shared" if effect_passed else "tau_mean_nonpositive",
            gate_name=self.gate_name,
            effect_condition_status="passed" if effect_passed else "failed",
            risk_condition_status="not_applicable",
        )

@dataclass(frozen=True)
class FactualSuccessGate:
    """Share iff the factual share-success probability clears a fixed threshold."""

    threshold: float
    gate_name: str = "factual_success_smtr"

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
        probability = getattr(estimate, "p_share_success", None)
        if probability is None:
            raise TypeError("FactualSuccessGate requires p_share_success estimates")
        passed = float(probability) >= self.threshold
        return GateDecision(
            share=passed,
            reason="shared" if passed else "factual_success_below_threshold",
            gate_name=self.gate_name,
            effect_condition_status="not_applicable",
            risk_condition_status="not_applicable",
        )
