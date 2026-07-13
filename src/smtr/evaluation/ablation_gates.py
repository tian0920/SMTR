"""Optional gate classes for SMTR evaluation ablations."""

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
            effect_condition_passed=effect_passed,
            risk_condition_passed=None,
        )
