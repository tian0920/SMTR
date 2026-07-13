"""LCB/UCB gate for the optional Robust-SMTR extension."""

from __future__ import annotations

from dataclasses import dataclass

from smtr.robust.config import RobustSMTRGateConfig
from smtr.robust.estimates import RobustTransferEstimate
from smtr.router.gate_protocol import GateDecision


@dataclass(frozen=True)
class RobustSMTRGate:
    config: RobustSMTRGateConfig
    gate_name: str = "robust_smtr_lcb_ucb"

    def decide(self, estimate: RobustTransferEstimate) -> GateDecision:
        effect_passed = estimate.tau_lcb > 0.0
        risk_passed = estimate.negative_risk_ucb <= self.config.negative_risk_budget
        if not effect_passed:
            reason = "tau_lcb_nonpositive"
        elif not risk_passed:
            reason = "negative_risk_ucb_exceeded"
        else:
            reason = "shared"
        return GateDecision(
            share=effect_passed and risk_passed,
            reason=reason,
            gate_name=self.gate_name,
            effect_condition_passed=effect_passed,
            risk_condition_passed=risk_passed,
        )
