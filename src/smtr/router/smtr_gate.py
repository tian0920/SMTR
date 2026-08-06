"""Formal SMTR mean-effect + mean-risk gate."""

from __future__ import annotations

from dataclasses import dataclass

from smtr.router.gate_protocol import GateDecision, TransferPointEstimate


@dataclass(frozen=True)
class SMTRGateConfig:
    """Configuration for the formal SMTR gate.

    ``None`` means the budget is unresolved; callers must resolve it from
    the critic checkpoint's validation-selected epsilon_star before any
    decision is made (清单 P1-2).
    """

    negative_risk_budget: float | None = None

    def __post_init__(self) -> None:
        if self.negative_risk_budget is not None and not (
            0.0 <= self.negative_risk_budget <= 1.0
        ):
            raise ValueError("negative_risk_budget must be in [0, 1]")


@dataclass(frozen=True)
class SMTRGate:
    """Share iff tau_mean > 0 and negative_risk_mean is within budget."""

    config: SMTRGateConfig
    gate_name: str = "smtr_mean_effect_mean_risk"

    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
        if self.config.negative_risk_budget is None:
            raise ValueError(
                "Checkpoint does not contain validation-selected epsilon_star."
            )
        effect_passed = estimate.tau_mean > 0.0
        risk_passed = estimate.negative_risk_mean <= self.config.negative_risk_budget

        if not effect_passed:
            reason = "tau_mean_nonpositive"
        elif not risk_passed:
            reason = "negative_risk_mean_exceeded"
        else:
            reason = "shared"

        return GateDecision(
            share=effect_passed and risk_passed,
            reason=reason,
            gate_name=self.gate_name,
            effect_condition_status="passed" if effect_passed else "failed",
            risk_condition_status="passed" if risk_passed else "failed",
        )
