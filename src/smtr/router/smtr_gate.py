"""Formal SMTR tau-only selective-exposure gate.

SMTR-v1 routes purely on tau > 0 (net transfer effect).  eta (= q01)
is reported as a diagnostic quantity but never used as a routing gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from smtr.router.gate_protocol import GateDecision, TransferPointEstimate


@dataclass(frozen=True)
class SMTRGateConfig:
    """Configuration for the formal SMTR gate.

    SMTR-v1 has no risk-budget threshold; the gate decides solely on
    tau > 0.  The config is kept as a dataclass for forward-compatibility
    but carries no tunable fields.
    """


@dataclass(frozen=True)
class SMTRGate:
    """Share iff tau_mean > 0 (pure net-transfer-effect gate)."""

    config: SMTRGateConfig = SMTRGateConfig()
    gate_name: str = "smtr_tau_selective_exposure"

    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
        effect_passed = estimate.tau_mean > 0.0

        reason = "shared" if effect_passed else "tau_mean_nonpositive"

        return GateDecision(
            share=effect_passed,
            reason=reason,
            gate_name=self.gate_name,
            effect_condition_status="passed" if effect_passed else "failed",
            risk_condition_status="n/a",
        )
