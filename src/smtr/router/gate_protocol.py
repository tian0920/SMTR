"""Minimal gate protocol for formal SMTR routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TransferPointEstimate:
    """Point estimates needed by the formal SMTR gate."""

    tau_mean: float
    negative_risk_mean: float


@dataclass(frozen=True)
class GateDecision:
    """Formal gate decision for one candidate memory."""

    share: bool
    reason: str
    gate_name: str
    effect_condition_passed: bool
    risk_condition_passed: bool | None


class RoutingGate(Protocol):
    """Protocol implemented by formal SMTR gates."""

    gate_name: str

    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
        ...
