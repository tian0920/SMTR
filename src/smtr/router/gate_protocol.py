"""Minimal gate protocol for formal SMTR routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ConditionStatus = Literal["passed", "failed", "not_applicable"]


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
    effect_condition_status: ConditionStatus
    risk_condition_status: ConditionStatus

    @property
    def effect_condition_passed(self) -> bool | None:
        if self.effect_condition_status == "not_applicable":
            return None
        return self.effect_condition_status == "passed"

    @property
    def risk_condition_passed(self) -> bool | None:
        if self.risk_condition_status == "not_applicable":
            return None
        return self.risk_condition_status == "passed"


class RoutingGate(Protocol):
    """Protocol implemented by formal SMTR gates."""

    @property
    def gate_name(self) -> str:
        ...

    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
        ...
