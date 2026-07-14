"""Outcome protocol for MARBLE scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from smtr.counterfactual.decision_points import canonical_digest


@dataclass(frozen=True)
class MarbleOutcome:
    success: bool
    score: float | None
    failure_reason: str | None
    environment_valid: bool
    evaluator_name: str
    raw_result_digest: str
    native_evaluator_executed: bool = False
    native_evaluator_name: str | None = None
    native_evaluator_result_digest: str | None = None


class MarbleOutcomeEvaluator(Protocol):
    scenario: str
    evaluator_name: str

    def evaluate(self, *, task: object, run_result: object) -> MarbleOutcome:
        ...


def outcome_from_failure(*, evaluator_name: str, reason: str, raw_result: object) -> MarbleOutcome:
    return MarbleOutcome(
        success=False,
        score=None,
        failure_reason=reason,
        environment_valid=False,
        evaluator_name=evaluator_name,
        raw_result_digest=canonical_digest(raw_result),
        native_evaluator_executed=False,
        native_evaluator_name=None,
        native_evaluator_result_digest=None,
    )
