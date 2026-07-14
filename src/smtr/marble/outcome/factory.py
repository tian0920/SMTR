"""Factory for MARBLE scenario outcome evaluators."""

from __future__ import annotations

from smtr.marble.outcome.protocol import MarbleOutcomeEvaluator
from smtr.marble.outcome.scenarios.database import DatabaseOutcomeEvaluator


def evaluator_for_scenario(scenario: str) -> MarbleOutcomeEvaluator:
    if scenario == "database":
        return DatabaseOutcomeEvaluator()
    raise ValueError(f"unsupported MARBLE outcome evaluator scenario: {scenario}")
