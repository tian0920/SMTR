"""Factory for MARBLE scenario outcome evaluators."""

from __future__ import annotations

from smtr.marble.outcome.protocol import MarbleOutcomeEvaluator
from smtr.marble.outcome.scenarios.database import DatabaseOutcomeEvaluator
from smtr.marble.outcome.scenarios.generic import GenericOutcomeEvaluator

_SUPPORTED_SCENARIOS = frozenset(
    {"bargaining", "coding", "database", "minecraft", "research"}
)


def evaluator_for_scenario(scenario: str) -> MarbleOutcomeEvaluator:
    if scenario == "database":
        return DatabaseOutcomeEvaluator()
    if scenario in _SUPPORTED_SCENARIOS:
        return GenericOutcomeEvaluator(scenario)
    raise ValueError(f"unsupported MARBLE outcome evaluator scenario: {scenario}")


def supported_scenarios() -> frozenset[str]:
    """Return the set of scenario names with registered evaluators."""
    return _SUPPORTED_SCENARIOS
