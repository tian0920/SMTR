"""Scenario adapter registry for MARBLE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioAdapter:
    """Static metadata for a MARBLE scenario type."""

    scenario: str
    environment_type: str
    default_max_iterations: int
    native_evaluator_method: str
    agent_ids: list[str]


_DATABASE_ADAPTER = ScenarioAdapter(
    scenario="database",
    environment_type="DB",
    default_max_iterations=1,
    native_evaluator_method="evaluate_task_db",
    agent_ids=["agent1", "agent2", "agent3", "agent4", "agent5"],
)

SCENARIO_ADAPTERS: dict[str, ScenarioAdapter] = {
    "database": _DATABASE_ADAPTER,
}


def adapter_for_scenario(scenario: str) -> ScenarioAdapter:
    """Return the adapter for a given scenario, raising if unknown."""
    adapter = SCENARIO_ADAPTERS.get(scenario)
    if adapter is None:
        known = sorted(SCENARIO_ADAPTERS.keys())
        raise ValueError(f"unknown scenario {scenario!r}, known: {known}")
    return adapter


def available_scenarios() -> list[str]:
    return sorted(SCENARIO_ADAPTERS.keys())


def scenario_metadata(scenario: str) -> dict[str, Any]:
    """Return a JSON-serialisable metadata dict for a scenario."""
    adapter = adapter_for_scenario(scenario)
    return {
        "scenario": adapter.scenario,
        "environment_type": adapter.environment_type,
        "default_max_iterations": adapter.default_max_iterations,
        "native_evaluator_method": adapter.native_evaluator_method,
        "default_agent_ids": adapter.agent_ids,
    }
