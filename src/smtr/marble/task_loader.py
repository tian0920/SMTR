"""MARBLE task loader: read official tasks from multiagentbench JSONL files.

Loads tasks from ``{marble_root}/multiagentbench/{scenario}/{scenario}_main.jsonl``.
Each task preserves its official ``task_id`` and full raw configuration.

Usage::

    loader = MarbleTaskLoader()
    tasks = loader.load_scenario("database")
    task = loader.get_task("database", "1")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")

DEFAULT_MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


@dataclass(frozen=True)
class MarbleTask:
    """One official MARBLE task from multiagentbench.

    Attributes:
        task_id: Official task identifier (e.g. ``"1"``, ``"51"``).
        scenario: Domain name (e.g. ``"database"``, ``"bargaining"``).
        raw_task: Full task configuration dict from the JSONL file.
        agents: Agent configuration list from the task.
        environment: Environment configuration dict.
        task_content: Task instruction / content dict.
    """

    task_id: str
    scenario: str
    raw_task: dict[str, Any] = field(repr=False)

    @property
    def agents(self) -> list[dict[str, Any]]:
        agents = self.raw_task.get("agents", [])
        return list(agents) if isinstance(agents, list) else []

    @property
    def environment(self) -> dict[str, Any]:
        env = self.raw_task.get("environment", {})
        return dict(env) if isinstance(env, dict) else {}

    @property
    def task_content(self) -> dict[str, Any]:
        task = self.raw_task.get("task", {})
        return dict(task) if isinstance(task, dict) else {}

    @property
    def coordinate_mode(self) -> str:
        return str(self.raw_task.get("coordinate_mode", ""))

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def get_agent_by_id(self, agent_id: str) -> dict[str, Any] | None:
        for agent in self.agents:
            if agent.get("agent_id") == agent_id or agent.get("id") == agent_id:
                return agent
        return None

    def get_agent_ids(self) -> list[str]:
        ids = []
        for agent in self.agents:
            aid = agent.get("agent_id") or agent.get("id") or ""
            if aid:
                ids.append(str(aid))
        return ids


class MarbleTaskLoader:
    """Load official MARBLE tasks from multiagentbench JSONL files.

    Parameters:
        marble_root: Path to the MARBLE repository root.
    """

    def __init__(
        self,
        marble_root: Path = DEFAULT_MARBLE_ROOT,
    ) -> None:
        self.marble_root = marble_root
        self._bench_root = marble_root / "multiagentbench"

    def _scenario_path(self, scenario: str) -> Path:
        """Return the JSONL file path for a scenario."""
        return self._bench_root / scenario / f"{scenario}_main.jsonl"

    def load_scenario(
        self,
        scenario: str,
        *,
        limit: int | None = None,
    ) -> list[MarbleTask]:
        """Load all tasks for one scenario.

        Parameters:
            scenario: Domain name (e.g. ``"database"``).
            limit: Maximum number of tasks to load.

        Returns:
            List of ``MarbleTask`` instances.

        Raises:
            FileNotFoundError: If the scenario JSONL file does not exist.
        """
        path = self._scenario_path(scenario)
        if not path.exists():
            raise FileNotFoundError(
                f"MARBLE scenario file not found: {path}"
            )

        tasks: list[MarbleTask] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                task_id = str(raw.get("task_id", len(tasks) + 1))
                # Normalise scenario field (minecraft may be missing it)
                raw_scenario = raw.get("scenario") or scenario
                raw["scenario"] = raw_scenario
                # Normalise environment.type — MARBLE engine requires
                # "Base" or "Web"; database JSONL ships with "" (empty).
                env = raw.get("environment")
                if isinstance(env, dict):
                    if not env.get("type"):
                        env["type"] = "Base"
                # Normalise coordinate_mode — MARBLE engine requires
                # star/graph/chain/tree; most JSONL files ship with "".
                if not raw.get("coordinate_mode"):
                    raw["coordinate_mode"] = "star"
                tasks.append(MarbleTask(
                    task_id=task_id,
                    scenario=raw_scenario,
                    raw_task=raw,
                ))
                if limit is not None and len(tasks) >= limit:
                    break

        return tasks

    def load_all(
        self,
        scenarios: list[str] | None = None,
        *,
        limit_per_scenario: int | None = None,
    ) -> list[MarbleTask]:
        """Load tasks from multiple scenarios.

        Parameters:
            scenarios: List of scenario names. ``None`` loads all 5.
            limit_per_scenario: Max tasks per scenario.

        Returns:
            Combined list of tasks across all requested scenarios.
        """
        if scenarios is None:
            scenarios = list(ALL_SCENARIOS)

        all_tasks: list[MarbleTask] = []
        for scenario in sorted(scenarios):
            try:
                tasks = self.load_scenario(
                    scenario, limit=limit_per_scenario
                )
                all_tasks.extend(tasks)
            except FileNotFoundError:
                # Skip missing scenarios with a warning
                import warnings
                warnings.warn(
                    f"Scenario not found, skipping: {scenario}",
                    stacklevel=2,
                )
        return all_tasks

    def get_task(self, scenario: str, task_id: str) -> MarbleTask:
        """Get a single task by scenario and task_id.

        Raises:
            KeyError: If the task is not found.
        """
        tasks = self.load_scenario(scenario)
        for t in tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(
            f"Task not found: scenario={scenario}, task_id={task_id}"
        )

    def list_scenarios(self) -> list[str]:
        """Return available scenarios (those with JSONL files)."""
        available = []
        for scenario in ALL_SCENARIOS:
            if self._scenario_path(scenario).exists():
                available.append(scenario)
        return available

    def load_task_file(
        self,
        path: Path,
    ) -> list[MarbleTask]:
        """Load specific tasks from a JSON task-file.

        The file must be a JSON array of objects with ``domain`` and
        ``task_id`` fields::

            [
                {"domain": "database", "task_id": "51"},
                {"domain": "database", "task_id": "73"},
                ...
            ]

        Returns:
            List of ``MarbleTask`` instances matching the entries.
        """
        with path.open("r", encoding="utf-8") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            raise ValueError(f"Task file must be a JSON array: {path}")

        # Group by domain for efficient loading
        by_domain: dict[str, list[str]] = {}
        for entry in entries:
            domain = str(entry.get("domain", entry.get("scenario", "")))
            task_id = str(entry["task_id"])
            by_domain.setdefault(domain, []).append(task_id)

        tasks: list[MarbleTask] = []
        for domain, task_ids in sorted(by_domain.items()):
            all_domain_tasks = self.load_scenario(domain)
            id_set = set(task_ids)
            for t in all_domain_tasks:
                if t.task_id in id_set:
                    tasks.append(t)

        return tasks

    def task_count(self, scenario: str) -> int:
        """Return the number of tasks in a scenario."""
        return len(self.load_scenario(scenario))
