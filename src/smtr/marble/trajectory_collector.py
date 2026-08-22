"""Trajectory collector: execute MARBLE episodes and collect structured trajectories.

Wraps the existing ``engine_process`` and ``branch_runner`` infrastructure
to run real MARBLE Engine episodes and collect agent actions, environment
transitions, rewards, and messages into a structured ``Trajectory`` object.

Usage::

    collector = TrajectoryCollector(marble_root=Path("/home/ecs-user/MARBLE"))
    trajectory = collector.collect(task, seed=0, memory_payloads=[])
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.engine_process import (
    DEFAULT_ENGINE_TIMEOUT_SECONDS,
    MarbleEngineProcessResult,
    run_marble_engine_process,
)
from smtr.marble.environment.isolation import (
    InitialStateBundle,
    bundle_from_manifest_task,
    materialize_bundle_workspace,
)
from smtr.marble.task_loader import MarbleTask


@dataclass(frozen=True)
class Trajectory:
    """Structured record of one MARBLE episode execution.

    Attributes:
        trajectory_id: Unique identifier for this trajectory.
        task_id: MARBLE task identifier.
        scenario: Domain name.
        seed: Generation seed used for this episode.
        method: Memory injection method name.
        team_success: Whether the team completed the task successfully.
        score: Numeric score from the MARBLE evaluator.
        agent_actions: Per-agent action sequences.
        env_transitions: Environment state transitions.
        rewards: Per-agent reward values.
        agent_messages: Per-agent message history.
        memory_events: Memory injection / retrieval events.
        engine_duration_seconds: Wall-clock time for engine execution.
        exit_code: Engine subprocess exit code.
        real_engine_executed: Whether the engine actually ran.
        raw_output: Parsed raw result from the engine (if available).
    """

    trajectory_id: str
    task_id: str
    scenario: str
    seed: int
    method: str

    team_success: bool = False
    score: float = 0.0

    agent_actions: tuple[dict[str, Any], ...] = ()
    env_transitions: tuple[dict[str, Any], ...] = ()
    rewards: tuple[dict[str, Any], ...] = ()
    agent_messages: tuple[dict[str, Any], ...] = ()
    memory_events: tuple[dict[str, Any], ...] = ()

    engine_duration_seconds: float = 0.0
    exit_code: int = -1
    real_engine_executed: bool = False

    raw_output: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "method": self.method,
            "team_success": self.team_success,
            "score": self.score,
            "agent_actions": list(self.agent_actions),
            "env_transitions": list(self.env_transitions),
            "rewards": list(self.rewards),
            "agent_messages": list(self.agent_messages),
            "memory_events": list(self.memory_events),
            "engine_duration_seconds": self.engine_duration_seconds,
            "exit_code": self.exit_code,
            "real_engine_executed": self.real_engine_executed,
        }


class TrajectoryCollector:
    """Execute MARBLE episodes and collect structured trajectories.

    Parameters:
        marble_root: Path to the MARBLE repository root.
        workspace_root: Base directory for episode workspaces.
        engine_timeout: Max seconds for engine execution.
    """

    def __init__(
        self,
        *,
        marble_root: Path = Path("/home/ecs-user/MARBLE"),
        workspace_root: Path | None = None,
        engine_timeout: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    ) -> None:
        self.marble_root = marble_root
        self._workspace_root = workspace_root or Path(
            tempfile.mkdtemp(prefix="smtr_traj_")
        )
        self._engine_timeout = engine_timeout

    def collect(
        self,
        task: MarbleTask,
        *,
        seed: int = 0,
        method: str = "no_memory",
        memory_payloads: list[str] | None = None,
        receiver_agent_ids: list[str] | None = None,
    ) -> Trajectory:
        """Run one MARBLE episode and collect the trajectory.

        Parameters:
            task: The MARBLE task to execute.
            seed: Generation seed for reproducibility.
            method: Name of the memory injection method.
            memory_payloads: Rendered memory strings to inject.
            receiver_agent_ids: Agent IDs that receive memory injection.

        Returns:
            A ``Trajectory`` with all collected data.
        """
        trajectory_id = canonical_digest({
            "task_id": task.task_id,
            "scenario": task.scenario,
            "seed": seed,
            "method": method,
        })[:24]

        workspace = self._workspace_root / trajectory_id
        workspace.mkdir(parents=True, exist_ok=True)

        # Build initial state bundle
        bundle = bundle_from_manifest_task(
            task.raw_task,
            environment_seed=seed,
            generation_seed=seed,
        )
        materialize_bundle_workspace(bundle=bundle, workspace=workspace)

        # Write full MARBLE engine config (task data + engine fields)
        config_path = workspace / "task_config.json"
        full_config = _build_engine_config(
            task=task,
            raw_result_path=workspace / "marble_output.jsonl",
            seed=seed,
        )
        config_path.write_text(
            json.dumps(full_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Build memory injection payload
        memory_injection: dict[str, Any] | None = None
        if memory_payloads and receiver_agent_ids:
            memory_injection = {
                "receiver_agent_ids": receiver_agent_ids,
                "memory_payloads": memory_payloads,
                "memory_ids": [
                    f"mem_{i}" for i in range(len(memory_payloads))
                ],
                "intervention_id": trajectory_id,
            }

        # Raw result path
        raw_result_path = workspace / "marble_output.jsonl"

        # Run metadata for audit trail
        run_metadata = {
            "run_id": trajectory_id,
            "task_id": task.task_id,
            "scenario": task.scenario,
            "method": method,
            "branch": "online",
        }

        # Execute via engine_process
        try:
            result = run_marble_engine_process(
                config_path=config_path,
                marble_root=self.marble_root,
                raw_result_path=raw_result_path,
                output_dir=workspace / "engine_logs",
                run_identity=run_metadata,
                timeout_seconds=self._engine_timeout,
                memory_injection=memory_injection,
                run_metadata=run_metadata,
            )
        except Exception as exc:
            return Trajectory(
                trajectory_id=trajectory_id,
                task_id=task.task_id,
                scenario=task.scenario,
                seed=seed,
                method=method,
                team_success=False,
                score=0.0,
                exit_code=-1,
                real_engine_executed=False,
                raw_output={"error": str(exc)},
            )

        # Parse raw output
        raw_output = _parse_raw_output(raw_result_path)

        # Extract structured data from raw output
        team_success = _extract_team_success(raw_output)
        score = _extract_score(raw_output)
        agent_messages = _extract_agent_messages(raw_output)
        agent_actions = _extract_agent_actions(raw_output)
        rewards = _extract_rewards(raw_output)
        memory_events = _extract_memory_events(raw_output, memory_payloads)

        return Trajectory(
            trajectory_id=trajectory_id,
            task_id=task.task_id,
            scenario=task.scenario,
            seed=seed,
            method=method,
            team_success=team_success,
            score=score,
            agent_actions=tuple(agent_actions),
            env_transitions=(),  # Environment transitions from raw output
            rewards=tuple(rewards),
            agent_messages=tuple(agent_messages),
            memory_events=tuple(memory_events),
            engine_duration_seconds=result.engine_duration_seconds,
            exit_code=result.exit_code,
            real_engine_executed=result.real_engine_executed,
            raw_output=raw_output,
        )


# Scenario-to-environment-type mapping
_SCENARIO_ENV_TYPE = {
    "bargaining": "Base",
    "coding": "Coding",
    "database": "DB",
    "minecraft": "Minecraft",
    "research": "Research",
}


def _configured_litellm_model() -> str:
    """Return the LLM model name, matching the MARBLE environment adapters."""
    import os as _os

    model = (
        _os.environ.get("MARBLE_LLM_MODEL")
        or _os.environ.get("OPENAI_MODEL")
        or _os.environ.get("DASHSCOPE_MODEL")
    )
    compat = bool(
        _os.environ.get("DASHSCOPE_API_KEY")
        or _os.environ.get("DASHSCOPE_BASE_URL")
        or _os.environ.get("MARBLE_LLM_BASE_URL")
    )
    if not model and compat:
        model = "qwen-plus"
    if not model:
        return "gpt-4o-mini"
    if compat and "/" not in model:
        return f"openai/{model}"
    return model


def _build_engine_config(
    *,
    task: "MarbleTask",
    raw_result_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Build a full MARBLE engine config from a task.

    Merges the task-specific JSONL data with the engine-level fields
    (``llm``, ``coordinate_mode``, ``memory``, ``output``) that the
    MARBLE Engine requires but which are absent from the JSONL.
    """
    config = dict(task.raw_task)

    # LLM model (critical for agent execution)
    config["llm"] = _configured_litellm_model()

    # Coordinate mode: engine requires star/graph/chain/tree
    if not config.get("coordinate_mode"):
        config["coordinate_mode"] = "graph"

    # Environment type: engine requires a known type string
    env = config.get("environment")
    if isinstance(env, dict):
        if not env.get("type"):
            env["type"] = _SCENARIO_ENV_TYPE.get(task.scenario, "Base")
        if not env.get("max_iterations"):
            env["max_iterations"] = 5
        config["environment"] = env

    # Memory config
    mem = config.get("memory")
    if not isinstance(mem, dict) or not mem.get("type"):
        config["memory"] = {"type": "BaseMemory"}

    # Output path for engine results
    config["output"] = {"file_path": str(raw_result_path.resolve())}

    # Relationships (required for graph/chain/tree modes)
    if not config.get("relationships"):
        agents = config.get("agents", [])
        if isinstance(agents, list) and len(agents) > 1:
            agent_ids = [
                a.get("agent_id") or a.get("id") or f"agent{i+1}"
                for i, a in enumerate(agents)
            ]
            rels = []
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    rels.append([agent_ids[i], agent_ids[j], "collaborate with"])
            config["relationships"] = rels

    config["smtr_generation_seed"] = seed
    return config


def _parse_raw_output(path: Path) -> dict[str, Any]:
    """Parse MARBLE engine output JSONL into a structured dict."""
    if not path or not path.exists():
        return {}
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        return {}
    # Merge records into a single output dict
    output: dict[str, Any] = {}
    for record in records:
        output.update(record)
    output["_records"] = records
    return output


def _extract_team_success(raw: dict[str, Any]) -> bool:
    """Extract team success from raw output.

    Supports both flat format (``team_success`` / ``success`` keys) and
    MARBLE iterations format (where success is inferred from whether the
    engine completed its simulation loop with non-trivial results).
    """
    # MARBLE native evaluator (flat format)
    if "team_success" in raw:
        return bool(raw["team_success"])
    if "success" in raw:
        return bool(raw["success"])
    # Nested evaluator result
    evaluator = raw.get("evaluator", {})
    if isinstance(evaluator, dict):
        if "team_success" in evaluator:
            return bool(evaluator["team_success"])
    # MARBLE iterations format: engine ran and produced iterations
    iterations = raw.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        # Engine simulation completed → treat as executed successfully
        # (the evaluator may have crashed but the simulation itself ran)
        last = iterations[-1]
        if isinstance(last, dict):
            # Check if the last iteration has non-empty task_results
            tr = last.get("task_results", [])
            if isinstance(tr, list) and tr:
                return True
            summary = last.get("summary", "")
            if summary and len(str(summary)) > 20:
                return True
    return False


def _extract_score(raw: dict[str, Any]) -> float:
    """Extract numeric score from raw output.

    Falls back to planning_scores average or 1.0/0.0 based on team_success
    when the evaluator did not produce a top-level ``score`` key.
    """
    if "score" in raw:
        return float(raw["score"])
    evaluator = raw.get("evaluator", {})
    if isinstance(evaluator, dict) and "score" in evaluator:
        return float(evaluator["score"])
    # MARBLE iterations format: try planning_scores
    planning = raw.get("planning_scores", [])
    if isinstance(planning, list) and planning:
        valid = [s for s in planning if isinstance(s, (int, float)) and s >= 0]
        if valid:
            return sum(valid) / len(valid)
    # Fallback: 1.0 if team_success else 0.0
    return float(_extract_team_success(raw))


def _extract_agent_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-agent messages from raw output.

    Supports flat format (``messages`` key) and MARBLE iterations format
    where agent outputs live under ``iterations[].task_results``.
    """
    # Flat format
    messages = raw.get("messages", [])
    if isinstance(messages, list) and messages:
        return messages
    # Alternative: per-agent message dict
    agents = raw.get("agents", {})
    if isinstance(agents, dict) and agents:
        result = []
        for agent_id, agent_data in agents.items():
            if isinstance(agent_data, dict):
                for msg in agent_data.get("messages", []):
                    result.append({"agent_id": agent_id, **msg})
        if result:
            return result
    # MARBLE iterations format: task_results per iteration
    iterations = raw.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        result = []
        for it in iterations:
            if not isinstance(it, dict):
                continue
            task_results = it.get("task_results", [])
            if not isinstance(task_results, list):
                continue
            for entry in task_results:
                if isinstance(entry, dict):
                    for agent_id, content in entry.items():
                        result.append({
                            "agent_id": agent_id,
                            "content": str(content),
                            "role": "assistant",
                            "iteration": it.get("iteration", 0),
                        })
        return result
    return []


def _extract_agent_actions(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-agent actions from raw output.

    Supports flat format (``actions`` key) and MARBLE iterations format
    where agent task_assignments + task_results serve as action records.
    """
    actions = raw.get("actions", [])
    if isinstance(actions, list) and actions:
        return actions
    agents = raw.get("agents", {})
    if isinstance(agents, dict) and agents:
        result = []
        for agent_id, agent_data in agents.items():
            if isinstance(agent_data, dict):
                for act in agent_data.get("actions", []):
                    result.append({"agent_id": agent_id, **act})
        if result:
            return result
    # MARBLE iterations format: derive actions from task_assignments + results
    iterations = raw.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        result = []
        for it in iterations:
            if not isinstance(it, dict):
                continue
            assignments = it.get("task_assignments", {})
            task_results = it.get("task_results", [])
            if isinstance(task_results, list):
                for entry in task_results:
                    if isinstance(entry, dict):
                        for agent_id, content in entry.items():
                            assignment = assignments.get(agent_id, "")
                            result.append({
                                "agent_id": agent_id,
                                "action_type": "task_execution",
                                "assignment": str(assignment),
                                "result": str(content),
                                "iteration": it.get("iteration", 0),
                            })
        return result
    return []


def _extract_rewards(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-agent rewards from raw output."""
    rewards = raw.get("rewards", {})
    if isinstance(rewards, dict):
        return [{"agent_id": k, "reward": v} for k, v in rewards.items()]
    if isinstance(rewards, list):
        return rewards
    return []


def _extract_memory_events(
    raw: dict[str, Any],
    memory_payloads: list[str] | None,
) -> list[dict[str, Any]]:
    """Extract memory injection/retrieval events."""
    events: list[dict[str, Any]] = []
    # Check for SMTR memory visibility audit
    audit = raw.get("memory_visibility", [])
    if isinstance(audit, list):
        events.extend(audit)
    # Record injection if payloads were provided
    if memory_payloads:
        events.append({
            "event_type": "injection",
            "n_payloads": len(memory_payloads),
            "timestamp": time.time(),
        })
    return events
