"""Pair execution context factory for MARBLE paired interventions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from smtr.marble.environment.isolation import InitialStateBundle
from smtr.marble.task_provider import load_database_task_by_id


class PairExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task: dict[str, Any]
    initial_state_bundle: InitialStateBundle
    agent_config: dict[str, Any]


def build_pair_execution_context(
    *,
    marble_root: Path,
    task_entry: dict[str, Any],
    receiver_agent_id: str,
    workspace: Path,
) -> PairExecutionContext:
    """Build the task, initial database state and agent configuration
    needed by the shared-control branch runner protocol (清单
    Shared-Control 第3章).

    The full MARBLE task is loaded from the source JSONL file so that
    ``agents``, ``relationships``, ``environment.init_sql``,
    ``task.content`` / ``labels`` / ``root_causes`` are available for the
    engine and outcome evaluator.

    Requirements:
    - target_receiver_agent_id must equal receiver_agent_id
    - task digest must come from the frozen MARBLE task
    - tool config must come from the actual MARBLE scenario config
    - initial state bundle must be reconstructable for both branches
    """
    from smtr.counterfactual.decision_points import canonical_digest
    from smtr.marble.environment.database_rebuild import SequentialDatabaseRebuilder

    task_id = str(task_entry["task_id"])
    scenario = task_entry.get("scenario", "database")

    # Load the full MARBLE task from source JSONL (agents, env, task content, etc.)
    full_task = load_database_task_by_id(marble_root, task_id)

    # Build agent config targeting the specific receiver
    agent_config: dict[str, Any] = {
        "target_receiver_agent_id": receiver_agent_id,
        "scenario": scenario,
        "task_id": task_id,
        "agents": full_task.get("agents", []),
    }

    # Compute digests from frozen task
    task_digest = canonical_digest(full_task)
    tool_config_digest = canonical_digest({"scenario": scenario, "task_id": task_id})
    agent_config_digest = canonical_digest(agent_config)

    # Build initial state bundle using full task source data
    initial_state_bundle = InitialStateBundle(
        task_id=task_id,
        scenario=scenario,
        task_source_snapshot=task_entry,
    )

    return PairExecutionContext(
        task=full_task,
        initial_state_bundle=initial_state_bundle,
        agent_config=agent_config,
    )
