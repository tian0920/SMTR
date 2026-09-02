"""Receiver topology for RIMA (Phase 11).

Receivers MUST come from the real agent topology of the current task::

    active_agents = task.get_agent_ids()

Hardcoded ``RECEIVER_IDS = ["agent1", "agent2", "agent3"]`` is forbidden.

For the receiver=3 main setting, exactly 3 agents are selected
DETERMINISTICALLY from the active agents. The exclusion policy is defined
a priori (fixed by task_id hash ordering) and must never depend on
outcomes.

If a scenario cannot support 3 receivers, the exclusion is declared up
front via :class:`ReceiverExclusionPolicy` — never post-hoc by result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ReceiverAssignment",
    "ReceiverExclusionPolicy",
    "select_receivers",
]

DEFAULT_RECEIVER_COUNT = 3


@dataclass(frozen=True)
class ReceiverExclusionPolicy:
    """A priori exclusion policy (declared before any run).

    Attributes:
        excluded_roles: agent roles that cannot be receivers (e.g. a
            coordinator that never consumes memories).
        reason: documented justification (audit field).
    """

    excluded_roles: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ReceiverAssignment:
    """The selected receiver topology for one task."""

    receiver_id: str
    receiver_role: str
    agent_topology: tuple[str, ...] = field(default_factory=tuple)


def _stable_order(agent_ids: list[str], task_id: str) -> list[str]:
    """Deterministic ordering of agents for a task (hash of task+agent)."""
    return sorted(
        agent_ids,
        key=lambda a: hashlib.sha256(f"{task_id}:{a}".encode()).hexdigest(),
    )


def select_receivers(
    *,
    task: dict[str, Any],
    task_id: str,
    receiver_count: int = DEFAULT_RECEIVER_COUNT,
    exclusion_policy: ReceiverExclusionPolicy | None = None,
) -> list[ReceiverAssignment]:
    """Deterministically select receivers from the task's real agents.

    Args:
        task: task dict; must expose agent ids under ``agent_ids``
            (``task.get_agent_ids()`` equivalent).
        task_id: used to make selection deterministic per task.
        receiver_count: number of receivers (3 for the main setting).
        exclusion_policy: optional a priori role exclusions.

    Returns:
        Ordered receiver assignments. Fewer than ``receiver_count`` are
        returned only when the topology genuinely lacks eligible agents;
        selection never depends on outcomes.

    Raises:
        ValueError: when the task exposes no agent ids at all.
    """
    agent_ids = list(task.get("agent_ids") or [])
    if not agent_ids:
        raise ValueError(
            "Task exposes no agent_ids; receivers must come from the real "
            "agent topology (Phase 11)."
        )
    policy = exclusion_policy or ReceiverExclusionPolicy()
    excluded_roles = set(policy.excluded_roles)

    roles: dict[str, str] = {}
    for agent in task.get("agents", []) or []:
        aid = agent.get("agent_id") or agent.get("id")
        role = agent.get("role", "unknown")
        if aid:
            roles[str(aid)] = str(role)

    eligible = [
        a for a in _stable_order(agent_ids, task_id) if roles.get(a, "unknown") not in excluded_roles
    ]
    topology = tuple(sorted(agent_ids))
    return [
        ReceiverAssignment(
            receiver_id=aid,
            receiver_role=roles.get(aid, "unknown"),
            agent_topology=topology,
        )
        for aid in eligible[:receiver_count]
    ]
