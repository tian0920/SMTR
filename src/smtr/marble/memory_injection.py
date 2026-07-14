"""Auditable memory intervention layer for MARBLE agent inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest


@dataclass(frozen=True)
class MarbleAgentInputAudit:
    system_section_digest: str
    task_section_digest: str
    tool_section_digest: str
    memory_section_digest: str | None
    full_input_digest: str
    memory_ids: tuple[str, ...]
    contains_memory_section: bool


class MarbleMemoryInjector:
    """Build share/withhold agent input while auditing only digests."""

    def build_agent_input(
        self,
        *,
        base_agent_input: dict[str, Any],
        memory_payloads: tuple[str, ...],
        memory_ids: tuple[str, ...],
    ) -> tuple[dict[str, Any], MarbleAgentInputAudit]:
        system = base_agent_input.get("system", {})
        task = base_agent_input.get("task", {})
        tools = base_agent_input.get("tools", {})
        agent_input = {
            "system": system,
            "task": task,
            "tools": tools,
        }
        memory_section = None
        if memory_payloads:
            memory_section = {
                "private_memory_payloads": list(memory_payloads),
                "memory_ids": list(memory_ids),
            }
            agent_input["memory"] = memory_section
        audit = MarbleAgentInputAudit(
            system_section_digest=canonical_digest(system),
            task_section_digest=canonical_digest(task),
            tool_section_digest=canonical_digest(tools),
            memory_section_digest=(
                canonical_digest(memory_section) if memory_section is not None else None
            ),
            full_input_digest=canonical_digest(agent_input),
            memory_ids=tuple(memory_ids) if memory_section is not None else (),
            contains_memory_section=memory_section is not None,
        )
        return agent_input, audit
