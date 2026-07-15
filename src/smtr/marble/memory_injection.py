"""Memory injection adapter for MARBLE engine runs.

Builds share/withhold agent inputs and generates runtime shim payloads
that inject procedural memories into specific MARBLE agents' BaseMemory.
"""

from __future__ import annotations

import uuid
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


@dataclass(frozen=True)
class MemoryPayload:
    """A single memory payload to inject into a MARBLE agent."""

    memory_id: str
    payload: str
    role: str = "procedural"


@dataclass(frozen=True)
class InjectionResult:
    """Result of building a memory injection config."""

    agent_input: dict[str, Any]
    audit: MarbleAgentInputAudit
    memory_injection: dict[str, Any] | None
    intervention_id: str


class MarbleMemoryInjector:
    """Build share/withhold agent input and runtime shim injection payloads."""

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

    def build_injection(
        self,
        *,
        base_agent_input: dict[str, Any],
        memory_payloads: list[MemoryPayload],
        receiver_agent_ids: list[str],
        intervention_id: str | None = None,
    ) -> InjectionResult:
        """Build both the agent input and the runtime shim injection payload."""
        payloads_tuple = tuple(m.payload for m in memory_payloads)
        ids_tuple = tuple(m.memory_id for m in memory_payloads)
        agent_input, audit = self.build_agent_input(
            base_agent_input=base_agent_input,
            memory_payloads=payloads_tuple,
            memory_ids=ids_tuple,
        )
        iid = intervention_id or uuid.uuid4().hex[:12]
        injection: dict[str, Any] | None = None
        if memory_payloads and receiver_agent_ids:
            injection = {
                "receiver_agent_ids": receiver_agent_ids,
                "memory_payloads": list(payloads_tuple),
                "memory_ids": list(ids_tuple),
                "intervention_id": iid,
            }
        return InjectionResult(
            agent_input=agent_input,
            audit=audit,
            memory_injection=injection,
            intervention_id=iid,
        )
