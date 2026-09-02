"""Receiver Knowledge State K_r^t for RIMA (Phase 6).

The paper distinguishes the shared pool ``M^t`` from the per-receiver
knowledge states ``K_r^t``:

* ``M^t``: all historical shared memories (Phase 5);
* ``K_r^t``: the memories ADMITTED to receiver ``r`` up to task ``t``.

Mixing ``C_r^t`` (current retrieval candidates) and ``K_r^t`` (persistent
admitted knowledge) is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from smtr.memory.shared_memory_pool import SharedMemory

__all__ = [
    "AdmissionRecord",
    "ReceiverKnowledgeState",
    "ReceiverKnowledgeContainer",
]


@dataclass(frozen=True)
class AdmissionRecord:
    """One historical admission event for a receiver."""

    memory_id: str
    task_id: str
    task_position: int
    tau_hat: float | None
    admitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReceiverKnowledgeState:
    """Persistent admitted knowledge of one receiver (K_r)."""

    def __init__(self, receiver_id: str) -> None:
        self.receiver_id = receiver_id
        self.admitted_memory_ids: list[str] = []
        self.admission_history: list[AdmissionRecord] = []
        self._memories: dict[str, SharedMemory] = {}

    def admit(self, memory: SharedMemory, tau_hat: float | None, *, task_id: str, task_position: int) -> None:
        """Admit a memory into K_r (called only for tau_hat > 0 decisions)."""
        if memory.source_agent_id == self.receiver_id:
            raise ValueError(
                f"Self-transfer forbidden: memory {memory.memory_id!r} "
                f"source==receiver=={self.receiver_id!r} (Phase 12)."
            )
        if memory.memory_id in self._memories:
            return  # idempotent re-admission
        self._memories[memory.memory_id] = memory
        self.admitted_memory_ids.append(memory.memory_id)
        self.admission_history.append(
            AdmissionRecord(
                memory_id=memory.memory_id,
                task_id=task_id,
                task_position=task_position,
                tau_hat=tau_hat,
            )
        )

    def contains(self, memory_id: str) -> bool:
        return memory_id in self._memories

    def retrieve(self, task: dict[str, Any], top_k: int) -> list[SharedMemory]:
        """Retrieve from K_r for execution injection (deterministic order)."""
        ordered = [self._memories[mid] for mid in self.admitted_memory_ids]
        return ordered[: max(0, top_k)]

    def __len__(self) -> int:
        return len(self._memories)


class ReceiverKnowledgeContainer:
    """Container K = {receiver_id: K_r}."""

    def __init__(self, receiver_ids: list[str] | None = None) -> None:
        self._states: dict[str, ReceiverKnowledgeState] = {}
        for rid in receiver_ids or []:
            self.ensure(rid)

    def ensure(self, receiver_id: str) -> ReceiverKnowledgeState:
        if receiver_id not in self._states:
            self._states[receiver_id] = ReceiverKnowledgeState(receiver_id)
        return self._states[receiver_id]

    def get(self, receiver_id: str) -> ReceiverKnowledgeState:
        if receiver_id not in self._states:
            raise KeyError(f"Unknown receiver: {receiver_id!r}")
        return self._states[receiver_id]

    def receiver_ids(self) -> list[str]:
        return list(self._states)

    def payloads(self, *, context_budget: int) -> dict[str, list[str]]:
        """Receiver-specific injection payloads (Phase 10).

        Returns one payload list PER receiver — never a union broadcast.
        """
        return {
            rid: [m.procedure_payload for m in state.retrieve({}, context_budget)]
            for rid, state in self._states.items()
        }
