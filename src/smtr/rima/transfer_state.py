"""Receiver-specific Transfer State (RIMA-v2 §2-3).

The transfer state K_r^{transfer} tracks which memories a receiver has
explored and the accumulated transfer-prediction information.

Semantics:
    memory in K_r^{transfer} means the receiver has explored/modelled
    this memory — NOT that the memory is permanently positive.

    Negative memories MUST remain in the state so they can become
    positive candidates for future tasks.

Payload separation (§3):
    ReceiverTransferState never stores SharedMemory objects or
    procedure_payload. Memory content lives only in SharedMemoryPool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "TransferPredictionRecord",
    "TransferStateEntry",
    "ReceiverTransferState",
    "ReceiverTransferStateContainer",
]


@dataclass
class TransferPredictionRecord:
    """One prediction record for a (memory, receiver, task) evaluation.

    Attributes:
        task_id: task at prediction time.
        task_position: position of the task in the continual stream.
        mu_tau: predicted mean transfer effect.
        sigma_tau: predicted uncertainty.
        lcb: lower confidence bound (mu - beta * sigma).
        status: one of "positive", "negative", "uncertain", "invalid".
        candidate_source: "known" (from transfer state) or "global" (new retrieval).
        predicted_at: timestamp of the prediction.
    """

    task_id: str
    task_position: int

    mu_tau: float | None
    sigma_tau: float | None
    lcb: float | None

    status: str
    candidate_source: str

    predicted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class TransferStateEntry:
    """One entry in the receiver's transfer state.

    Attributes:
        memory_id: the memory being tracked.
        source_agent_id: agent that produced this memory.
        first_seen_task_id: task when this memory was first explored.
        first_seen_task_position: position in the continual stream.
        times_considered: how many times this memory was evaluated.
        times_selected: how many times this memory was selected for injection.
        last_mu_tau: most recent predicted mu_tau.
        last_sigma_tau: most recent predicted sigma_tau.
        last_lcb: most recent LCB.
        last_task_id: most recent task at which this was evaluated.
        prediction_history: full history of predictions.
    """

    memory_id: str
    source_agent_id: str

    first_seen_task_id: str
    first_seen_task_position: int

    times_considered: int = 0
    times_selected: int = 0

    last_mu_tau: float | None = None
    last_sigma_tau: float | None = None
    last_lcb: float | None = None
    last_task_id: str | None = None

    prediction_history: list[TransferPredictionRecord] = field(
        default_factory=list
    )


class ReceiverTransferState:
    """Receiver-specific transfer state tracking explored memories.

    The state is keyed by memory_id and tracks prediction history,
    selection counts, and last-known transfer estimates.
    """

    def __init__(self, receiver_id: str) -> None:
        self.receiver_id = receiver_id
        self._entries: dict[str, TransferStateEntry] = {}

    def register_memory(
        self,
        memory_id: str,
        source_agent_id: str,
        task_id: str,
        task_position: int,
    ) -> TransferStateEntry:
        """Register a memory in the transfer state (idempotent).

        If the memory is already registered, returns the existing entry
        without modification.

        Raises:
            ValueError: if source_agent_id == receiver_id (self-transfer).
        """
        if source_agent_id == self.receiver_id:
            raise ValueError(
                f"Self-transfer forbidden: source_agent_id={source_agent_id} "
                f"== receiver_id={self.receiver_id}"
            )
        if memory_id in self._entries:
            return self._entries[memory_id]
        entry = TransferStateEntry(
            memory_id=memory_id,
            source_agent_id=source_agent_id,
            first_seen_task_id=task_id,
            first_seen_task_position=task_position,
        )
        self._entries[memory_id] = entry
        return entry

    def contains(self, memory_id: str) -> bool:
        return memory_id in self._entries

    def known_memory_ids(self) -> set[str]:
        return set(self._entries.keys())

    def get_entry(self, memory_id: str) -> TransferStateEntry | None:
        return self._entries.get(memory_id)

    def record_prediction(
        self,
        memory_id: str,
        task_id: str,
        task_position: int,
        mu_tau: float | None,
        sigma_tau: float | None,
        lcb: float | None,
        status: str,
        candidate_source: str,
    ) -> TransferPredictionRecord:
        """Record a prediction for a known memory.

        Updates last_* fields and appends to prediction_history.
        Increments times_considered.

        Raises:
            KeyError: if memory_id not registered.
        """
        entry = self._entries.get(memory_id)
        if entry is None:
            raise KeyError(
                f"memory_id={memory_id!r} not in transfer state for "
                f"receiver={self.receiver_id!r}. Register first."
            )
        record = TransferPredictionRecord(
            task_id=task_id,
            task_position=task_position,
            mu_tau=mu_tau,
            sigma_tau=sigma_tau,
            lcb=lcb,
            status=status,
            candidate_source=candidate_source,
        )
        entry.prediction_history.append(record)
        entry.times_considered += 1
        entry.last_mu_tau = mu_tau
        entry.last_sigma_tau = sigma_tau
        entry.last_lcb = lcb
        entry.last_task_id = task_id
        return record

    def mark_selected(self, memory_id: str) -> None:
        """Mark that a memory was selected for injection."""
        entry = self._entries.get(memory_id)
        if entry is None:
            raise KeyError(
                f"memory_id={memory_id!r} not in transfer state for "
                f"receiver={self.receiver_id!r}."
            )
        entry.times_selected += 1

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        entries = {}
        for mid, entry in self._entries.items():
            entries[mid] = {
                "memory_id": entry.memory_id,
                "source_agent_id": entry.source_agent_id,
                "first_seen_task_id": entry.first_seen_task_id,
                "first_seen_task_position": entry.first_seen_task_position,
                "times_considered": entry.times_considered,
                "times_selected": entry.times_selected,
                "last_mu_tau": entry.last_mu_tau,
                "last_sigma_tau": entry.last_sigma_tau,
                "last_lcb": entry.last_lcb,
                "last_task_id": entry.last_task_id,
                "n_prediction_records": len(entry.prediction_history),
            }
        return {
            "receiver_id": self.receiver_id,
            "n_entries": len(self._entries),
            "entries": entries,
        }


class ReceiverTransferStateContainer:
    """Container managing transfer states for all receivers."""

    def __init__(self) -> None:
        self._states: dict[str, ReceiverTransferState] = {}

    def ensure(self, receiver_id: str) -> ReceiverTransferState:
        """Get or create a transfer state for a receiver."""
        if receiver_id not in self._states:
            self._states[receiver_id] = ReceiverTransferState(receiver_id)
        return self._states[receiver_id]

    def get(self, receiver_id: str) -> ReceiverTransferState | None:
        return self._states.get(receiver_id)

    def receiver_ids(self) -> set[str]:
        return set(self._states.keys())
