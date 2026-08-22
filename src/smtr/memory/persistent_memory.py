"""Persistent memory bank for long-term knowledge accumulation (Task 1).

Lifecycle:

    experience -> candidate memory -> TCI validation
        -> validated (persistent knowledge) | rejected

The bank is an independent module: it never touches
``SharedMemoryPool`` / ``SQLiteSharedMemoryRepository`` and keeps its own
JSONL persistence so lifelong experiments can survive across episodes.

Receiver lifecycle state:
    ``receiver_status`` on each entry is the authoritative lifecycle
    state per receiver. The legacy ``status`` field is kept for
    backward compatibility but is NOT the source of truth for
    receiver-conditioned decisions.
"""

import json
from pathlib import Path
from typing import Literal

from smtr.memory.memory_schema import (
    MemoryLifecycleStatus,
    PersistentMemoryEntry,
    ValidationRecord,
    utc_now,
)


class PersistentMemoryBank:
    """In-memory bank of lifecycle-tracked memories.

    State transitions:
      - ``add_candidate``: creates a ``candidate`` entry (id must be new)
      - ``validate_memory``: candidate|rejected -> ``validated``
      - ``reject_memory``: candidate|validated -> ``rejected``
    Every validation/rejection increments ``validation_count`` and records
    the observed TCI effect.
    """

    def __init__(self) -> None:
        self._entries: dict[str, PersistentMemoryEntry] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_candidate(
        self,
        *,
        memory_id: str,
        content: str,
        source_episode: int,
        receiver: str,
        created_step: int,
    ) -> PersistentMemoryEntry:
        """Register a freshly extracted candidate memory."""
        if memory_id in self._entries:
            raise ValueError(f"duplicate memory_id: {memory_id}")
        entry = PersistentMemoryEntry(
            memory_id=memory_id,
            content=content,
            source_episode=source_episode,
            receiver=receiver,
            created_step=created_step,
        )
        self._entries[memory_id] = entry
        return entry

    def validate_memory(self, memory_id: str, tci_effect: float,
                        *, episode_id: int = -1,
                        expose_reward: float | None = None,
                        withhold_reward: float | None = None,
                        decision: str = "validated") -> PersistentMemoryEntry:
        """Mark a memory validated after positive TCI evidence (delta > 0)."""
        return self._transition(memory_id, "validated", tci_effect,
                                episode_id=episode_id,
                                expose_reward=expose_reward if expose_reward is not None else tci_effect,
                                withhold_reward=withhold_reward if withhold_reward is not None else 0.0,
                                audit_decision=decision)

    def reject_memory(self, memory_id: str, tci_effect: float,
                      *, episode_id: int = -1,
                      expose_reward: float | None = None,
                      withhold_reward: float | None = None,
                      decision: str = "rejected") -> PersistentMemoryEntry:
        """Mark a memory rejected after non-positive TCI evidence."""
        return self._transition(memory_id, "rejected", tci_effect,
                                episode_id=episode_id,
                                expose_reward=expose_reward if expose_reward is not None else tci_effect,
                                withhold_reward=withhold_reward if withhold_reward is not None else 0.0,
                                audit_decision=decision)

    def _transition(
        self, memory_id: str, status: str, tci_effect: float,
        *, episode_id: int = -1,
        expose_reward: float = 0.0,
        withhold_reward: float = 0.0,
        audit_decision: str | None = None,
    ) -> PersistentMemoryEntry:
        entry = self.get(memory_id)
        # Build audit record (P0-3)
        rec = ValidationRecord(
            episode_id=episode_id,
            expose_reward=expose_reward,
            withhold_reward=withhold_reward,
            delta=tci_effect,
            decision=audit_decision or status,
        )
        updated = entry.model_copy(
            update={
                "status": status,
                "tci_effect": tci_effect,
                "validation_count": entry.validation_count + 1,
                "validation_history": entry.validation_history + (rec,),
                "updated_at": utc_now(),
            }
        )
        self._entries[memory_id] = updated
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, memory_id: str) -> PersistentMemoryEntry:
        if memory_id not in self._entries:
            raise KeyError(f"unknown memory_id: {memory_id}")
        return self._entries[memory_id]

    def retrieve_validated(self, receiver: str | None = None) -> list[PersistentMemoryEntry]:
        """Validated memories, optionally filtered by receiver, oldest first."""
        entries = [e for e in self._entries.values() if e.status == "validated"]
        if receiver is not None:
            entries = [e for e in entries if e.receiver == receiver]
        return sorted(entries, key=lambda e: (e.created_step, e.memory_id))

    def all_entries(self) -> list[PersistentMemoryEntry]:
        return sorted(self._entries.values(), key=lambda e: (e.created_step, e.memory_id))

    # ------------------------------------------------------------------
    # Receiver lifecycle state (authoritative per-receiver status)
    # ------------------------------------------------------------------
    def set_receiver_status(
        self,
        memory_id: str,
        receiver_id: str,
        status: MemoryLifecycleStatus,
    ) -> PersistentMemoryEntry:
        """Set the authoritative lifecycle state for one (memory, receiver).

        This is the single source of truth for receiver-conditioned
        retrieval. The legacy ``status`` field is NOT modified.
        """
        entry = self.get(memory_id)
        new_status = dict(entry.receiver_status)
        new_status[receiver_id] = status
        updated = entry.model_copy(
            update={"receiver_status": new_status, "updated_at": utc_now()}
        )
        self._entries[memory_id] = updated
        return updated

    def get_receiver_status(
        self, memory_id: str, receiver_id: str
    ) -> MemoryLifecycleStatus | None:
        """Return lifecycle state for one (memory, receiver), or None if unset."""
        entry = self.get(memory_id)
        return entry.receiver_status.get(receiver_id)

    def get_receiver_validated_memories(
        self, receiver_id: str
    ) -> list[PersistentMemoryEntry]:
        """Memories whose ``receiver_status[receiver_id] == 'validated'``.

        This is the authoritative retrieval path for receiver-conditioned
        knowledge. Sorted oldest-first by created_step.
        """
        entries = [
            e for e in self._entries.values()
            if e.receiver_status.get(receiver_id) == "validated"
        ]
        return sorted(entries, key=lambda e: (e.created_step, e.memory_id))

    def receiver_validation_summary(self) -> dict[str, dict[str, object]]:
        """Per-receiver summary of lifecycle decisions.

        Returns a dict keyed by receiver_id with:
          - decision: latest receiver_status for each memory (validated/rejected/candidate)
          - latest_delta: most recent delta from receiver_validation_history
          - validation_count: number of receiver-conditioned validations
        """
        summary: dict[str, dict[str, object]] = {}
        for entry in self._entries.values():
            for rid, status in entry.receiver_status.items():
                if rid not in summary:
                    summary[rid] = {
                        "validated": 0,
                        "rejected": 0,
                        "candidate": 0,
                        "latest_deltas": [],
                    }
                s = summary[rid]
                s[status] = s.get(status, 0) + 1  # type: ignore[arg-type]
                # Collect latest delta from receiver_validation_history
                for rec in entry.receiver_validation_history:
                    if rec.receiver_id == rid:
                        s["latest_deltas"].append(rec.delta)  # type: ignore[union-attr]
        # Finalise: compute latest_delta and validation_count
        for rid, s in summary.items():
            deltas: list[float] = s.pop("latest_deltas")  # type: ignore[assignment]
            s["latest_delta"] = deltas[-1] if deltas else None
            s["validation_count"] = len(deltas)
        return summary

    def get_statistics(self) -> dict[str, object]:
        statuses = [e.status for e in self._entries.values()]
        return {
            "total": len(self._entries),
            "candidate": statuses.count("candidate"),
            "validated": statuses.count("validated"),
            "rejected": statuses.count("rejected"),
            "mean_tci_effect_validated": _mean(
                [e.tci_effect for e in self._entries.values()
                 if e.status == "validated" and e.tci_effect is not None]
            ),
            "mean_validation_count": _mean(
                [float(e.validation_count) for e in self._entries.values()]
            ),
        }

    # ------------------------------------------------------------------
    # Audit export (P0-3)
    # ------------------------------------------------------------------
    def export_memory_audit(self) -> list[dict]:
        """Export full audit trail for every memory (for paper case study).

        Each entry contains:
          - source: memory_id, content, source_episode, receiver
          - validation_evidence: list of {episode_id, expose_reward,
            withhold_reward, delta, decision}
          - retention_decision: current status
          - later_utility: latest tci_effect
        """
        audit: list[dict] = []
        for entry in self.all_entries():
            audit.append({
                "memory_id": entry.memory_id,
                "content": entry.content,
                "source_episode": entry.source_episode,
                "receiver": entry.receiver,
                "created_step": entry.created_step,
                "status": entry.status,
                "tci_effect": entry.tci_effect,
                "validation_count": entry.validation_count,
                "validation_history": [
                    {
                        "episode_id": r.episode_id,
                        "expose_reward": r.expose_reward,
                        "withhold_reward": r.withhold_reward,
                        "delta": r.delta,
                        "decision": r.decision,
                    }
                    for r in entry.validation_history
                ],
            })
        return audit

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.all_entries():
                handle.write(entry.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "PersistentMemoryBank":
        bank = cls()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = PersistentMemoryEntry.model_validate(json.loads(line))
                bank._entries[entry.memory_id] = entry
        return bank


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
