"""Persistent memory bank for long-term knowledge accumulation (Task 1).

Lifecycle:

    experience -> candidate memory -> TCI validation
        -> validated (persistent knowledge) | rejected

The bank is an independent module: it never touches
``SharedMemoryPool`` / ``SQLiteSharedMemoryRepository`` and keeps its own
JSONL persistence so lifelong experiments can survive across episodes.
"""

import json
from pathlib import Path

from smtr.memory.memory_schema import PersistentMemoryEntry, utc_now


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

    def validate_memory(self, memory_id: str, tci_effect: float) -> PersistentMemoryEntry:
        """Mark a memory validated after positive TCI evidence (delta > 0)."""
        return self._transition(memory_id, "validated", tci_effect)

    def reject_memory(self, memory_id: str, tci_effect: float) -> PersistentMemoryEntry:
        """Mark a memory rejected after non-positive TCI evidence."""
        return self._transition(memory_id, "rejected", tci_effect)

    def _transition(
        self, memory_id: str, status: str, tci_effect: float
    ) -> PersistentMemoryEntry:
        entry = self.get(memory_id)
        updated = entry.model_copy(
            update={
                "status": status,
                "tci_effect": tci_effect,
                "validation_count": entry.validation_count + 1,
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
