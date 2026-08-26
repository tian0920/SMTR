"""Method-specific persistent memory state (P0-3).

Each method in the online experiment maintains its **own** persistent
memory bank. Methods MUST NOT share mutable state.

The design follows the experimental principle:

    All methods see the same real MARBLE discovery stream,
    but how they persist / retrieve is method-specific.

Methods:
  - no_memory:     Never persists anything.
  - full_memory:   Stores ALL candidates; retrieves ALL history.
  - retrieval:     Stores ALL candidates; retrieves top-k from history.
  - smtr_uniform:  Stores candidates with mean_delta > 0 (global).
  - smtr_receiver: Stores candidates with per-receiver delta > 0.

Scenario boundary:
    All method states reset when the scenario changes.

Usage::

    state = MethodMemoryState(methods=["no_memory", "full_memory", ...])
    # After TCI for task_t:
    state.update_after_tci("full_memory", candidates, records, receiver_ids)
    # Before evaluation for task_{t+1}:
    payloads = state.get_injection_payloads("full_memory", receiver_id="agent1")
"""

from __future__ import annotations

import logging
from typing import Any

from smtr.baselines.base_memory_controller import CandidateMemory
from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.online_receiver_intervention import OnlineValidationRecord
from smtr.memory.persistent_memory import PersistentMemoryBank

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 3


class MethodMemoryState:
    """Isolated persistent state for one method.

    Each instance holds its own ``PersistentMemoryBank`` and
    ``MemoryAdmissionController``. No shared mutable state.

    Attributes:
        method: Method name (e.g. "full_memory", "smtr_receiver").
        bank: Isolated persistent memory bank.
        admission: Admission controller bound to ``bank``.
    """

    def __init__(self, method: str) -> None:
        self.method = method
        self.bank = PersistentMemoryBank()
        self.admission = MemoryAdmissionController(self.bank)
        self._global_step = 0

    def reset(self) -> None:
        """Reset all state (scenario boundary)."""
        self.bank = PersistentMemoryBank()
        self.admission = MemoryAdmissionController(self.bank)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Candidate registration (shared discovery)
    # ------------------------------------------------------------------
    def register_candidates(
        self,
        candidates: list[CandidateMemory],
        receiver_ids: list[str],
    ) -> None:
        """Register discovery candidates in this method's bank."""
        for c in candidates:
            try:
                self.bank.add_candidate(
                    memory_id=c.memory_id,
                    content=c.content,
                    source_episode=c.source_episode,
                    receiver=receiver_ids[0] if receiver_ids else "unknown",
                    created_step=self._global_step,
                )
            except ValueError:
                pass  # duplicate memory_id

    # ------------------------------------------------------------------
    # TCI update (method-specific admission logic)
    # ------------------------------------------------------------------
    def update_after_tci(
        self,
        candidates: list[CandidateMemory],
        records: list[OnlineValidationRecord],
        receiver_ids: list[str],
    ) -> None:
        """Apply TCI validation records to this method's bank.

        Method-specific logic:
          - no_memory: do nothing (never persists)
          - full_memory: store all (register_candidates already did this)
          - retrieval: same as full_memory (store all, retrieve top-k later)
          - smtr_uniform: admit if mean_delta across receivers > 0
          - smtr_receiver: admit per-receiver if delta > 0
        """
        if self.method == "no_memory":
            return

        if self.method in ("full_memory", "retrieval"):
            # full_memory and retrieval store ALL candidates.
            # Admission is not TCI-gated; just validate everything.
            for c in candidates:
                try:
                    self.bank.validate_memory(c.memory_id, 0.0)
                except (KeyError, Exception):
                    pass
            return

        if self.method == "smtr_uniform":
            # mean_delta across all receivers > 0 → validated
            per_memory_deltas: dict[str, list[float]] = {}
            for rec in records:
                if rec.delta is not None:
                    per_memory_deltas.setdefault(rec.memory_id, []).append(rec.delta)
            for c in candidates:
                deltas = per_memory_deltas.get(c.memory_id, [])
                if not deltas:
                    continue
                mean_delta = sum(deltas) / len(deltas)
                try:
                    if mean_delta > 0:
                        self.bank.validate_memory(
                            c.memory_id, mean_delta,
                            episode_id=c.source_episode,
                        )
                    else:
                        self.bank.reject_memory(
                            c.memory_id, mean_delta,
                            episode_id=c.source_episode,
                        )
                except (KeyError, Exception):
                    pass
            return

        if self.method == "smtr_receiver":
            # Per-receiver admission via TCI records
            for rec in records:
                if rec.decision not in ("validated", "rejected"):
                    continue
                try:
                    self.admission.admit_for_receiver(
                        rec.memory_id,
                        receiver_id=rec.receiver_id,
                        reward_expose=rec.normalized_expose_score or 0.0,
                        reward_withhold=rec.normalized_withhold_score or 0.0,
                        episode_id=rec.seed,
                        validation_source="online_counterfactual_rollout",
                    )
                except KeyError:
                    pass
            return

    # ------------------------------------------------------------------
    # Retrieval (method-specific)
    # ------------------------------------------------------------------
    def get_injection_payloads(
        self,
        receiver_id: str,
        *,
        render_fn,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[str]:
        """Return rendered memory payloads for injection into one receiver.

        Parameters:
            receiver_id: Target receiver agent ID.
            render_fn: Callable that renders a PersistentMemoryEntry into text.
            top_k: Max memories for retrieval method.
        """
        if self.method == "no_memory":
            return []

        if self.method == "full_memory":
            entries = self.bank.retrieve_validated()
            return [render_fn(e) for e in entries]

        if self.method == "retrieval":
            # Top-k by tci_effect (or creation order)
            entries = self.bank.retrieve_validated()
            # Sort by tci_effect descending, take top_k
            entries = sorted(
                entries,
                key=lambda e: (e.tci_effect or 0.0),
                reverse=True,
            )[:top_k]
            return [render_fn(e) for e in entries]

        if self.method == "smtr_uniform":
            entries = self.bank.retrieve_validated()
            return [render_fn(e) for e in entries]

        if self.method == "smtr_receiver":
            entries = self.bank.get_receiver_validated_memories(receiver_id)
            return [render_fn(e) for e in entries]

        return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def memory_size(self) -> int:
        """Number of entries in the bank."""
        return len(self.bank.all_entries())

    def validated_size(self) -> int:
        """Number of validated entries."""
        return len(self.bank.retrieve_validated())

    def get_statistics(self) -> dict[str, Any]:
        return self.bank.get_statistics()

    def advance_step(self) -> None:
        self._global_step += 1


class MethodStateContainer:
    """Manages isolated state for ALL methods in one experiment.

    Ensures no shared mutable state between methods.
    """

    def __init__(self, methods: list[str]) -> None:
        self._states: dict[str, MethodMemoryState] = {
            m: MethodMemoryState(m) for m in methods
        }

    def get(self, method: str) -> MethodMemoryState:
        return self._states[method]

    def reset_all(self) -> None:
        """Reset all method states (scenario boundary)."""
        for state in self._states.values():
            state.reset()

    def register_candidates_all(
        self,
        candidates: list[CandidateMemory],
        receiver_ids: list[str],
    ) -> None:
        """Register discovery candidates in all methods' banks."""
        for state in self._states.values():
            state.register_candidates(candidates, receiver_ids)

    def advance_all(self) -> None:
        for state in self._states.values():
            state.advance_step()

    def all_statistics(self) -> dict[str, dict[str, Any]]:
        return {m: s.get_statistics() for m, s in self._states.items()}
