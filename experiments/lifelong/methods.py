"""Memory policies for lifelong experiments.

Four methods compared in the long-term knowledge experiments:

  - ``no_memory``:     never inject memory
  - ``full_memory``:   every extracted experience stored permanently
  - ``retrieval``:     store everything, inject only top-k topic matches
  - ``smtr_tci``:      candidate -> TCI gate -> validated memories only

All stateful policies keep lifecycle state in ``PersistentMemoryBank``
(Task 1) and route admission through ``MemoryAdmissionController``
(Task 2) where applicable. No tunable thresholds are introduced: the
gate rule is delta > 0 and the retrieval cap is part of the baseline
definition, not a hyperparameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.persistent_memory import PersistentMemoryBank

from experiments.lifelong.lifelong_env import (
    LifelongEnvironment,
    StoredMemory,
    TaskSample,
    topic_affinity,
)

RETRIEVAL_TOP_K = 3


class LifelongPolicy:
    name = "base"

    def __init__(self, env: LifelongEnvironment, capacity: int | None = None) -> None:
        self._env = env
        self._capacity = capacity
        self._meta: dict[str, StoredMemory] = {}
        self.bank = PersistentMemoryBank()
        self.admission = MemoryAdmissionController(self.bank)

    # ------------------------------------------------------------------
    def select_memories(self, task: TaskSample) -> list[StoredMemory]:
        raise NotImplementedError

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _store(self, candidate: StoredMemory) -> None:
        self._meta[candidate.memory_id] = candidate
        self.bank.add_candidate(
            memory_id=candidate.memory_id,
            content=candidate.content,
            source_episode=candidate.source_episode,
            receiver="agent1",
            created_step=candidate.source_episode,
        )

    def _all_stored(self) -> list[StoredMemory]:
        return [self._meta[e.memory_id] for e in self.bank.all_entries()]

    def _enforce_capacity(self) -> None:
        """FIFO eviction when the bank exceeds its capacity (Task 8)."""
        if self._capacity is None:
            return
        entries = self.bank.all_entries()
        overflow = len(entries) - self._capacity
        for entry in entries[: max(0, overflow)]:
            self.bank._entries.pop(entry.memory_id, None)
            self._meta.pop(entry.memory_id, None)


class NoMemoryPolicy(LifelongPolicy):
    name = "no_memory"

    def select_memories(self, task: TaskSample) -> list[StoredMemory]:
        return []

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        return None  # experience discarded


class FullMemoryPolicy(LifelongPolicy):
    """Every experience is stored permanently and always injected.

    With ``capacity`` set (Task 8 memory-budget experiment) the oldest
    memories are evicted FIFO once the bank is full.
    """

    name = "full_memory"

    def select_memories(self, task: TaskSample) -> list[StoredMemory]:
        return self._all_stored()

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        self._store(candidate)
        self._enforce_capacity()


class RetrievalPolicy(LifelongPolicy):
    """Store everything, inject only the top-k topic-matching memories."""

    name = "retrieval"

    def select_memories(self, task: TaskSample) -> list[StoredMemory]:
        matches = [m for m in self._all_stored()
                   if topic_affinity(m.topic, task.topic) > 0]
        matches.sort(key=lambda m: (topic_affinity(m.topic, task.topic),
                                    m.source_episode), reverse=True)
        return matches[:RETRIEVAL_TOP_K]

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        self._store(candidate)


class SMTRTCIPolicy(LifelongPolicy):
    """TCI-gated persistent knowledge: only validated memories are used.

    Admission rule (no threshold): delta > 0 -> validated, else rejected.
    Re-validation of already-validated knowledge is conservative: a
    single non-positive probe only demotes the memory to ``suspect``;
    rejection requires two consecutive non-positive probes. This guards
    against stochastic probe noise purging genuine long-term knowledge
    while still letting outdated memories fail fast (they probe
    persistently negative).
    """

    name = "smtr_tci"

    def __init__(self, env: LifelongEnvironment, capacity: int | None = None) -> None:
        super().__init__(env, capacity)
        self._suspect_streak: dict[str, int] = {}

    def select_memories(self, task: TaskSample) -> list[StoredMemory]:
        validated = self.bank.retrieve_validated()
        return [self._meta[e.memory_id] for e in validated
                if topic_affinity(self._meta[e.memory_id].topic, task.topic) > 0]

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        self._store(candidate)
        delta = self._env.tci_probe_delta(candidate, episode=task.episode)
        self.admission.admit(
            candidate.memory_id,
            reward_expose=delta,
            reward_withhold=0.0,
            episode_id=task.episode,
        )
        self._revalidate_topic(task)
        self._enforce_capacity()

    def _revalidate_topic(self, task: TaskSample) -> None:
        """Re-run the TCI gate on previously validated same-topic memories.

        Persistent knowledge stays persistent only while it keeps passing
        the gate. A non-positive probe is suspect; two consecutive
        non-positive probes reject (delta <= 0 rule). This is what lets
        SMTR-TCI recover after an environment change: outdated memories
        probe persistently negative and are rejected, so they stop being
        injected.
        """
        for entry in self.bank.retrieve_validated():
            memory = self._meta.get(entry.memory_id)
            if memory is None or topic_affinity(memory.topic, task.topic) == 0:
                continue
            if memory.source_episode == task.episode:
                continue  # the fresh candidate was just admitted
            delta = self._env.tci_probe_delta(memory, episode=task.episode)
            if delta > 0:
                self._suspect_streak[memory.memory_id] = 0
                self.admission.admit(
                    memory.memory_id,
                    reward_expose=delta,
                    reward_withhold=0.0,
                    episode_id=task.episode,
                )
                continue
            streak = self._suspect_streak.get(memory.memory_id, 0) + 1
            self._suspect_streak[memory.memory_id] = streak
            if streak >= 2:
                self.admission.admit(
                    memory.memory_id,
                    reward_expose=delta,
                    reward_withhold=0.0,
                    episode_id=task.episode,
                )  # delta <= 0 -> rejected by the gate rule
            else:
                # suspect but not yet rejected: record the probe only
                self.bank.validate_memory(
                    memory.memory_id, delta,
                    episode_id=task.episode,
                    expose_reward=delta,
                    withhold_reward=0.0,
                    decision="suspect",
                )


METHODS: dict[str, type[LifelongPolicy]] = {
    "no_memory": NoMemoryPolicy,
    "full_memory": FullMemoryPolicy,
    "retrieval": RetrievalPolicy,
    "smtr_tci": SMTRTCIPolicy,
}
