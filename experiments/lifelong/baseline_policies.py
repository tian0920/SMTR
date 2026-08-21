"""Baseline memory policies for lifelong experiments.

Wraps each :class:`BaseMemoryController` implementation as a
:class:`LifelongPolicy` so it can be compared head-to-head with the
existing methods (``no_memory``, ``full_memory``, ``retrieval``,
``smtr_tci``) under the same task stream, seed and evaluation.

All baselines share:
  - Same task stream (paired design).
  - Same seed (environment RNG).
  - Same evaluation (success probability model).
  - Same memory budget when ``capacity`` is set.

No baseline uses TCI, gradient updates, or extra LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from smtr.baselines.agemem.agemem_controller import AgeMemController
from smtr.baselines.agile.agile_controller import AgileController
from smtr.baselines.base_memory_controller import MemoryQuery
from smtr.baselines.heuristic_memory.heuristic_controller import HeuristicMemoryController
from smtr.baselines.reflexion.reflexion_controller import ReflexionController
from smtr.memory.persistent_memory import PersistentMemoryBank

from experiments.lifelong.lifelong_env import (
    LifelongEnvironment,
    StoredMemory,
    TaskSample,
    topic_affinity,
)
from experiments.lifelong.methods import LifelongPolicy

RETRIEVAL_TOP_K = 3


class _BaselinePolicy(LifelongPolicy):
    """Adapter: BaseMemoryController -> LifelongPolicy.

    Bridges the uniform baseline interface to the experiment harness
    by translating between ``StoredMemory`` (env objects) and
    ``CandidateMemory`` (baseline objects).
    """

    controller_name = "base"

    def __init__(self, env: LifelongEnvironment, capacity: int | None = None) -> None:
        # Bypass LifelongPolicy.__init__ to avoid creating an unused
        # PersistentMemoryBank + MemoryAdmissionController.
        self._env = env
        self._capacity = capacity
        self._meta: dict[str, StoredMemory] = {}
        self.bank = PersistentMemoryBank()
        self.admission = None  # type: ignore[assignment]
        self._controller = self._make_controller(capacity)

    def _make_controller(self, capacity: int | None):  # noqa: ANN202
        raise NotImplementedError

    # ------------------------------------------------------------------
    # LifelongPolicy interface
    # ------------------------------------------------------------------
    def select_memories(self, task: TaskSample) -> list[StoredMemory]:
        """Retrieve memories via the controller, map back to StoredMemory."""
        query = MemoryQuery(topic=task.topic, episode=task.episode, top_k=RETRIEVAL_TOP_K)
        retrieved = self._controller.retrieve_memory(query)
        result: list[StoredMemory] = []
        for cand in retrieved:
            stored = self._meta.get(cand.memory_id)
            if stored is not None and topic_affinity(stored.topic, task.topic) > 0:
                result.append(stored)
        return result

    def process_candidate(self, task: TaskSample, candidate: StoredMemory) -> None:
        """Extract -> update via the controller, register in bank.

        Uses the environment's ``candidate.memory_id`` for all downstream
        references so that ``memory_history.jsonl`` and the bank stay in
        sync.
        """
        trajectory = {
            "episode": task.episode,
            "topic": task.topic,
            "success": True,  # unknown at extraction time
            "reward": 0.0,
            "content": candidate.content,
        }
        candidates = self._controller.extract_memory(trajectory)
        context = {
            "episode": task.episode,
            "bank_size": len(self._meta),
            "capacity": self._capacity,
        }
        for cand in candidates:
            # Rewrite the controller-assigned ID to match the env's
            # candidate ID so that memory_history.jsonl and the bank
            # share the same identifier.
            env_id = candidate.memory_id
            # Remove old ID from controller stores, re-add with env ID
            self._remove_from_controller(cand.memory_id)
            import dataclasses
            remapped = dataclasses.replace(cand, memory_id=env_id)
            decision = self._controller.update_memory(remapped, context)
            if decision == "store":
                self._meta[env_id] = candidate
                try:
                    self.bank.add_candidate(
                        memory_id=env_id,
                        content=candidate.content,
                        source_episode=candidate.source_episode,
                        receiver="agent1",
                        created_step=candidate.source_episode,
                    )
                except ValueError:
                    pass  # duplicate

    def _remove_from_controller(self, memory_id: str) -> None:
        """Remove a memory from the controller's internal stores.

        This is needed because extract_memory creates a controller-
        specific ID that must be replaced by the environment ID before
        update_memory is called.
        """
        ctrl = self._controller
        # Remove from _store list
        if hasattr(ctrl, "_store"):
            ctrl._store = [m for m in ctrl._store if m.memory_id != memory_id]
        # Remove from topic index
        if hasattr(ctrl, "_topic_index"):
            for topic in list(ctrl._topic_index.keys()):
                ctrl._topic_index[topic] = [
                    m for m in ctrl._topic_index[topic]
                    if m.memory_id != memory_id
                ]
        # Decrement counters that extract_memory may have incremented
        if hasattr(ctrl, "_topic_counts"):
            for topic in list(ctrl._topic_counts.keys()):
                ctrl._topic_counts[topic] = max(0, ctrl._topic_counts.get(topic, 0))


class ReflexionPolicy(_BaselinePolicy):
    name = "reflexion"

    def _make_controller(self, capacity: int | None) -> ReflexionController:
        return ReflexionController(top_k=RETRIEVAL_TOP_K)


class AgilePolicy(_BaselinePolicy):
    name = "agile"

    def _make_controller(self, capacity: int | None) -> AgileController:
        return AgileController(top_k=capacity)


class HeuristicPolicy(_BaselinePolicy):
    name = "heuristic"

    def _make_controller(self, capacity: int | None) -> HeuristicMemoryController:
        return HeuristicMemoryController(budget=capacity)


class AgeMemPolicy(_BaselinePolicy):
    name = "agemem"

    def _make_controller(self, capacity: int | None) -> AgeMemController:
        return AgeMemController(budget=capacity)


BASELINE_METHODS: dict[str, type[LifelongPolicy]] = {
    "reflexion": ReflexionPolicy,
    "agile": AgilePolicy,
    "heuristic": HeuristicPolicy,
    "agemem": AgeMemPolicy,
}
