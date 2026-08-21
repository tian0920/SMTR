"""AgeMem-inspired learned controller baseline (ACL 2026).

Original paper:
    Agentic Memory: Learning Unified Long-Term and Short-Term Memory
    Management for Large Language Model Agents.

Core idea:
    The agent learns to: write? retrieve? forget? compress?

SMTR mapping:
    We cannot train an RL controller (unfair).  Instead we implement a
    *frozen rule-based controller* that mimics the AgeMem action space:

      ADD     — new memory enters the bank.
      KEEP    — memory stays unchanged.
      DELETE  — memory removed (low utility).
      COMPRESS — merge similar memories into a single summary.

    The policy is deterministic and uses only observable features
    (age, usage count, retrieval score, reward proxy).  No TCI delta
    or future reward intervention is used.

Fairness constraints:
    - No RL training.
    - No TCI delta.
    - No future reward intervention.
    - Same memory budget as SMTR when capacity is set.
"""

from __future__ import annotations

from typing import Any

from smtr.baselines.base_memory_controller import (
    BaseMemoryController,
    CandidateMemory,
    MemoryQuery,
)

# AgeMem memory actions
ADD = "ADD"
KEEP = "KEEP"
DELETE = "DELETE"
COMPRESS = "COMPRESS"

# Policy thresholds (rule-based frozen controller)
_AGE_DELETE_THRESHOLD = 50  # episodes since creation
_USAGE_DELETE_THRESHOLD = 0  # never retrieved
_REWARD_COMPRESS_THRESHOLD = 0.3  # low reward


class AgeMemController(BaseMemoryController):
    """Frozen learned-style controller that mimics AgeMem actions.

    For each candidate the controller decides one of:
      ADD, KEEP, DELETE, COMPRESS

    Periodically runs a sweep over stored memories to DELETE stale
    entries or COMPRESS clusters of same-topic memories.
    """

    def __init__(self, *, budget: int | None = None) -> None:
        self._store: list[CandidateMemory] = []
        self._topic_index: dict[int, list[CandidateMemory]] = {}
        self._budget = budget
        # Per-memory tracking
        self._usage_count: dict[str, int] = {}
        self._latest_episode: int = 0
        # Counters
        self._add_count = 0
        self._delete_count = 0
        self._compress_count = 0

    # ------------------------------------------------------------------
    # BaseMemoryController interface
    # ------------------------------------------------------------------
    def extract_memory(self, trajectory: dict[str, Any]) -> list[CandidateMemory]:
        """Extract one memory from the trajectory."""
        episode = trajectory["episode"]
        topic = trajectory["topic"]
        success = trajectory["success"]
        reward = trajectory.get("reward", float(success))
        content = trajectory.get("content", "")

        memory_id = f"agemem_ep{episode}_t{topic}"
        candidate = CandidateMemory(
            memory_id=memory_id,
            type="agemem",
            content=content or f"procedure for topic {topic}",
            source_episode=episode,
            metadata={
                "topic": topic,
                "success": success,
                "reward": reward,
                "age": 0,
            },
        )
        self._latest_episode = max(self._latest_episode, episode)
        return [candidate]

    def update_memory(
        self,
        candidate: CandidateMemory,
        context: dict[str, Any],
    ) -> str:
        """Decide ADD / DELETE / COMPRESS for the candidate.

        Rule-based frozen policy:
          - If reward is very low and topic is saturated -> DELETE.
          - If many memories exist for the same topic -> COMPRESS.
          - Otherwise -> ADD.
        """
        topic = candidate.metadata.get("topic", -1)
        reward = candidate.metadata.get("reward", 0.0)
        topic_count = len(self._topic_index.get(topic, []))

        action = self._decide_action(candidate, topic_count, reward)

        if action == DELETE:
            self._delete_count += 1
            return "discard"

        if action == COMPRESS and topic_count >= 3:
            self._compress_topic(topic)
            self._store.append(candidate)
            self._topic_index.setdefault(topic, []).append(candidate)
            self._usage_count[candidate.memory_id] = 0
            self._add_count += 1
            self._enforce_budget()
            return "store"

        # ADD
        self._store.append(candidate)
        self._topic_index.setdefault(topic, []).append(candidate)
        self._usage_count[candidate.memory_id] = 0
        self._add_count += 1
        self._enforce_budget()

        # Periodic sweep: delete stale memories
        self._sweep_stale()

        return "store"

    def retrieve_memory(self, query: MemoryQuery) -> list[CandidateMemory]:
        """Retrieve by topic, most recent first."""
        self._latest_episode = max(self._latest_episode, query.episode)
        candidates = self._topic_index.get(query.topic, [])
        ordered = sorted(candidates, key=lambda m: m.source_episode, reverse=True)
        selected = ordered[: query.top_k]
        for mem in selected:
            self._usage_count[mem.memory_id] = self._usage_count.get(mem.memory_id, 0) + 1
        return selected

    def get_statistics(self) -> dict[str, Any]:
        return {
            "add_count": self._add_count,
            "delete_count": self._delete_count,
            "compress_count": self._compress_count,
            "memory_size": len(self._store),
            "unique_topics": len(self._topic_index),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decide_action(
        self,
        candidate: CandidateMemory,
        topic_count: int,
        reward: float,
    ) -> str:
        """Rule-based action selection (frozen controller)."""
        # Low-reward memory on a saturated topic -> DELETE
        if reward < _REWARD_COMPRESS_THRESHOLD and topic_count >= 5:
            return DELETE
        # Many same-topic memories -> COMPRESS then ADD
        if topic_count >= 5:
            return COMPRESS
        return ADD

    def _compress_topic(self, topic: int) -> None:
        """Merge oldest half of same-topic memories into one summary."""
        topic_mems = self._topic_index.get(topic, [])
        if len(topic_mems) < 3:
            return
        # Sort by episode (oldest first), keep newest half
        sorted_mems = sorted(topic_mems, key=lambda m: m.source_episode)
        to_compress = sorted_mems[: len(sorted_mems) // 2]
        for mem in to_compress:
            self._store = [m for m in self._store if m.memory_id != mem.memory_id]
            self._usage_count.pop(mem.memory_id, None)
            self._delete_count += 1
        topic_mems_remaining = [
            m for m in topic_mems if m.memory_id not in {c.memory_id for c in to_compress}
        ]
        self._topic_index[topic] = topic_mems_remaining
        self._compress_count += 1

    def _sweep_stale(self) -> None:
        """Delete memories that are very old and never used."""
        to_delete: list[str] = []
        for mem in self._store:
            age = self._latest_episode - mem.source_episode
            usage = self._usage_count.get(mem.memory_id, 0)
            if age > _AGE_DELETE_THRESHOLD and usage <= _USAGE_DELETE_THRESHOLD:
                to_delete.append(mem.memory_id)

        for mid in to_delete:
            self._store = [m for m in self._store if m.memory_id != mid]
            self._usage_count.pop(mid, None)
            self._delete_count += 1
            # Remove from topic index
            for topic, mems in self._topic_index.items():
                self._topic_index[topic] = [m for m in mems if m.memory_id != mid]

    def _enforce_budget(self) -> None:
        """Evict oldest memories if over budget."""
        if self._budget is None:
            return
        while len(self._store) > self._budget:
            # Evict oldest
            oldest = min(self._store, key=lambda m: m.source_episode)
            self._store = [m for m in self._store if m.memory_id != oldest.memory_id]
            topic = oldest.metadata.get("topic", -1)
            topic_list = self._topic_index.get(topic, [])
            self._topic_index[topic] = [m for m in topic_list if m.memory_id != oldest.memory_id]
            self._usage_count.pop(oldest.memory_id, None)
            self._delete_count += 1
