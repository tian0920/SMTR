"""Heuristic memory management controller (ACL 2026).

Reference:
    How Memory Management Impacts LLM Agents: An Empirical Study of
    Experience-Following Behavior.

Core idea:
    memory -> importance score -> keep / delete

    importance = 0.5 * recency + 0.3 * usage_frequency + 0.2 * retrieval_success

Fairness constraints:
    - No TCI (reward intervention).
    - No causal validation.
    - Same memory budget as SMTR when capacity is set.
    - Pure heuristic scoring — no learned parameters.
"""

from __future__ import annotations

from typing import Any

from smtr.baselines.base_memory_controller import (
    BaseMemoryController,
    CandidateMemory,
    MemoryQuery,
)


class HeuristicMemoryController(BaseMemoryController):
    """Importance-scored memory management baseline.

    Each stored memory carries a running importance score that blends
    recency, usage frequency, and retrieval success rate.  When the
    bank exceeds the budget the lowest-scored memory is evicted.
    """

    # Score weights (from the paper heuristic)
    W_RECENCY = 0.5
    W_USAGE = 0.3
    W_RETRIEVAL = 0.2

    def __init__(self, *, budget: int | None = None) -> None:
        self._store: list[CandidateMemory] = []
        self._topic_index: dict[int, list[CandidateMemory]] = {}
        self._budget = budget
        # Per-memory tracking
        self._usage_count: dict[str, int] = {}
        self._retrieval_hits: dict[str, int] = {}
        self._retrieval_total: dict[str, int] = {}
        self._latest_episode: int = 0
        # Counters
        self._deleted_count = 0
        self._retained_count = 0

    # ------------------------------------------------------------------
    # BaseMemoryController interface
    # ------------------------------------------------------------------
    def extract_memory(self, trajectory: dict[str, Any]) -> list[CandidateMemory]:
        """Extract one memory from the trajectory (same as full_memory)."""
        episode = trajectory["episode"]
        topic = trajectory["topic"]
        success = trajectory["success"]
        reward = trajectory.get("reward", float(success))
        content = trajectory.get("content", "")

        memory_id = f"heuristic_ep{episode}_t{topic}"
        candidate = CandidateMemory(
            memory_id=memory_id,
            type="heuristic",
            content=content or f"procedure for topic {topic}",
            source_episode=episode,
            metadata={"topic": topic, "success": success, "reward": reward},
        )
        self._latest_episode = max(self._latest_episode, episode)
        return [candidate]

    def update_memory(
        self,
        candidate: CandidateMemory,
        context: dict[str, Any],
    ) -> str:
        """Score and store; evict lowest-scored if over budget."""
        self._store.append(candidate)
        topic = candidate.metadata.get("topic", -1)
        self._topic_index.setdefault(topic, []).append(candidate)
        self._usage_count[candidate.memory_id] = 0
        self._retrieval_hits[candidate.memory_id] = 0
        self._retrieval_total[candidate.memory_id] = 0
        self._retained_count += 1

        # Evict if over budget
        if self._budget is not None and len(self._store) > self._budget:
            self._evict_lowest()

        return "store"

    def retrieve_memory(self, query: MemoryQuery) -> list[CandidateMemory]:
        """Retrieve by importance score, topic-filtered."""
        self._latest_episode = max(self._latest_episode, query.episode)
        candidates = self._topic_index.get(query.topic, [])

        scored: list[tuple[float, CandidateMemory]] = []
        for mem in candidates:
            score = self._compute_score(mem)
            scored.append((score, mem))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [mem for _, mem in scored[: query.top_k]]

        # Track usage and retrieval success
        for mem in selected:
            self._usage_count[mem.memory_id] = self._usage_count.get(mem.memory_id, 0) + 1
            self._retrieval_total[mem.memory_id] = self._retrieval_total.get(mem.memory_id, 0) + 1
            # Assume retrieved memories are "successful" (used by agent)
            self._retrieval_hits[mem.memory_id] = self._retrieval_hits.get(mem.memory_id, 0) + 1

        return selected

    def get_statistics(self) -> dict[str, Any]:
        return {
            "retained_memory": len(self._store),
            "deleted_memory": self._deleted_count,
            "memory_turnover": self._deleted_count + self._retained_count,
            "unique_topics": len(self._topic_index),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_score(self, memory: CandidateMemory) -> float:
        """Composite importance score."""
        # Recency: normalised by latest episode (1.0 for most recent)
        max_ep = max(self._latest_episode, 1)
        recency = memory.source_episode / max_ep

        # Usage frequency: log-scaled count
        usage = self._usage_count.get(memory.memory_id, 0)
        usage_norm = min(usage / 5.0, 1.0)  # cap at 5 uses

        # Retrieval success rate
        total = self._retrieval_total.get(memory.memory_id, 0)
        hits = self._retrieval_hits.get(memory.memory_id, 0)
        retrieval = hits / total if total > 0 else 0.5  # prior

        return (
            self.W_RECENCY * recency
            + self.W_USAGE * usage_norm
            + self.W_RETRIEVAL * retrieval
        )

    def _evict_lowest(self) -> None:
        """Remove the memory with the lowest importance score."""
        if not self._store:
            return
        worst_idx = min(
            range(len(self._store)),
            key=lambda i: self._compute_score(self._store[i]),
        )
        evicted = self._store.pop(worst_idx)
        topic = evicted.metadata.get("topic", -1)
        topic_list = self._topic_index.get(topic, [])
        self._topic_index[topic] = [m for m in topic_list if m.memory_id != evicted.memory_id]
        # Clean up tracking
        self._usage_count.pop(evicted.memory_id, None)
        self._retrieval_hits.pop(evicted.memory_id, None)
        self._retrieval_total.pop(evicted.memory_id, None)
        self._deleted_count += 1
